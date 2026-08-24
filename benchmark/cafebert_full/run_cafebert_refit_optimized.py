#!/usr/bin/env python3
"""S3-only re-run of the CafeBERT benchmark, using turftopic's refit_transform()
and estimate_components() to avoid redundant work that run_cafebert_full.py pays
for every one of its 60 (variant, seed, k) rows per corpus.

Why this exists
----------------
benchmark/cafebert_full/reference/full_results.csv shows S3 fit_seconds is flat
across n_topics and scales almost perfectly (Pearson r=0.9998) with corpus
vocabulary size, not document count. Root cause (read directly from the
installed turftopic source, .venv/Lib/site-packages/turftopic/models/decomp.py):
SemanticSignalSeparation.fit_transform() ALWAYS re-encodes the entire
vocabulary through the encoder, every call -- there is no parameter to inject
a precomputed vocab_embeddings. The benchmark harness (run_cafebert_smoke.
run_model) creates a brand new SemanticSignalSeparation instance and calls
.fit() independently for every (variant, seed, k) combination, even though the
vocabulary is identical across all of them for a given corpus.

turftopic itself provides two ways to avoid this that the benchmark harness
does not use:
  1. refit_transform(raw_documents, embeddings=..., n_components=k,
     random_state=seed) -- reuses the ALREADY-encoded self.vocab_embeddings
     from a prior fit, only reruns FastICA. Docstring: "significantly faster
     than fitting a new model from scratch."
  2. estimate_components(feature_importance) -- axial/angular/combined are a
     cheap post-hoc reweighting of the SAME axial_components_ (angular is a
     cosine_similarity property, combined is axial**2 * angular); no need to
     fit three separate models for the same (corpus, seed, k).

This script combines both: per corpus, the FIRST (seed, k) pays one real
fit() (vocabulary encoding + FastICA); the remaining 19 combinations use
refit_transform() (FastICA only, vocabulary reused); and all three variants
per combination are derived via estimate_components() from one decomposition
instead of three independent fits.

This does NOT touch benchmark/cafebert_full/reference/ (the audited 480-row
artifact) -- results go to S3_CAFEBERT_RESULTS_DIR (default:
benchmark/cafebert_full/results_refit/). Before running, copy the cached
document-embedding .npy files from reference/representation_cache/ into that
same results dir's representation_cache/ subfolder to skip re-encoding
documents (see run_overnight_refit_experiment.ps1, which does this step).

Usage:
    python -m benchmark.cafebert_full.run_cafebert_refit_optimized
    python -m benchmark.cafebert_full.run_cafebert_refit_optimized --corpora vietnamese-news,visfd
    python -m benchmark.cafebert_full.run_cafebert_refit_optimized --force
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from turftopic import SemanticSignalSeparation

try:
    from . import run_cafebert_full as fullrun
    from . import run_cafebert_smoke as base
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    import run_cafebert_full as fullrun
    import run_cafebert_smoke as base
    from paths import ROOT, RESULTS

CONFIG_PATH = ROOT / "cafebert_full_config.json"
REFERENCE_RESULTS = ROOT / "reference" / "full_results.csv"
VARIANTS = ["axial", "angular", "combined"]


def load_done_keys(path: Path) -> set[tuple[str, str, int, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty or "status" not in frame.columns:
        return set()
    valid = frame.loc[frame["status"].eq("ok"), ["corpus", "model", "seed", "n_topics"]]
    return set(map(tuple, valid.itertuples(index=False, name=None)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", help="Comma-separated corpus ids (default: all four).")
    parser.add_argument("--seeds", help="Comma-separated seeds (default: full sensitivity list 11,29,42,47).")
    parser.add_argument("--topic-counts", help="Comma-separated k values (default: 10,20,30,40,50).")
    parser.add_argument("--force", action="store_true", help="Rerun rows even if already marked ok.")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = fullrun.parse_csv_set(args.seeds, config["seeds_sensitivity"])
    corpora = fullrun.parse_csv_set(args.corpora, config["corpora"])
    topic_counts = fullrun.parse_csv_set(args.topic_counts, config["topic_counts"])
    topn = int(config["top_terms"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / "s3_refit_results.csv"
    summary_path = RESULTS / "s3_refit_speed_summary.json"
    existing_rows: list[dict[str, Any]] = pd.read_csv(result_path).to_dict(orient="records") if result_path.exists() else []
    done = set() if args.force else load_done_keys(result_path)

    print(json.dumps({"corpora": corpora, "seeds": seeds, "topic_counts": topic_counts}), flush=True)

    bundles: dict[str, Any] = {}
    for name in corpora:
        try:
            bundles[name] = base.load_corpus(name, config)
        except Exception as exc:
            # e.g. vntc-it needs `unrar` to extract its source archive; if that's
            # missing, fetch_sources.py never produced VNTC_extracted/ and
            # load_vntc_it() hits an empty frame. Skip this corpus and keep going
            # with whatever loaded, rather than losing the whole overnight run.
            print(json.dumps({"SKIP_CORPUS": name, "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
    if not bundles:
        raise SystemExit("No corpus loaded successfully -- check fetch_sources output above.")
    for name, bundle in bundles.items():
        (RESULTS / f"{name}_manifest.json").write_text(json.dumps(bundle.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"prepared": name, "n_documents": len(bundle.docs)}), flush=True)

    encoder_started = time.perf_counter()
    encoder = base.CafeBERTMeanEncoder(
        str(config["encoder"]["checkpoint"]),
        int(config["encoder"]["max_length"]),
        int(config["encoder"]["batch_size"]),
        str(config["encoder"].get("revision", "")) or None,
    )
    print(json.dumps({"encoder_loaded_seconds": time.perf_counter() - encoder_started}), flush=True)

    old_reference = pd.read_csv(REFERENCE_RESULTS) if REFERENCE_RESULTS.exists() else None

    speed_summary: list[dict[str, Any]] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    for corpus_name, bundle in bundles.items():
        print(f"=== {corpus_name}: {len(bundle.docs)} docs ===", flush=True)
        embeddings, representation_seconds, _ = fullrun.representation_for_corpus(bundle, encoder, config)
        print(json.dumps({"representation_seconds": representation_seconds, "cache_hit": representation_seconds == 0.0}), flush=True)

        vectorizer = base.build_vectorizer(config)
        vectorizer.fit(bundle.docs)
        analyzer = vectorizer.build_analyzer()
        tokenized = [analyzer(document) for document in bundle.docs]
        from gensim.corpora import Dictionary
        dictionary = Dictionary(tokenized)
        word2vec = base.make_word2vec(tokenized)

        model: SemanticSignalSeparation | None = None
        corpus_wall_seconds = 0.0
        corpus_events = 0
        combos = [(seed, k) for seed in seeds for k in topic_counts]
        for seed, n_topics in combos:
            keys_needed = [(corpus_name, f"s3_{v}", seed, n_topics) for v in VARIANTS]
            if not args.force and all(k in done for k in keys_needed):
                print(f"SKIP completed {corpus_name} seed={seed} k={n_topics}", flush=True)
                continue
            try:
                event_started = time.perf_counter()
                if model is None:
                    model = SemanticSignalSeparation(
                        n_components=n_topics, encoder=encoder, vectorizer=clone(vectorizer),
                        random_state=seed, feature_importance="axial",
                    )
                    model.fit(bundle.docs, embeddings=embeddings)
                    strategy = "anchor_fit"
                else:
                    model.refit_transform(bundle.docs, embeddings=embeddings, n_components=n_topics, random_state=seed)
                    strategy = "refit_transform"
                event_seconds = time.perf_counter() - event_started
                corpus_wall_seconds += event_seconds
                corpus_events += 1

                for variant in VARIANTS:
                    variant_started = time.perf_counter()
                    model.estimate_components(variant)
                    topics = base.extract_topics(model.get_topics(), n_topics, topn)
                    variant_seconds = time.perf_counter() - variant_started
                    row = {
                        "corpus": corpus_name, "model": f"s3_{variant}", "seed": seed, "n_topics": n_topics,
                        "n_documents": len(bundle.docs), "strategy": strategy,
                        "shared_event_seconds": event_seconds, "variant_extract_seconds": variant_seconds,
                        "fit_seconds": event_seconds + variant_seconds,
                        "wec_in": base.wec_in(topics, word2vec.wv),
                        "topic_diversity": base.diversity(topics, topn),
                        "c_npmi": base.c_npmi(topics, tokenized, dictionary),
                        "alphabetic_term_rate": base.alpha_rate(topics),
                        "topic_count_returned": len(topics),
                        "min_topic_term_count": min((len(t) for t in topics), default=0),
                        "status": "ok", "error": "",
                    }
                    existing_rows.append(row)
                print(json.dumps({"done": corpus_name, "seed": seed, "n_topics": n_topics, "strategy": strategy, "event_seconds": round(event_seconds, 3)}), flush=True)
            except Exception as exc:  # keep going overnight even if one combo fails
                existing_rows.append({
                    "corpus": corpus_name, "model": "s3_*", "seed": seed, "n_topics": n_topics,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=7),
                })
                print(f"FAILED {corpus_name} seed={seed} k={n_topics}: {exc}", flush=True)
            pd.DataFrame(existing_rows).to_csv(result_path, index=False)

        if old_reference is not None:
            old_corpus_s3 = old_reference.loc[
                old_reference["corpus"].eq(corpus_name) & old_reference["model"].str.startswith("s3_")
            ]
            old_total = float(old_corpus_s3["fit_seconds"].sum())
            entry = {
                "corpus": corpus_name, "old_total_fit_seconds": old_total,
                "new_total_wall_seconds": corpus_wall_seconds, "new_fit_events": corpus_events,
                "speedup_x": (old_total / corpus_wall_seconds) if corpus_wall_seconds else None,
            }
            speed_summary = [entry_ for entry_ in speed_summary if entry_["corpus"] != corpus_name] + [entry]
            summary_path.write_text(json.dumps(speed_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(entry), flush=True)

    print("=== FINAL SPEEDUP SUMMARY ===", flush=True)
    print(json.dumps(speed_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

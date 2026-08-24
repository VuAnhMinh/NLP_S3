#!/usr/bin/env python3
"""Compare the original per-row S3 benchmark (reference/full_results.csv,
independent fit() per variant/seed/k) against the refit_transform()+
estimate_components() optimized re-run (results_refit/s3_refit_results.csv).

Writes a Markdown report meant to be read back when updating report/paper.tex
Muc 4 ("Toc do") with real, re-measured numbers -- not to be pasted verbatim,
since the paper text also needs the honest framing already written there.

Usage:
    python -m benchmark.cafebert_full.compare_refit_speedup
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover
    from paths import ROOT, RESULTS

OLD_PATH = ROOT / "reference" / "full_results.csv"
NEW_PATH = RESULTS / "s3_refit_results.csv"
REPORT_PATH = RESULTS / "REFIT_SPEEDUP_COMPARISON.md"

CORPUS_ORDER = ["vietnamese-news", "visfd", "vi-medical", "vntc-it"]
VARIANT_ORDER = ["s3_axial", "s3_angular", "s3_combined"]


def main() -> None:
    if not OLD_PATH.exists():
        raise SystemExit(f"Missing old reference results: {OLD_PATH}")
    if not NEW_PATH.exists():
        raise SystemExit(f"Missing new refit results: {NEW_PATH} -- run run_cafebert_refit_optimized.py first")

    old = pd.read_csv(OLD_PATH)
    new = pd.read_csv(NEW_PATH)
    # A crash-and-resume can leave duplicate rows for a partially-completed
    # (corpus, seed, n_topics) combo (some variants written, then interrupted,
    # then redone on the next run) -- keep the latest row per key.
    new = new.drop_duplicates(["corpus", "model", "seed", "n_topics"], keep="last")
    new_ok = new.loc[new["status"].eq("ok")].copy()

    # IMPORTANT: only compare corpora present in BOTH old and new (e.g. a run
    # without `unrar` skips vntc-it). Averaging "old" over 4 corpora against
    # "new" over 3 is comparing different populations -- vntc-it's WEC-in is
    # much lower than the other three, so silently dropping it from only one
    # side inflates the apparent "new" mean and looks like a bug that isn't
    # there. This bit it a real one that produced a false alarm once already.
    common_corpora = sorted(set(old["corpus"]) & set(new_ok["corpus"]))
    dropped = sorted(set(old["corpus"]) - set(common_corpora))
    if dropped:
        print(f"NOTE: restricting comparison to corpora present in both: {common_corpora} (dropped from OLD for fairness: {dropped})")
    old = old.loc[old["corpus"].isin(common_corpora)]
    new_ok = new_ok.loc[new_ok["corpus"].isin(common_corpora)]

    lines: list[str] = ["# So sanh toc do: fit doc lap (cu) vs refit_transform (moi)", ""]

    lines.append("## 1. Tong thoi gian S3 fit theo corpus (giay)")
    lines.append("")
    lines.append("| Corpus | Cu (tong 60 fit doc lap) | Moi (tong wall-clock: 1 anchor fit + refit) | Speedup |")
    lines.append("|---|---:|---:|---:|")
    for corpus in CORPUS_ORDER:
        old_corpus = old.loc[old["corpus"].eq(corpus) & old["model"].str.startswith("s3_"), "fit_seconds"]
        new_corpus = new_ok.loc[new_ok["corpus"].eq(corpus)]
        if old_corpus.empty or new_corpus.empty:
            lines.append(f"| {corpus} | (thieu du lieu) | (thieu du lieu) | - |")
            continue
        old_total = float(old_corpus.sum())
        # new_total: shared_event_seconds counted once per (seed,k), not per variant row
        new_events = new_corpus.drop_duplicates(["seed", "n_topics"])
        new_total = float(new_events["shared_event_seconds"].sum()) + float(new_corpus["variant_extract_seconds"].sum())
        speedup = old_total / new_total if new_total else float("nan")
        lines.append(f"| {corpus} | {old_total:.1f} | {new_total:.1f} | {speedup:.1f}x |")

    lines.append("")
    lines.append("## 2. fit_seconds trung binh moi model (cu vs moi, gop ca 4 corpus)")
    lines.append("")
    lines.append("| Model | Cu: fit_seconds TB | Moi: fit_seconds TB (amortized) |")
    lines.append("|---|---:|---:|")
    for model in VARIANT_ORDER:
        old_mean = old.loc[old["model"].eq(model), "fit_seconds"].mean()
        new_mean = new_ok.loc[new_ok["model"].eq(model), "fit_seconds"].mean()
        lines.append(f"| {model} | {old_mean:.2f} | {new_mean:.2f} |")

    lines.append("")
    lines.append("## 3. Metric chat luong: cu vs moi (WEC-in / diversity / C_NPMI trung binh)")
    lines.append("Neu chenh lech lon giua cu/moi o day, co gi do sai trong buoc refit -- KHONG chi dua tren")
    lines.append("toc do la du, phai kiem tra metric van dung truoc khi cap nhat paper.tex.")
    lines.append("")
    lines.append("| Model | WEC-in cu | WEC-in moi | Diversity cu | Diversity moi | C_NPMI cu | C_NPMI moi |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in VARIANT_ORDER:
        o = old.loc[old["model"].eq(model)]
        n = new_ok.loc[new_ok["model"].eq(model)]
        lines.append(
            f"| {model} | {o['wec_in'].mean():.4f} | {n['wec_in'].mean():.4f} "
            f"| {o['topic_diversity'].mean():.4f} | {n['topic_diversity'].mean():.4f} "
            f"| {o['c_npmi'].mean():.4f} | {n['c_npmi'].mean():.4f} |"
        )

    failed = new.loc[new["status"].ne("ok")] if "status" in new.columns else pd.DataFrame()
    lines.append("")
    lines.append(f"## 4. Hang loi (status != ok): {len(failed)}")
    if not failed.empty:
        lines.append("")
        lines.append(failed[["corpus", "seed", "n_topics", "error"]].to_markdown(index=False))

    speed_summary_path = RESULTS / "s3_refit_speed_summary.json"
    if speed_summary_path.exists():
        lines.append("")
        lines.append("## 5. Tom tat speedup theo corpus (ghi truc tiep tu script chay)")
        lines.append("")
        lines.append("```json")
        lines.append(speed_summary_path.read_text(encoding="utf-8"))
        lines.append("```")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Da ghi: {REPORT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

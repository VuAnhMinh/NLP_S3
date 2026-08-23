---
name: s3-repro-check
description: Use for anything touching s3_reproduction/ or demo/ — checking the S³ (Semantic Signal Separation) Vietnamese reproduction pipeline can run, training/retraining, choosing an encoder, or using the analysis tools (full-vocabulary word+score reports, ViSFD aspect-label validation/AUC, crisis-monitoring dashboard). Verifies venv, deps, dataset files, GPU, then explains cli.py/inspect.py/validate_visfd.py/monitor.py/demo output.
---

# S3 reproduction: pipeline, encoders, and the analysis toolkit built on top

Context: `s3_reproduction/` reimplements S³ (ACL 2025, Semantic Signal
Separation via FastICA) on Vietnamese text, plus a substantial analysis layer
built on top of it that the paper itself doesn't cover (see `2025.acl-long.32.pdf`
§3.1 for the algorithm, §8.6 for what the paper explicitly declines to
evaluate — document-topic proportions for downstream classification, which
`validate_visfd.py`/`monitor.py` do anyway as a deliberate extension, not a
misreading of the paper). `REPRODUCE.md` has the original install notes but
is stale on encoder choice — see below.

## Chosen encoder: use `--encoder e5`, not the CLI default

`cli.py --encoder` defaults to `cafebert` for backward compatibility with
existing checkpoints, but **CafeBERT is no longer the recommended encoder**.
It's a masked-LM (XLM-R continued-pretrained), not trained as a sentence
embedder, and needs hand-rolled masked-mean pooling (`encoder.py`
`CafeBERTEncoder`). Measured on ViSFD (11,122 phone reviews) against the
dataset's own 11 ground-truth aspect labels via `validate_visfd.py`:

| Encoder | avg single-axis AUC | avg combined-axes AUC | encode time (11k docs) |
|---|---|---|---|
| `uitnlp/CafeBERT` (masked-LM, default) | 0.689 | 0.867 | 34.9s |
| `intfloat/multilingual-e5-base` (`--encoder e5`) | **0.722** | 0.887 | **21.9s** |
| `BAAI/bge-m3` (`--encoder e5 --encoder-model BAAI/bge-m3`) | 0.719 | **0.898** | 69.6s |

E5 and bge-m3 are close in quality and both clearly beat CafeBERT; E5 wins on
speed (3.2x faster than bge-m3), so it's the project default going forward.
Always pass `--encoder e5` explicitly until the CLI default itself is changed.
`--encoder-model <any sentence-transformers checkpoint>` swaps the underlying
model freely (`encoder.py` `default_prefix()` auto-detects the "query: " E5
prefix by checking for "e5" in the name; anything else gets no prefix —
**do not** point this at a PhoBERT-based model like
`VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` without first word-segmenting
the document text fed to `encoder.encode()` — PhoBERT was pretrained on
segmented text, unlike CafeBERT/E5/bge-m3's generic multilingual subword
tokenizers which don't care).

## Checklist before claiming "it can run"

Don't answer from memory of a previous check — deps, dataset files, and
installed packages drift. Re-verify each time:

1. **Venv**: `.venv/Scripts/python.exe` (Windows). Check `pip show` for
   pinned versions against `requirements.txt` if in doubt.
2. **turftopic + pyvi + sentence-transformers installed?** `pip show turftopic
   pyvi sentence-transformers` — all three are load-bearing (default backend,
   Vietnamese segmentation, and the E5/bge-m3 encoder path respectively) and
   none are part of a generic ML env by default.
3. **Dataset files present** (check files, not folders):
   - `dataset/ViSFD/ViSFD.csv` — `data.py` `load_visfd` hard-requires this.
   - `dataset/Vietnamese-News/all/*.parquet` — `load_vietnamese_news` globs
     whatever shards exist; partial sets still work if `--max-documents` fits.
   - `VTSNLP/vietnamese_curated_dataset` (12.17M docs, 34.65GB, 132 shards) —
     `data.py` `load_vietnamese_curated` pulls shards on demand from the HF
     Hub (needs `HF_TOKEN` in env for reasonable rate limits), does **not**
     require pre-downloading; blocks `--max-documents 0` outright (too big).
   - `dataset/` is untracked by git — these arrive by manual copy/download,
     not `git pull`. No downloader is checked in for ViSFD/Vietnamese-News.
4. **GPU**: `nvidia-smi` — check VRAM vs `--batch-size`. CafeBERT/E5/bge-m3
   all run fp16 on CUDA by default when available.
5. **`artifacts/` is gitignored** — its presence only proves a prior run
   happened *somewhere*, not that this checkout can reproduce it. Some
   `artifacts/<backend>/<dataset>/*.json` and `models/*.joblib` files ARE
   git-tracked from earlier commits despite the ignore rule (already-tracked
   files aren't retroactively ignored) — `git status` after a run shows
   these as modified, not new; that's expected, not a bug.

## If blocked

Report the specific blocker with the file:line that hard-requires it. Missing
dataset files usually aren't fixable by Claude alone — for ViSFD/Vietnamese-News
surface it to the user; for the VTSNLP curated set, `hf_hub_download` inside
`load_vietnamese_curated` can fetch it directly given `HF_TOKEN`.

## Running it

```
.\.venv\Scripts\python.exe -m s3_reproduction.cli --backend turftopic --encoder e5 \
    --dataset visfd --n-topics 10 --max-documents 500 --batch-size 8
```
Smoke test first (small `--max-documents`) before a full `--n-topics 10 20 30 40 50` sweep.

**Vietnamese word segmentation**: `cli.py` runs `pyvi.ViTokenizer.tokenize()`
on every document before `CountVectorizer.fit()` — Vietnamese is written
syllable-by-syllable, so an unsegmented vectorizer splits real words
("trầy xước", "bảo hành") into meaningless separate syllables. This ONLY
affects vocabulary extraction (the candidate words topics get described
with); the encoder still sees raw, unsegmented document text (fine for
CafeBERT/E5/bge-m3's generic subword tokenizers — see the PhoBERT caveat
above for when this assumption breaks).

**Checkpoint naming**: `checkpoint.py` `save_model` embeds `n_topics` in the
filename (`model_n<N>_HHMM_DDMMYY.joblib`) because a `--n-topics 10 20 30
40 50` sweep fits several models within the same minute — without the
n_topics tag, later values silently overwrote earlier ones (a real bug hit
and fixed mid-project; if you see a `model_HHMM_DDMMYY.joblib` with no `n<N>`
in the name, it predates the fix and only represents whichever n_topics ran
last). `latest.joblib` always holds the most recently fitted model for that
backend/dataset/encoder combo, regardless of which n_topics that was.
Non-default `--encoder`/`--encoder-model` choices get their own dataset
subfolder (`visfd-e5`, `visfd-e5-bge-m3`, ...) so they can never overwrite
another encoder's cached embeddings.

## Output shape

Per `(backend, dataset, n_topics)` run, `cli.py` writes
`artifacts/<backend>/<dataset-or-dataset-encoder>/topics_<N>.json`:
```json
{
  "dataset": "...", "backend": "...", "encoder": "intfloat/multilingual-e5-base",
  "encoder_kind": "e5", "pooling": "sentence-transformer",
  "documents": 0, "vocabulary_size": 0, "n_topics": 0, "top_n": 10,
  "topics": [["từ1", "từ2"]], "topics_negative": [["từ1", "từ2"]],
  "topic_diversity": 0.0, "embedding_coherence": 0.0,
  "timing": {"segment_seconds": 0.0, "embedding_seconds": 0.0, "model_seconds": 0.0}
}
```
`topics_negative` is each axis' negative pole (paper §3.1: "negative
definition" of a topic) — a checkpoint's saved `topics`/`topics_negative`
are only the top 10 words per pole; the full vocabulary always has scores
for every word on every axis (see `inspect.py` below).

`python -m s3_reproduction.visualize` aggregates n_topics ∈ {10,...,50} runs
into `artifacts/visualizations/`: `metrics.csv`, `summary.md`, PNG/PDF plots
(topic tables now show both poles side by side; `s3_timing.png` shows
encode-vs-fit time). Aggregate score = `sqrt(coherence * diversity)`.

## The analysis toolkit (built on top, not part of the paper)

- **`s3_reproduction/inspect.py`** — `rank_vocabulary(ica, vocabulary,
  embeddings, top_k)` re-scores the ENTIRE vocabulary against every axis
  (checkpoints only keep top 10/pole). CLI: `python -m s3_reproduction.inspect
  <checkpoint> --top-k 20` writes a `.report.md` with 20+ words and their
  actual score per pole — enough for a human or an LLM to name an axis with
  confidence; 6-10 bare words usually isn't.
- **`s3_reproduction/validate_visfd.py`** — ViSFD ships 11 human-annotated
  aspect tags (BATTERY, CAMERA, DESIGN, FEATURES, GENERAL, OTHERS,
  PERFORMANCE, PRICE, SCREEN, SER&ACC, STORAGE) that nothing in the paper's
  own benchmark datasets has. `correlate_axes()` gives each axis a
  point-biserial `r` and AUC against each aspect (AUC is the number that
  actually answers "reliable enough to auto-tag with?" — `r` can look
  nontrivial while classes still overlap heavily). `--combined` additionally
  fits a 5-fold-CV logistic regression over ALL axes per aspect — usually
  much higher AUC, because an aspect's signal is often spread across several
  axes rather than isolated in one. Caveat worth repeating to whoever reads
  the numbers: a combined-axes classifier is close to a linear classifier on
  the raw embedding space (FastICA is just a rotation), so a high combined
  AUC mostly reflects encoder quality, not topic-model quality — the
  single-axis AUC is the number closer to the paper's own interpretability
  concern.
- **`s3_reproduction/monitor.py`** — anomaly/crisis detection for a batch of
  NEW, unlabeled comments: `compute_baseline()` gets each axis' mean/std from
  the training corpus, `batch_zscores()` tests whether a new batch's mean
  score per axis is a statistically significant departure (two-sample z-test
  using standard error, not raw std). This is NOT AUC — AUC needs ground-truth
  labels for the batch being scored, which fresh unlabeled comments don't
  have; AUC is used once, offline, only to attach a human-readable aspect
  name to whichever axis fires. With many axes tested at once, low z-thresholds
  flag axes by chance alone (multiple-comparisons problem) — the CLI and demo
  both surface labeled axes first and collapse unlabeled hits into a count,
  since an unlabeled spike isn't actionable even if statistically real.
- **`demo/app.py`** (`make demo` or `streamlit run demo/app.py`) — sidebar
  picks any checkpoint found under `artifacts/*/*/models/*.joblib` (labeled
  with dataset/backend/n_topics/timestamp, reads metadata so old
  cafebert-only checkpoints still load correctly). Three sections: analyze a
  full paragraph or short keyword/concept (both converge on the same
  `render_axis_scores()`), and "Giám sát & Cảnh báo" (the monitor.py-backed
  crisis dashboard, `demo/sample_data/battery_crisis.json` is a canned 51-comment
  simulated scenario for it — not real crawled data).

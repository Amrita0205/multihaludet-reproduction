# MultiHaluDet — Extended Multi-Model Results

**HaluEval (English) · 10,000 samples · NVIDIA L4 (23 GB)**
Run date: 2026-08-19

This document reports an **extension** of MultiHaluDet to additional open-source
LLMs beyond the two in the original release. It is separate from the paper's
headline numbers in [`README.md`](README.md); see [Relation to the paper](#relation-to-the-published-numbers).

---

## Headline result

| Model | Params | Layers | Hidden | **AUC** | F1 | Acc | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`qwen2.5-7b`** | 7.6B | 28 | 3584 | **0.9889** | 0.9778 | 0.9780 | **0.9910** |
| `phi3.5-mini` | **3.8B** | 32 | 3072 | 0.9868 | 0.9758 | 0.9760 | 0.9901 |
| `mistral-7b` | 7.2B | 32 | 4096 | 0.9864 | **0.9784** | **0.9785** | 0.9889 |
| `yi1.5-9b` | 8.8B | 48 | 4096 | 0.9851 | 0.9783 | 0.9785 | 0.9884 |

Test set: 2,000 held-out samples (20% split, `random_state=42`) for every model.

### Two takeaways

**1. The probing method generalizes across model families.**
Four different families — different tokenizers, hidden sizes (3072–4096),
depths (28/32/32/48) and pretraining corpora — all land within **0.38% AUC**
of each other. This supports the claim that the probe reads a general property
of transformer hidden states rather than something specific to Mistral.

**2. Detection quality does not scale with model size.**
The ranking runs roughly *opposite* to parameter count:

- `phi3.5-mini` (**3.8B**) beats `mistral-7b` (7.2B) on both AUC and AP, at
  half the VRAM (7.6 GB vs 14.5 GB) and 10 min of Stage 1 vs 17 min.
- `yi1.5-9b`, the largest and deepest model tested, ranks **last** on AUC.

Practically: a ~4B model is the sensible default for this probe. There is no
accuracy argument for paying 7B–9B inference costs.

> **Caveat.** These are single-seed, single-dataset, single-split results. The
> defensible claim is *"all four models are equivalent within noise"*, **not**
> "Qwen is the best". Ranking a 0.9889 against a 0.9851 on one 2,000-sample
> test split is not statistically meaningful. Repeat across seeds and datasets
> before claiming an ordering.

---

## Per-model detail

Cross-validation AUCs from the meta-learner's 5 folds, plus the out-of-fold
training estimate. Healthy runs show **OOF ≈ test**; a large gap indicates a
validation problem (see [Known issue](#known-issue-cv-seed-collision)).

| Model | Fold val AUCs | OOF AUC | Test AUC |
|---|---|---:|---:|
| `qwen2.5-7b` | 0.9882 · 0.9868 · 0.9836 · 0.9867 · 0.9894 | 0.9869 | 0.9889 |
| `phi3.5-mini` | 0.9844 · 0.9869 · 0.9854 · 0.9853 · 0.9909 | 0.9849 | 0.9868 |
| `mistral-7b` | 0.9843 · 0.9867 · 0.9848 · 0.9858 · 0.9879 | 0.9849 | 0.9864 |
| `yi1.5-9b` | 0.9842 · 0.9885 · 0.9813 · 0.9828 · 0.9877 | 0.9826 | 0.9851 |

Confusion matrix, `mistral-7b` (n = 2,000):

```
            pred 0   pred 1
true 0        984       14        998 clean
true 1         29      973       1002 hallucinated
```

---

## Feature extraction health

Every model produced the **same feature shape** despite different depths — the
layer sampler in `src/data/feature_extractor.py` normalizes any depth to
`n_sample_layers = 32`.

| Model | Native layers | `X_seq` shape | Samples | Failed | zero_frac | dead cols |
|---|---:|---|---:|---:|---:|---:|
| `qwen2.5-7b` | 28 (upsampled) | (10000, 32, 12) | 10000 | **0** | 0.0001 | 0/12 |
| `phi3.5-mini` | 32 (exact) | (10000, 32, 12) | 10000 | **0** | 0.0179 | 0/12 |
| `yi1.5-9b` | 48 (downsampled) | (10000, 32, 12) | 10000 | **0** | 0.0051 | 0/12 |
| `mistral-7b` | 32 (exact) | (10000, 32, 12) | 10000 | **0** | n/a¹ | n/a¹ |

¹ `mistral-7b` ran before the health check was added.

`zero_frac` is the fraction of exactly-zero feature entries. It matters because
`safe_stat()` converts `inf`/`NaN` to `0.0` — so a dtype overflow appears as a
flood of zeros rather than as an error. All values here are far below the 0.20
warning threshold. Phi-3.5's 0.0179 is a genuine property of its activations
(its 20-sample pre-flight gave 0.0182), not a numerical problem.

---

## Reproducing

```bash
# 1. Extract hidden-state features (GPU, ~10-17 min per model)
python run_pipeline.py --dataset halueval --lang en --model qwen2.5-7b --stage extract

# 2. Multi-scale training + out-of-fold features (GPU, ~10 min)
python run_pipeline.py --dataset halueval --lang en --model qwen2.5-7b --stage train_oof

# 3. Meta-learner (GPU MLP, ~15 s)
python stage4_mlp.py --dataset halueval --lang en --model qwen2.5-7b

# 4. Comparison table across every model already run
python compare_models.py --dataset halueval --lang en
```

Before a first run with a **new** model, use the pre-flight check — it catches
dtype and compatibility problems in ~2 min instead of after a 17-min Stage 1:

```bash
python verify_model.py --model qwen2.5-7b --n 20
```

### Runtimes (NVIDIA L4, 23 GB)

| Model | Stage 1 | Stage 1 rate | Stages 2–3 | Peak VRAM | Total |
|---|---:|---:|---:|---:|---:|
| `phi3.5-mini` | 10:00 | 16.7 it/s | ~11 min | **7.6 GB** | **21:30** |
| `yi1.5-9b` | ~17:00 | 9.9 it/s | ~11 min | 17.7 GB | 28:19 |
| `qwen2.5-7b` | 17:02 | 11.8 it/s | ~14 min | 15.2 GB | 31:50 |
| `mistral-7b` | 14:34 | 11.4 it/s | ~13 min | 14.5 GB | — |

Stages 2–3 cost roughly the same for every model, because they train on the
identical `(N, 32, 12)` array regardless of the base LLM. Only Stage 1 scales
with model size.

---

## Meta-learner: GPU MLP vs. the original stacking ensemble

The numbers above use a **GPU MLP meta-learner** (`stage4_mlp.py`), not the
paper's `LogOddsStacker`. Reason:

| | Stacking ensemble | GPU MLP |
|---|---|---|
| Wall time (per model) | ~50 min | **51 s** |
| Hardware | CPU (28 cores) | GPU |
| Mistral-7B AUC | see below | 0.9864 |

The ensemble fits **36 models** (6 classifiers × [5 folds + 1 full fit]) and is
bounded by `GradientBoostingClassifier`, which is inherently sequential — 250
boosted trees cannot be parallelized *within* a fit. Parallelizing *across* the
36 fits cut it from ~3 h to ~50 min, but no further.

For comparing many base LLMs, the MLP is the practical choice. Run the full
stacking ensemble once, on whichever model you settle on.

### Known issue: CV seed collision

**`src/ensemble/meta_learner.py` currently produces inverted predictions.** On
`mistral-7b` it reported:

```
AUC: 0.0153,  F1: 0.0000,  Acc: 0.4990
```

An AUC of 0.0153 is not "poor" — it is near-perfectly **backwards**
(`1 − 0.0153 = 0.9847`). The discriminative signal is intact; the meta-learner
learned the wrong sign.

**Cause.** `meta_learner.py:28` builds folds with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)` over the
same `y_train` that Stage 2–3 already split with `n_splits=5, random_state=42`
(`run_pipeline.py:89`). The splits therefore **coincide exactly**. Since
`X_train_deep` is assembled as `oof_features[val_idx] = feats_val`, each row's
576 features come from whichever Stage-2/3 fold-model produced it — and those
five models have unrelated embedding bases. Every validation fold thus contains
only rows from an embedding space the classifier never trained on, so the OOF
predictions are anti-correlated and the logistic meta-learner fits negative
weights.

**Evidence.** The same collision was reproduced and fixed in the MLP path.
Using `random_state=cfg.seed` (colliding):

```
fold AUCs: 0.9742 · 0.5659 · 0.0631 · 0.0389 · 0.2413    OOF 0.4136
```

Changing **only** the CV seed to `cfg.seed + 1000` (decorrelated):

```
fold AUCs: 0.9843 · 0.9867 · 0.9848 · 0.9858 · 0.9879    OOF 0.9849
```

Held-out test AUC barely moved (0.9863 → 0.9864), confirming the problem is
**validation-only** — it corrupts model selection and meta-learner weights, not
the underlying features.

**Suggested fix (not applied):** decorrelate the seed in `meta_learner.py`,
e.g. `random_state=self.seed + 1000`, then re-run `--stage ensemble` and A/B
against these MLP numbers.

---

## Relation to the published numbers

`README.md` reports **98.43** AUROC for HaluEval / Mistral-7B.

- The GPU MLP reproduces this closely: **98.64**.
- The stacking ensemble as currently checked in produced `AUC 0.0153`, whose
  complement is `1 − 0.0153 = ` **98.47** — within 0.04 of the published value.

So the pipeline does reproduce the paper's discriminative performance; the
ensemble stage differs only by a sign flip. Whether the seed collision predates
the published experiments or was introduced afterwards has **not** been
determined, and someone with the original run history should check before
drawing conclusions.

---

## Environment

```
GPU            NVIDIA L4, 23 GB          (JarvisLabs VM, 28 vCPU, 121 GB RAM)
OS             Ubuntu 22.04.5 LTS
Python         3.10.12
torch          2.11.0+cu128
transformers   5.15.1
accelerate     1.14.0
scikit-learn   1.7.2
numpy          2.2.6
```

Dataset: `pminervini/HaluEval`, config `qa_samples` — 10,000 samples,
5,010 hallucinated / 4,990 clean. Downloaded automatically for `--lang en`.

> `--lang fr|bn|am` require translated `data/halueval_<lang>.json` files that
> are **not** in this repository (`src/data/loader.py` raises `FileNotFoundError`).

---

## Output files

```
results/
├── features/   X_seq_halueval_en_<model>.npy      (10000, 32, 12)
│               X_glob_halueval_en_<model>.npy     (10000, 30)
│               y_halueval_en_<model>.npy          (10000,)
├── oof/        X_train_deep_… X_test_deep_… y_train_… y_test_…
├── plots/      results_MLP_halueval_en_<model>.png
└── probs_MLP_halueval_en_<model>.npy
```

Plots for each model (ROC, precision-recall, calibration, confusion matrix,
score distribution, confidence) are committed under `figures/`:
[Qwen2.5-7B](figures/results_MLP_halueval_en_qwen2.5-7b.png) ·
[Phi-3.5-mini](figures/results_MLP_halueval_en_phi3.5-mini.png) ·
[Mistral-7B](figures/results_MLP_halueval_en_mistral-7b.png) ·
[Yi-1.5-9B](figures/results_MLP_halueval_en_yi1.5-9b.png)

For the code changes behind these results, see
[`CHANGE_FROM_ORIGINAL_REPO.md`](CHANGE_FROM_ORIGINAL_REPO.md).

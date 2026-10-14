# Changes from the Original Repository

Every modification made to the upstream MultiHaluDet release, and why.
Date: **2026-08-19**. Baseline commit: `c759751` ("Update citation to official
ACL Anthology BibTeX").

Results produced with these changes: [`RESULTS_README.md`](RESULTS_README.md).

**Summary**

| # | Category | File | Status |
|---|---|---|---|
| 1 | 🐞 Bug fix | `src/utils/visualization.py` | applied |
| 2 | 🐞 Bug fix | `requirements.txt` | applied |
| 3 | ✨ Feature | `src/config.py` | applied |
| 4 | ✨ Feature + 🛡 Safety | `run_pipeline.py` | applied |
| 5 | ⚡ Performance | `src/ensemble/meta_learner.py` | on the run VM only — not in this commit |
| 6 | ➕ New file | `stage4_mlp.py` | added |
| 7 | ➕ New file | `verify_model.py` | added |
| 8 | ➕ New file | `compare_models.py` | added |
| 9 | ⚠️ Known issue | `src/ensemble/meta_learner.py` | **NOT fixed** |

`README.md` keeps the original text unchanged; a short "What I added" section was
prepended above it.

---

## 1. 🐞 `src/utils/visualization.py` — wrong import crashed every stage

`calibration_curve` was imported from `sklearn.metrics`, but it lives in
`sklearn.calibration`. Because `run_pipeline.py:18` imports this module at top
level, **the entire pipeline died with `ImportError` before Stage 1 began** — on
any modern scikit-learn, for every stage.

```diff
- from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve
+ from sklearn.metrics import roc_curve, precision_recall_curve
+ from sklearn.calibration import calibration_curve
```

This is a hard blocker on a fresh install. Verified against scikit-learn 1.7.2.

---

## 2. 🐞 `requirements.txt` — missing `accelerate`

`run_pipeline.py:36` passes `device_map="auto"` to `from_pretrained`, which
requires `accelerate`. It was absent from requirements, so a clean install
raised `ValueError` at Stage 1.

```diff
  scipy
+ accelerate
```

---

## 3. ✨ `src/config.py` — model registry expanded, per-model load options

### Registry: 2 → 11 models

All additions are **<10B and ungated** (no HF token required). `mistral-7b`
and `llama2-7b` are unchanged.

```python
MODEL_REGISTRY = {
    "mistral-7b"    : "mistralai/Mistral-7B-Instruct-v0.2",   # unchanged
    "llama2-7b"     : "meta-llama/Llama-2-7b-hf",             # unchanged
    "qwen2.5-7b"    : "Qwen/Qwen2.5-7B-Instruct",
    "phi3.5-mini"   : "microsoft/Phi-3.5-mini-instruct",
    "yi1.5-9b"      : "01-ai/Yi-1.5-9B-Chat",
    "qwen3-8b"      : "Qwen/Qwen3-8B",
    "qwen2.5-3b"    : "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-1.5b"  : "Qwen/Qwen2.5-1.5B-Instruct",
    "zephyr-7b"     : "HuggingFaceH4/zephyr-7b-beta",
    "olmo2-7b"      : "allenai/OLMo-2-1124-7B-Instruct",
    "internlm2.5-7b": "internlm/internlm2_5-7b-chat",
}
```

Argparse builds `--model` choices from these keys, so new entries appear in the
CLI automatically. **Tested:** `qwen2.5-7b`, `phi3.5-mini`, `yi1.5-9b`.
**Untested:** the remaining five.

### New: `MODEL_OVERRIDES` + `get_model_load_kwargs()`

`Config.model_dtype` hardcoded `torch.float16`. Every newly added checkpoint is
natively **bfloat16**, and loading bf16 weights as fp16 can overflow to
`inf`/`NaN`.

That failure would be **silent**: `safe_stat()` in `feature_extractor.py`
converts non-finite values to `0.0`, so a broken model yields near-zero features
and a plausible-but-meaningless AUC instead of an error. Per-model dtype is
therefore a correctness requirement, not a tuning knob.

```python
MODEL_OVERRIDES = {
    "_default"      : {"dtype": torch.bfloat16},
    "mistral-7b"    : {"dtype": torch.float16},   # keep published baseline reproducible
    "llama2-7b"     : {"dtype": torch.float16},
    "internlm2.5-7b": {"trust_remote_code": True},  # untested
}

def get_model_load_kwargs(alias, config):
    """Load options for `alias`: _default overridden by the model entry."""
```

`MODEL_REGISTRY` keeps its original `alias -> hf_id` shape so nothing
downstream breaks. `Config.model_dtype` is untouched and still used as the
fallback — **the two original models load exactly as before.**

---

## 4. ✨🛡 `run_pipeline.py` — per-model loading, and no more silent failures

### 4a. Honour per-model load options

```python
load_kw = get_model_load_kwargs(args.model, config)
trc = load_kw["trust_remote_code"]
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=trc)
model_llm = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=load_kw["dtype"],
    device_map="auto", trust_remote_code=trc)
```

Also: `tokenizer.pad_token` is only overwritten when it is `None` (some models
ship a real pad token), and the model's depth is logged:
`Model depth: 28 layers -> sampling 32`.

> **`torch_dtype=` is kept deliberately, not renamed to `dtype=`.** transformers
> 5.x warns that `torch_dtype` is deprecated, but still honours it (verified: it
> produced correctly-typed fp16 parameters). `dtype=` does **not** exist in
> transformers 4.x. `torch_dtype` is the version-portable choice.

### 4b. 🛡 The silent-failure fix (most important change here)

The original swallowed **every** per-sample exception with no record:

```python
except Exception as e:
    continue        # original
```

If a model was incompatible, every sample failed silently and the run saved an
empty or truncated feature array — surfacing much later as a confusing shape
error, or not at all.

Replaced with failure accounting and a hard stop:

```python
except Exception as e:
    fail_kinds[type(e).__name__] += 1
    if sum(fail_kinds.values()) <= 3:
        print(f"  sample {i} failed: {type(e).__name__}: {e}")
    continue
...
print(f"Extracted {len(all_seq)}/{len(samples)} samples (failed: {n_fail})")
if not all_seq:
    raise RuntimeError("Every sample failed feature extraction - aborting.")
if n_fail > 0.05 * len(samples):
    raise RuntimeError(f"{n_fail}/{len(samples)} samples failed (>5%) ...")
```

**This immediately caught a real bug.** `phi3.5-mini` failed all 20 pre-flight
samples with `AttributeError: type object 'DynamicCache' has no attribute
'from_legacy_cache'` — its bundled remote code targets transformers 4.x. Under
the original bare `except`, this would have produced silent garbage. Fixed by
dropping `trust_remote_code` for that model so transformers' **native** Phi-3
implementation is used instead.

### 4c. 🛡 Feature health check before saving

```python
zero_frac = float((X_seq == 0).mean())
dead_cols = int((X_seq.reshape(-1, X_seq.shape[-1]).std(axis=0) == 0).sum())
print(f"Feature health: shape={X_seq.shape}, zero_frac={zero_frac:.4f}, "
      f"dead_cols={dead_cols}/{X_seq.shape[-1]}")
if zero_frac > 0.20 or dead_cols:
    print("  WARNING: suspicious features - likely inf/NaN from a dtype mismatch.")
```

Catches the `safe_stat()` NaN→0.0 masking described in §3. Observed healthy
values: 0.0001 (Qwen), 0.0051 (Yi), 0.0179 (Phi-3.5).

---

## 5. ⚡ `src/ensemble/meta_learner.py` — parallelized `fit()`

> **Not included in this repository yet.** This change was made on the run VM; the
> `meta_learner.py` committed here is still the original. The description is kept
> so the runtime numbers below can be traced.

Stage 4 fits **36 models** (6 classifiers × [5 CV folds + 1 full-data fit])
sequentially on a single core. Measured: ~50 min on a 28-core machine at ~3.5%
CPU utilization; projected ~1.5–3 h under contention.

The 36 fits are mutually independent, so they now run through `joblib.Parallel`:

```python
jobs = []
for i, (name, clf) in enumerate(self.base_classifiers):
    for train_idx, val_idx in splits:
        jobs.append(delayed(_fit_fold)(clf, X, y, train_idx, val_idx, i))
    jobs.append(delayed(_fit_full)(clf, X, y, i))

for res in Parallel(n_jobs=-1, verbose=10)(jobs):
    ...
```

**Results are identical.** The original refits the *same* estimator object in a
loop; sklearn's `fit()` fully resets estimator state by contract, so fitting a
`clone()` in a worker process is equivalent. Same `random_state`, same
`StratifiedKFold` splits, same data — only execution order changes.

Module-level `_fit_fold` / `_fit_full` helpers exist because joblib's loky
backend must pickle the work functions (closures are not picklable).

Original preserved at `src/ensemble/meta_learner.py.orig` on the run VM.

**Note on the remaining bottleneck:** this does *not* make Stage 4 fast.
`GradientBoostingClassifier` builds 250 boosted trees sequentially — each fits
the previous trees' residuals — so a single fit cannot be parallelized
internally. It dominates the remaining ~50 min. Set `OMP_NUM_THREADS=1` when
running, or the 28 worker processes each spawn thread pools (observed load
average 140 on 28 cores).

---

## 6. ➕ `stage4_mlp.py` (new) — GPU meta-learner

A ~60× faster alternative to the stacking ensemble: a small MLP
(576→512→256→64→1, BatchNorm + GELU + dropout) trained per CV fold on GPU,
test predictions averaged across folds.

**51 seconds** vs the ensemble's ~50 minutes, at equivalent accuracy
(Mistral-7B: MLP 0.9864). Intended for comparing many base LLMs; run the full
stacking ensemble once on the model you finally choose.

```bash
python stage4_mlp.py --dataset halueval --lang en --model qwen2.5-7b
```

Writes `results/plots/results_MLP_<tag>.png` and `results/probs_MLP_<tag>.npy`
— distinct filenames, so it never overwrites the original Stage 4 outputs.

> **This is a different method from the paper's `LogOddsStacker`.** Do not
> report its numbers as reproductions of the published ensemble.

It uses `random_state = cfg.seed + 1000` deliberately — see §9.

---

## 7. ➕ `verify_model.py` (new) — pre-flight check

Loads a model exactly as Stage 1 does, extracts features from ~20 samples, and
reports shape, non-finite counts, `zero_frac`, and dead columns.

```bash
python verify_model.py --model qwen2.5-7b --n 20
```

~2 minutes instead of discovering a problem after a 17-minute Stage 1. This is
what caught the Phi-3.5 `DynamicCache` incompatibility (§4b).

---

## 8. ➕ `compare_models.py` (new) — comparison table

Scans `results/probs_MLP_*.npy`, pairs each with its `y_test`, and emits a
markdown table sorted by AUC. Reuses `find_best_thresholds` and `evaluate_all`
from `src/utils/metrics.py` so numbers stay consistent with the pipeline.

```bash
python compare_models.py --dataset halueval --lang en
```

---

## 9. ⚠️ Known issue — CV seed collision (**NOT fixed**)

`src/ensemble/meta_learner.py` currently produces **inverted** predictions.
Left unfixed on purpose: the fix changes the published ensemble's results, so it
should be applied and re-run deliberately rather than slipped in.

On `mistral-7b`, Stage 4 reported:

```
AUC: 0.0153,  F1: 0.0000,  Acc: 0.4990
```

AUC 0.0153 is near-perfectly backwards (`1 − 0.0153 = 0.9847`). The signal is
intact; the meta-learner learned the wrong sign.

**Cause.** `meta_learner.py:28` uses
`StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)` on the same
`y_train` that Stage 2–3 already split with identical parameters
(`run_pipeline.py:89`), so the splits coincide exactly. `X_train_deep` is built
as `oof_features[val_idx] = feats_val`, meaning each row's features come from
one of five Stage-2/3 fold-models with unrelated embedding bases. Every
validation fold therefore holds only rows from an embedding space the
classifier never saw — producing anti-correlated OOF log-odds, from which the
logistic meta-learner fits negative weights.

**Evidence** (reproduced in the MLP path). With the colliding seed:

```
fold AUCs: 0.9742 · 0.5659 · 0.0631 · 0.0389 · 0.2413    OOF 0.4136
```

Changing **only** the CV seed to `cfg.seed + 1000`:

```
fold AUCs: 0.9843 · 0.9867 · 0.9848 · 0.9858 · 0.9879    OOF 0.9849
```

Held-out test AUC was essentially unchanged (0.9863 → 0.9864) — the defect is
**validation-only**. It corrupts model selection and meta-learner weights, not
the features.

**Suggested fix, deliberately not applied:**

```python
# src/ensemble/meta_learner.py:28
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed + 1000)
```

Then re-run `--stage ensemble` and A/B against the MLP numbers.

> `README.md` reports **98.43** AUROC for HaluEval/Mistral-7B, and the broken
> ensemble's complement is **98.47**. Whether the collision predates the
> published experiments has **not** been determined — check the original run
> history before drawing conclusions.

---

## Files touched

```
modified:   requirements.txt                  (+1 line)
modified:   src/config.py                     (registry 2->11, overrides, helper)
(on run VM) src/ensemble/meta_learner.py  (parallel fit - not committed)
modified:   src/utils/visualization.py        (import fix)
modified:   run_pipeline.py                   (load kwargs, failure accounting, health check)
new file:   stage4_mlp.py
new file:   verify_model.py
new file:   compare_models.py
new file:   RESULTS_README.md
new file:   CHANGE_FROM_ORIGINAL_REPO.md
modified:   README.md                         ("What I added" section prepended; original text unchanged)
```

## Backward compatibility

The original command still behaves identically:

```bash
python run_pipeline.py --dataset halueval --model mistral-7b --lang en --stage all
```

`mistral-7b` and `llama2-7b` remain pinned to `torch.float16` and
`trust_remote_code=False`, so their load path is byte-for-byte the original.
Changes 1 and 2 are strictly bug fixes; 3, 4a, 6–8 are additive; 4b/4c add
diagnostics that only raise on genuinely bad runs (>5% sample failures); 5 is a
performance change with identical numerical output.

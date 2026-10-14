# MultiHaluDet: Reproduction and Multi-Model Extension

> Built on [alvi-uiu/MULTIHALUDET](https://github.com/alvi-uiu/MULTIHALUDET), the code for
> *MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing*
> (Alvi et al., MeLLM Workshop @ ACL 2026). The method and the paper are the authors'.
> This repository adds a reproduction, an extension to more model families, and fixes.

## What I added

- **Reproduced** the HaluEval (English) result for Mistral-7B and **extended** the pipeline
  to Qwen2.5-7B, Phi-3.5-mini and Yi-1.5-9B (10,000 samples, NVIDIA L4).
- **Found a cross-validation seed collision** that makes the Stage 4 ensemble's predictions
  come out inverted (AUC 0.0153, i.e. 1 − 0.9847), with the cause, evidence and a fix.
- **Fixed two install blockers** (a wrong `calibration_curve` import, missing `accelerate`).
- **Added** per-model dtype handling, failure accounting and feature health checks
  (so an incompatible model fails loudly), a pre-flight check (`verify_model.py`),
  a GPU meta-learner (`stage4_mlp.py`) and a comparison script (`compare_models.py`).

| Model | Params | AUROC | F1 | AP |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 7.6B | 0.9889 | 0.9778 | 0.9910 |
| Phi-3.5-mini | 3.8B | 0.9868 | 0.9758 | 0.9901 |
| Mistral-7B | 7.2B | 0.9864 | 0.9784 | 0.9889 |
| Yi-1.5-9B | 8.8B | 0.9851 | 0.9783 | 0.9884 |

Single seed, single split: the honest reading is that all four are equivalent within noise.
These numbers use the GPU meta-learner, not the paper's stacking ensemble.

### Where to read next

1. **[RESULTS_README.md](RESULTS_README.md)**: full results, per-fold numbers, runtimes, plots,
   and the seed-collision write-up.
2. **[CHANGE_FROM_ORIGINAL_REPO.md](CHANGE_FROM_ORIGINAL_REPO.md)**: every code change, with
   the reason for it.

### Files I added or changed

```text
run_pipeline.py            # changed: per-model loading, failure accounting, feature health check
src/config.py              # changed: 11-model registry, MODEL_OVERRIDES, get_model_load_kwargs()
src/utils/visualization.py # changed: calibration_curve import fix
requirements.txt           # changed: + accelerate
verify_model.py            # new: 2-minute pre-flight check before a full Stage 1 run
stage4_mlp.py              # new: GPU MLP meta-learner (fast alternative to Stage 4)
compare_models.py          # new: markdown table across every model already run
figures/results_MLP_*.png  # new: result plots for the four models
```

### Quick start (reproduce one model)

```bash
pip install -r requirements.txt
python verify_model.py  --model qwen2.5-7b --n 20                                # pre-flight, ~2 min
python run_pipeline.py  --dataset halueval --lang en --model qwen2.5-7b --stage extract
python run_pipeline.py  --dataset halueval --lang en --model qwen2.5-7b --stage train_oof
python stage4_mlp.py    --dataset halueval --lang en --model qwen2.5-7b         # ~1 min on GPU
python compare_models.py --dataset halueval --lang en
```

— Amrita Kadam

---

*The original project README follows, unchanged.*

# MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing

[![Paper](https://img.shields.io/badge/Paper-ACL_Anthology-red)](https://aclanthology.org/2026.mellm-1.6/)
[![Poster](https://img.shields.io/badge/Poster-PDF-blueviolet)](MultiHaluDet_Poster.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

**MultiHaluDet** is a novel four-stage framework designed to detect multilingual hallucinations by probing the internal hidden state trajectories of frozen Large Language Models (LLMs). Unlike confidence-based heuristics, MultiHaluDet captures the "reasoning process" by analyzing distributional activation patterns across transformer layers.

## Key Framework Stages
1. **Dynamic Layer Probing:** Uniformly samples hidden states from any LLM architecture (Mistral, LLaMA).
2. **Multi-Scale Sequential Modeling:** Processes depth-wise features using a hybrid architecture of Multi-Scale Attention and Transformer Encoders.
3. **Out-of-Fold Feature Generation:** Produces unbiased deep embeddings via K-fold stacking to prevent data leakage.
4. **Log-Odds Ensemble Meta-Learning:** A learned logistic meta-regressor that weighs base classifiers (XGBoost, RF, SVM, etc.) in the log-odds space for robust, calibrated detection.

## Architecture

![MultiHaluDet Framework](figures/architecture.png)

## Performance Summary
| Dataset | Base Model | AUROC (%) |
| :--- | :--- | :---: |
| **HaluEval** | Mistral-7B | 98.43 |
| **HaluEval** | LLaMA2-7B | 98.55 |
| **TriviaQA** | Mistral-7B | 98.30 |
| **TriviaQA** | LLaMA2-7B | 98.26 |

*MultiHaluDet consistently maintains strong performance across High (French), Medium (Bangla), and Low-resource (Amharic) languages without language-specific fine-tuning.*

## Installation

```bash
git clone https://github.com/alvi-uiu/MultiHaluDet.git
cd MultiHaluDet
pip install -r requirements.txt
```

## Usage

Run the 4-stage pipeline sequentially (feature extraction, deep training, and ensemble meta-learning):

```bash
# Detect hallucinations in HaluEval using Mistral-7B (runs all stages)
python run_pipeline.py --dataset halueval --model mistral-7b --lang en --stage all

# Or run stages individually:
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage extract
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage train_oof
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage ensemble
```

### Supported Models
The framework supports any model from the `src/config.py` registry:
- `mistral-7b`
- `llama2-7b`

## Repository Structure
```text
src/
├── config.py              # Hyperparameters & Model Registry
├── data/
│   ├── loader.py          # HaluEval & TriviaQA Data Loading
│   └── feature_extractor.py # LLM Probing & Statistical Feature Extraction
├── models/
│   ├── multihaludet.py    # MultiHaluDet Architecture
│   └── losses.py          # Focal, Asymmetric, & Contrastive Loss
├── training/
│   ├── trainer.py         # OOF training loops & EMA
│   └── augmentations.py   # MixUp & CutMix implementation
├── ensemble/
│   └── meta_learner.py    # Log-Odds Logistic Meta-Regressor
└── utils/
    ├── metrics.py         # Calibration & Evaluation (ECE, AUROC, Brier)
    └── visualization.py   # Plots
```

## Citation
If you find this code or methodology useful, please cite our paper:
```bibtex
@inproceedings{alvi-etal-2026-multihaludet,
    title = "{M}ulti{H}alu{D}et: Multilingual Hallucination Detection via {LLM} Hidden State Probing",
    author = "Alvi, Riasad  and
      Sayeedi, Nurul Labib  and
      Sayeedi, Md. Faiyaz Abdullah",
    editor = "Huang, Kaiyu  and
      Mo, Fengran  and
      Chen, Pinzhen  and
      Jiang, Meng",
    booktitle = "Proceedings of the 1st Workshop on Multilinguality in the Era of Large Language Models ({M}e{LLM} 2026)",
    month = jul,
    year = "2026",
    address = "San Diego, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.mellm-1.6/",
    doi = "10.18653/v1/2026.mellm-1.6",
    pages = "63--74",
    ISBN = "979-8-89176-430-9"
}
```

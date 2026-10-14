import torch

class Config:
    seed = 42
    n_inner_folds = 5
    test_size = 0.20
    
    hidden_dim = 384
    num_heads = 8
    num_layers = 6
    scales = [1, 2, 4, 8]
    mlp_hidden_dim = 288
    dropout = 0.25
    
    batch_size = 28
    epochs = 45
    learning_rate = 2e-4
    weight_decay = 6e-5
    patience = 15
    min_lr = 1e-7
    warmup_epochs = 5
    
    use_mixup = True
    mixup_alpha = 0.15
    use_cutmix = True
    cutmix_alpha = 0.15
    label_smoothing = 0.02
    contrastive_weight = 0.20
    contrastive_temp = 0.04
    grad_clip = 0.5
    
    use_ema = True
    ema_decay = 0.999
    use_swa = True
    swa_start = 30
    swa_lr = 5e-5
    
    use_class_weights = False
    minority_oversample = 1.0
    decision_bias = 0.0
    
    focal_alpha_pos = 0.5
    focal_alpha_neg = 0.5
    focal_gamma = 2.0

    n_sample_layers  = 32
    anchor_fractions = [0.25, 0.50, 0.75, 1.00]
    model_dtype      = torch.float16

MODEL_REGISTRY = {
    "mistral-7b"    : "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2-7b"     : "meta-llama/Llama-2-7b-hf",
    # --- additional open-source probes (all <10B, ungated) ---
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

# Per-model load overrides, merged over "_default".
# Default is bfloat16: every checkpoint above except the two originals is
# natively bf16, and loading bf16 weights as fp16 can overflow to inf/NaN.
# That failure is silent -- safe_stat() in feature_extractor.py turns
# non-finite values into 0.0 -- so the dtype must be set per model.
MODEL_OVERRIDES = {
    "_default"      : {"dtype": torch.bfloat16},
    "mistral-7b"    : {"dtype": torch.float16},   # keep published baseline reproducible
    "llama2-7b"     : {"dtype": torch.float16},
    # phi3.5-mini deliberately has no entry (so no trust_remote_code): its HF
    # repo ships modeling code written for transformers 4.x
    # (DynamicCache.from_legacy_cache, removed in v5). transformers has native
    # Phi-3 support, so that is used instead.
    "internlm2.5-7b": {"trust_remote_code": True},  # untested; may hit the same v5 issue
}

def get_model_load_kwargs(alias, config):
    """Load options for `alias`: _default overridden by the model entry."""
    merged = dict(MODEL_OVERRIDES.get("_default", {}))
    merged.update(MODEL_OVERRIDES.get(alias, {}))
    return {
        "dtype": merged.get("dtype", config.model_dtype),
        "trust_remote_code": merged.get("trust_remote_code", False),
    }

def get_config():
    return Config()

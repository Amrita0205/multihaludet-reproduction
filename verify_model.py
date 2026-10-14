"""
Cheap pre-flight check for a newly registered model.

Loads the model exactly as stage_1_extract does, runs feature extraction on a
handful of samples, and reports shape + feature health. Catches dtype overflow
and layer-count problems in ~2 minutes instead of after a 20-minute Stage 1.
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import get_config, MODEL_REGISTRY, get_model_load_kwargs
from src.data.loader import load_halueval
from src.data.feature_extractor import extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    cfg = get_config()
    cfg.model_name = MODEL_REGISTRY[a.model]
    kw = get_model_load_kwargs(a.model, cfg)
    print(f"model : {cfg.model_name}")
    print(f"load  : dtype={kw['dtype']} trust_remote_code={kw['trust_remote_code']}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=kw["trust_remote_code"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=kw["dtype"], device_map="auto",
        trust_remote_code=kw["trust_remote_code"])
    m.eval()

    n_layers = m.config.num_hidden_layers
    print(f"depth : {n_layers} layers, hidden={m.config.hidden_size}")
    print(f"param dtype: {next(m.parameters()).dtype}")
    print(f"VRAM  : {torch.cuda.memory_allocated()/1e9:.1f} GB")

    samples = load_halueval("en")[:a.n]
    seq, glob, fails = [], [], 0
    for s in samples:
        try:
            x, g = extract_features(s["question"], s["answer"], tok, m, cfg)
            seq.append(x); glob.append(g)
        except Exception as e:
            fails += 1
            print(f"  FAIL {type(e).__name__}: {str(e)[:140]}")

    print()
    print("---- RESULT ----")
    print(f"ok/fail        : {len(seq)}/{fails}")
    if not seq:
        print()
        print("VERDICT: FAIL - every sample errored, see messages above")
        return
    X = np.array(seq); G = np.array(glob)
    print(f"X_seq shape    : {X.shape}   (expected ({len(seq)}, {cfg.n_sample_layers}, 12))")
    print(f"X_glob shape   : {G.shape}   (expected ({len(seq)}, 30))")
    print(f"non-finite     : X_seq={int((~np.isfinite(X)).sum())}  X_glob={int((~np.isfinite(G)).sum())}")
    print(f"zero fraction  : {float((X == 0).mean()):.4f}")
    dead = int((X.reshape(-1, X.shape[-1]).std(axis=0) == 0).sum())
    print(f"dead columns   : {dead}/{X.shape[-1]}")

    ok = (X.shape[1:] == (cfg.n_sample_layers, 12) and G.shape[1:] == (30,)
          and fails == 0 and dead == 0 and float((X == 0).mean()) < 0.20)
    print()
    print("VERDICT: " + ("PASS" if ok else "INVESTIGATE"))


if __name__ == "__main__":
    main()

"""
Build a markdown comparison table across every model that has MLP predictions.

Scans results/probs_MLP_<dataset>_<lang>_<model>.npy and pairs each with the
matching results/oof/y_test_<...>.npy, then reports the same metrics the
pipeline reports, reusing src/utils/metrics.py so numbers stay comparable.
"""
import argparse
import glob
import os
import re
import numpy as np

from src.utils.metrics import find_best_thresholds, evaluate_all


def rows(dataset, lang):
    pat = f"results/probs_MLP_{dataset}_{lang}_*.npy"
    out = []
    for f in sorted(glob.glob(pat)):
        m = re.match(rf"results/probs_MLP_{dataset}_{lang}_(.+)\.npy$",
                     f.replace(os.sep, "/"))
        if not m:
            continue
        model = m.group(1)
        ytest = f"results/oof/y_test_{dataset}_{lang}_{model}.npy"
        if not os.path.exists(ytest):
            print(f"  skip {model}: missing {ytest}")
            continue
        probs = np.load(f)
        y = np.load(ytest)
        if len(probs) != len(y):
            print(f"  skip {model}: length mismatch {len(probs)} vs {len(y)}")
            continue
        th = find_best_thresholds(probs, y)
        r = evaluate_all(probs, y, th["youden"])
        out.append((model, r, len(y)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare probe results across models")
    ap.add_argument("--dataset", default="halueval")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()

    res = rows(a.dataset, a.lang)
    if not res:
        print("No results found. Run stage4_mlp.py for at least one model first.")
        return

    res.sort(key=lambda t: t[1]["auc"], reverse=True)

    print()
    print(f"### {a.dataset} / {a.lang} - probe performance by base LLM")
    print()
    print("| Model | AUC | F1 | Acc | AP | n_test |")
    print("|---|---|---|---|---|---|")
    for model, r, n in res:
        print(f"| `{model}` | {r['auc']:.4f} | {r['f1']:.4f} | "
              f"{r['acc']:.4f} | {r['ap']:.4f} | {n} |")
    print()

    best = res[0]
    print(f"Best: {best[0]} (AUC {best[1]['auc']:.4f})")


if __name__ == "__main__":
    main()

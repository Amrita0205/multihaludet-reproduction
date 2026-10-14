"""
GPU MLP alternative to Stage 4's log-odds stacking ensemble.

NOTE: this is a DIFFERENT method from the paper's LogOddsStacker. It replaces
the 6-classifier CPU stack with a small MLP meta-learner trained on the same
deep OOF features. Use it for speed / comparison, not as a drop-in result.

Reads results/oof/*.npy (read-only) and writes only *_MLP_* filenames, so it
cannot disturb a concurrently running Stage 4.

5-fold CV: one MLP per fold, test predictions averaged across folds -- the same
bagging idea the stacker uses, but on GPU it takes seconds instead of an hour.
"""
import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.config import get_config, MODEL_REGISTRY
from src.utils.metrics import find_best_thresholds, evaluate_all, compute_uncertainty_metrics
from src.utils.visualization import plot_publication_results

EPOCHS = 300
PATIENCE = 40
BATCH = 256
LR = 1e-3
WD = 1e-4


class MetaMLP(nn.Module):
    def __init__(self, d_in, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(p * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(Xtr, ytr, Xva, yva, Xte, device, seed):
    torch.manual_seed(seed)
    model = MetaMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    n = Xtr_t.shape[0]
    best_auc, best_state, bad = -1.0, None, 0

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, va_p)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        te_p = torch.sigmoid(model(Xte_t)).cpu().numpy()
    return best_auc, va_p, te_p


def main(TAG):
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()

    X_train = np.load(f'results/oof/X_train_deep_{TAG}.npy')
    X_test = np.load(f'results/oof/X_test_deep_{TAG}.npy')
    y_train = np.load(f'results/oof/y_train_{TAG}.npy')
    y_test = np.load(f'results/oof/y_test_{TAG}.npy')
    print(f"loaded train={X_train.shape} test={X_test.shape}", flush=True)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train).astype(np.float32)
    X_test = sc.transform(X_test).astype(np.float32)

    # NOTE: deliberately NOT cfg.seed. Stage 2-3 built X_train_deep with
    # StratifiedKFold(5, shuffle=True, random_state=cfg.seed) on this same
    # y_train, so reusing that seed makes every validation fold consist of
    # rows from a single Stage-2/3 embedding space -> val AUC collapses.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed + 1000)
    oof = np.zeros(len(y_train))
    test_acc = np.zeros(len(y_test))

    for k, (tr, va) in enumerate(skf.split(X_train, y_train)):
        fa, vp, tp = train_fold(X_train[tr], y_train[tr], X_train[va], y_train[va],
                                X_test, device, cfg.seed + k)
        oof[va] = vp
        test_acc += tp
        print(f"  fold {k+1}/5  val AUC {fa:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    probs = test_acc / 5.0

    th = find_best_thresholds(probs, y_test)
    res = evaluate_all(probs, y_test, th['youden'])
    preds = (probs >= th['youden']).astype(int)
    unc = compute_uncertainty_metrics(probs, y_test, preds)

    print("\n" + "=" * 54)
    print("  GPU MLP META-LEARNER  (not the paper's stacking ensemble)")
    print("=" * 54)
    print(f"  OOF AUC (train) : {roc_auc_score(y_train, oof):.4f}")
    print(f"  AUC             : {res['auc']:.4f}")
    print(f"  F1              : {res['f1']:.4f}")
    print(f"  Acc             : {res['acc']:.4f}")
    print(f"  AP              : {res['ap']:.4f}")
    print(f"  confusion matrix:\n{res['cm']}")
    print(f"\n  total elapsed   : {time.time()-t0:.1f}s")
    print("=" * 54, flush=True)

    os.makedirs("results/plots", exist_ok=True)
    out = f"results/plots/results_MLP_{TAG}.png"
    plot_publication_results(probs, y_test, preds, unc, res['auc'], res['ap'], res['cm'], out)
    print(f"saved plot -> {out}", flush=True)
    np.save(f"results/probs_MLP_{TAG}.npy", probs)


def parse_args():
    ap = argparse.ArgumentParser(
        description="GPU MLP meta-learner (fast alternative to Stage 4 stacking)")
    ap.add_argument("--dataset", type=str, default="halueval",
                    choices=["halueval", "triviaqa"])
    ap.add_argument("--lang", type=str, default="en",
                    choices=["en", "fr", "bn", "am"])
    ap.add_argument("--model", type=str, default="mistral-7b",
                    choices=list(MODEL_REGISTRY.keys()))
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(f"{a.dataset}_{a.lang}_{a.model}"))

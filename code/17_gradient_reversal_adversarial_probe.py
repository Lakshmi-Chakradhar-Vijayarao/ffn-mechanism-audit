"""
Paper 1 -- elite-review follow-up: runs the gradient-reversal adversarial
probe specified in S3.7 as "a stronger follow-up if these simpler
[difficulty-matched] controls are later contested," rather than leaving
it as a specified-but-unrun protocol.

Shared representation z (a small MLP encoder on the same FFN/Attn
mean-pooled activations used throughout this paper) feeds two heads:
g_hall(z) (hallucination classifier, BCE) and g_diff(z) (entropy-proxy
regressor, MSE) through a gradient-reversal layer, so the encoder is
trained to predict hallucination while ACTIVELY discarding whatever
predicts entropy. If hallucination-AUROC survives above chance under
this adversarial pressure, that AUROC cannot be explained by the
entropy proxy leaking through z -- a strictly stronger claim than the
difficulty-MATCHING control in S3.7, which only showed no detectable
pre-existing confound to remove.

Data: identical to code/06_difficulty_matched_control.py -- GPT-2's
534-sample activations.pkl (external mech-int dependency) for FFN L8 /
Attn L3 mean-pooled last-token outputs, features.npy[:,0] (mean entropy)
as the difficulty proxy, labels.npy as the hallucination label.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "surface_baseline"
MECH_INT = os.path.expanduser(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int/data/processed"))

PEAK_FFN_LAYER = 8
PEAK_ATTN_LAYER = 3
RANDOM_STATE = 42
N_FOLDS = 5
HIDDEN = 64
LAMBDA_ADV = 1.0
EPOCHS = 100
N_PERM = 200


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class AdversarialProbe(nn.Module):
    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.head_hall = nn.Linear(hidden, 1)
        self.head_diff = nn.Linear(hidden, 1)

    def forward(self, x, lambd=LAMBDA_ADV):
        z = self.encoder(x)
        hall_logit = self.head_hall(z).squeeze(-1)
        diff_pred = self.head_diff(grad_reverse(z, lambd)).squeeze(-1)
        return hall_logit, diff_pred


def train_and_eval_fold(X_tr, y_tr, delta_tr, X_te, y_te, delta_te, seed):
    torch.manual_seed(seed)
    model = AdversarialProbe(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    dt = torch.tensor(delta_tr, dtype=torch.float32)

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        hall_logit, diff_pred = model(Xt)
        loss = bce(hall_logit, yt) + mse(diff_pred, dt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(X_te, dtype=torch.float32)
        hall_logit_te, diff_pred_te = model(Xte_t)
        probs = torch.sigmoid(hall_logit_te).numpy()
        diff_pred_np = diff_pred_te.numpy()
    try:
        auc = roc_auc_score(y_te, probs)
    except ValueError:
        auc = 0.5
    try:
        diff_r2 = r2_score(delta_te, diff_pred_np)
    except ValueError:
        diff_r2 = float("nan")
    return auc, diff_r2


def cv_adversarial(X, y, delta, n_splits=N_FOLDS, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, diff_r2s = [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        sc_d = StandardScaler()
        delta_tr = sc_d.fit_transform(delta[tr].reshape(-1, 1)).ravel()
        delta_te = sc_d.transform(delta[te].reshape(-1, 1)).ravel()
        auc, diff_r2 = train_and_eval_fold(X_tr, y[tr], delta_tr, X_te, y[te], delta_te, seed=seed + fold_i)
        aucs.append(auc)
        diff_r2s.append(diff_r2)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(diff_r2s))


def main():
    with open(f"{MECH_INT}/activations.pkl", "rb") as f:
        import pickle
        activations = pickle.load(f)
    y = np.load(VENDORED / "labels.npy")
    X_features = np.load(VENDORED / "features.npy")
    delta = X_features[:, 0]  # mean_entropy, identical proxy to 06_difficulty_matched_control.py
    n = len(y)
    print(f"Loaded N={n}, correct={int(y.sum())}, hallucinated={int((y==0).sum())}")

    X_ffn_full = np.stack([act["ffn_outputs"][PEAK_FFN_LAYER].mean(axis=0) for act in activations])
    X_attn_full = np.stack([act["attn_outputs"][PEAK_ATTN_LAYER].mean(axis=0) for act in activations])

    results = {}
    for name, X in [("FFN", X_ffn_full), ("Attn", X_attn_full)]:
        print(f"\n{'='*60}\n{name} adversarial probe (gradient-reversal on entropy)\n{'='*60}")
        auc, auc_std, diff_r2 = cv_adversarial(X, y, delta)
        print(f"{name}: hallucination AUROC={auc:.4f}+-{auc_std:.4f}  "
              f"(entropy-head R^2 under adversarial pressure={diff_r2:.4f})")

        # Permutation test: shuffle y, rerun identical adversarial CV pipeline
        perm_aucs = []
        rng = np.random.default_rng(RANDOM_STATE)
        for p in range(N_PERM):
            y_perm = rng.permutation(y)
            auc_p, _, _ = cv_adversarial(X, y_perm, delta, seed=RANDOM_STATE + 1000 + p)
            perm_aucs.append(auc_p)
        perm_aucs = np.array(perm_aucs)
        p_value = float((perm_aucs >= auc).sum() + 1) / (N_PERM + 1)
        print(f"{name}: permutation test ({N_PERM} shuffles): perm_mean={perm_aucs.mean():.4f} "
              f"observed={auc:.4f} p={p_value:.4f}")

        results[name] = {
            "adversarial_auroc": auc, "adversarial_auroc_std": auc_std,
            "entropy_head_r2_under_adversarial_pressure": diff_r2,
            "n_permutations": N_PERM, "perm_mean": float(perm_aucs.mean()),
            "perm_std": float(perm_aucs.std()), "p_value": p_value,
        }

    out_path = ROOT / "results" / "gradient_reversal_adversarial_probe.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

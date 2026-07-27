"""
Paper 1 -- elite-review follow-up: turns the SAE feature-clamp's
positive-control result (331/24,576 features significant on the
companion HaluEval Pipeline-A dataset, best feature p=4.8e-11; see
S3.6 and results/sae_feature_clamp_paper1.json's "we ran the identical
Step 2-4 procedure on a companion paper's HaluEval Pipeline-A dataset")
into a precision/recall gating-utility curve, rather than leaving S4's
"No inference-economy claim" as an unsubstantiated disclaimer.

Question: does the single best-surviving feature's raw activation
value, thresholded, predict hallucination well enough to be useful as
a cheap (no extra forward pass beyond the SAE encode, itself a single
matrix multiply) routing/early-exit signal? This reuses the exact same
SAE weights and cached hidden states already used for the causal-clamp
test -- no new model inference, pure post-hoc analysis.
"""
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
HIDDEN_STATES_PATH = Path(
    "/Users/chakrivijayarao/Desktop/geom-proof/results/hidden_states/15_hidden_states_gpt2_117m.npz"
)
LAYER = 8
FDR_ALPHA = 0.05


def load_sae():
    path = hf_hub_download(repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
                            filename="blocks.8.hook_resid_pre/sae_weights.safetensors")
    w = load_file(path)
    return w["W_enc"].numpy(), w["b_enc"].numpy(), w["W_dec"].numpy(), w["b_dec"].numpy()


def encode_sae(H, W_enc, b_enc, b_dec):
    pre = (H - b_dec) @ W_enc + b_enc
    return np.maximum(pre, 0.0)


def select_best_feature(features, labels):
    n_features = features.shape[1]
    correct = features[labels == 1]
    hall = features[labels == 0]
    p_values = np.ones(n_features)
    for j in range(n_features):
        c, h = correct[:, j], hall[:, j]
        if c.std() == 0 and h.std() == 0:
            continue
        try:
            _, p = mannwhitneyu(c, h, alternative="two-sided")
            p_values[j] = p
        except ValueError:
            continue
    order = np.argsort(p_values)
    m = n_features
    bh_significant = np.zeros(n_features, dtype=bool)
    max_k = 0
    for rank, idx in enumerate(order):
        k = rank + 1
        if p_values[idx] <= (k / m) * FDR_ALPHA:
            max_k = k
    if max_k > 0:
        bh_significant[order[:max_k]] = True
    n_sig = int(bh_significant.sum())
    if n_sig == 0:
        return None, 0, p_values
    sig_idx = np.where(bh_significant)[0]
    best = sig_idx[np.argmin(p_values[sig_idx])]
    return int(best), n_sig, p_values


def main():
    print(f"Loading cached hidden states: {HIDDEN_STATES_PATH}")
    d = np.load(HIDDEN_STATES_PATH)
    H = d["hidden_states"][:, LAYER, :]
    y = d["labels"]
    print(f"H shape: {H.shape}, hall_rate={1 - y.mean():.3f}")

    print("Loading SAE weights (jbloom/GPT2-Small-SAEs-Reformatted, blocks.8.hook_resid_pre)...")
    W_enc, b_enc, W_dec, b_dec = load_sae()
    features = encode_sae(H, W_enc, b_enc, b_dec)

    best_feat, n_sig, p_values = select_best_feature(features, y)
    print(f"n_significant_bh_fdr={n_sig}, best_feature_idx={best_feat}, p={p_values[best_feat]:.3e}")
    assert best_feat is not None, "Expected a significant feature (companion positive control)."

    activations = features[:, best_feat]
    # Score = raw feature activation (higher = more "correct"-associated, per
    # the same correct-vs-hallucinated convention used throughout this project:
    # direction = mean(correct) - mean(hallucinated) uses y==1 as correct).
    auroc = roc_auc_score(y, activations)
    ap = average_precision_score(y, activations)
    precision, recall, thresholds = precision_recall_curve(y, activations)

    # Report precision/recall at a handful of representative recall levels,
    # rather than the full curve, for a compact "is this gate-worthy" table.
    target_recalls = [0.50, 0.75, 0.90, 0.95]
    curve_points = []
    for tr in target_recalls:
        idx_candidates = np.where(recall[:-1] >= tr)[0]
        if len(idx_candidates) == 0:
            continue
        idx = idx_candidates[-1]  # highest-threshold point still meeting this recall
        curve_points.append({
            "target_recall": tr, "actual_recall": float(recall[idx]),
            "precision": float(precision[idx]), "threshold": float(thresholds[idx]) if idx < len(thresholds) else None,
        })

    print(f"\nSingle-feature-activation-as-classifier: AUROC={auroc:.4f}, AP={ap:.4f}")
    print(f"Baseline (positive-class prior, i.e. correct-rate) = {y.mean():.4f}")
    for cp in curve_points:
        print(f"  recall>={cp['target_recall']}: actual_recall={cp['actual_recall']:.3f} "
              f"precision={cp['precision']:.3f}")

    out = {
        "n_samples": int(len(y)), "hall_rate": float(1 - y.mean()),
        "n_significant_bh_fdr": n_sig, "best_feature_idx": best_feat,
        "best_feature_p": float(p_values[best_feat]),
        "single_feature_auroc": float(auroc),
        "single_feature_average_precision": float(ap),
        "positive_class_prior": float(y.mean()),
        "precision_recall_operating_points": curve_points,
    }
    out_path = ROOT / "results" / "sae_feature_gating_utility.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

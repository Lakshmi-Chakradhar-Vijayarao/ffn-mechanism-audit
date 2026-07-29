"""
Pre-registered difficulty-matched control (Sec 3.6 of paper_draft.md),
actually run rather than left as a specified-but-unrun protocol.

Uses the parent MECH-INT project's real GPT-2 data:
  - data/processed/activations.pkl: 534 samples, each with
    ffn_outputs/attn_outputs of shape (12 layers, seq_len, 768)
  - data/processed/features.npy: 6 surface features per sample,
    feature 0 = mean_entropy (the difficulty proxy used here)
  - data/processed/labels.npy: 1=correct, 0=hallucinated

Difficulty proxy: mean_entropy (feature 0), already computed from the
model's own logit distribution during generation -- exactly the
delta(q) proxy specified in Sec 3.6, just using an already-available
feature rather than recomputing entropy from scratch.

Procedure: stratify into B=10 quantile bins by mean_entropy; within each
bin, subsample correct/hallucinated down to min(n_c,b, n_h,b); refit the
FFN-vs-Attention component probe (identical methodology to
src/probing/component_probe.py: mean-pooled layer output, 5-fold CV
logistic regression) on the matched set only, at the two peak layers
(FFN L8, Attn L3) identified in the original analysis.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
from scipy.stats import norm

# Fixed post-final-audit: was hardcoded to "~/Desktop/MECH-INT"
# (uppercase) against the real directory "mech-int" (lowercase) -- silently
# correct only on macOS's case-insensitive default filesystem, would break
# on Linux. features.npy/labels.npy are small and already vendored into
# this repo (results/surface_baseline/, verified byte-identical to
# mech-int's copy via md5); activations.pkl (2.9GB) is not vendored and
# remains a genuine external dependency on the mech-int sibling project.
ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "surface_baseline"
MECH_INT = os.path.expanduser(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int/data/processed"))
B = 10          # difficulty bins
PEAK_FFN_LAYER = 8
PEAK_ATTN_LAYER = 3
RANDOM_STATE = 42


def build_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)),
    ])


def cv_auroc(X, y, n_splits=5):
    actual_splits = min(n_splits, min(int(y.sum()), int((y == 0).sum())))
    actual_splits = max(2, actual_splits)
    probe = build_probe()
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std()), actual_splits


def run_matched_control(delta, delta_name, y, X_ffn_full, X_attn_full, n,
                         ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std):
    """Stratified difficulty-matched control for one choice of difficulty
    score `delta`. Factored out so the same procedure can be applied both
    to the single-feature entropy proxy (round-4) and the full 6-feature
    composite score (round-5 review: matching on entropy alone doesn't
    control for the other 5 features feeding the 0.576 surface baseline)."""
    print(f"\n{'='*70}\nMatched control using difficulty score: {delta_name}\n{'='*70}")
    bin_edges = np.quantile(delta, np.linspace(0, 1, B + 1))
    bin_edges[-1] += 1e-9  # include max in last bin
    bin_idx = np.digitize(delta, bin_edges[1:-1])  # 0..B-1

    matched_indices = []
    bin_report = []
    for b in range(B):
        idx_b = np.where(bin_idx == b)[0]
        correct_b = idx_b[y[idx_b] == 1]
        hall_b = idx_b[y[idx_b] == 0]
        k = min(len(correct_b), len(hall_b))
        bin_report.append((b, len(correct_b), len(hall_b), k))
        if k == 0:
            continue
        rng = np.random.default_rng(RANDOM_STATE + b)
        sel_correct = rng.choice(correct_b, size=k, replace=False)
        sel_hall = rng.choice(hall_b, size=k, replace=False)
        matched_indices.extend(sel_correct.tolist())
        matched_indices.extend(sel_hall.tolist())

    matched_indices = np.array(sorted(matched_indices))
    n_matched = len(matched_indices)
    print(f"\nBin report (bin, n_correct, n_hallucinated, matched_per_class):")
    for b, nc, nh, k in bin_report:
        print(f"  bin {b}: correct={nc:3d} hallucinated={nh:3d} matched/class={k:3d}")
    print(f"\nMatched set size: {n_matched} (from original {n}) "
          f"-- {n_matched/n*100:.1f}% retained")

    y_matched = y[matched_indices]
    print(f"Matched set balance: correct={y_matched.sum()}, hallucinated={(y_matched==0).sum()}")

    # Verify difficulty is no longer predictive of label in the matched set
    from scipy.stats import pointbiserialr
    r_before, p_before = pointbiserialr(y, delta)
    r_after, p_after = pointbiserialr(y_matched, delta[matched_indices])
    print(f"\nDifficulty-label correlation before matching: r={r_before:.4f}, p={p_before:.4f}")
    print(f"Difficulty-label correlation after matching:  r={r_after:.4f}, p={p_after:.4f}")

    # ── Refit FFN-vs-Attn probe on matched set ──
    X_ffn_matched = X_ffn_full[matched_indices]
    X_attn_matched = X_attn_full[matched_indices]

    ffn_m_auroc, ffn_m_std, ffn_splits = cv_auroc(X_ffn_matched, y_matched)
    attn_m_auroc, attn_m_std, attn_splits = cv_auroc(X_attn_matched, y_matched)

    print(f"\n=== RESULT: difficulty-matched FFN/Attn probe (N={n_matched}) ===")
    print(f"FFN  L{PEAK_FFN_LAYER}  AUROC = {ffn_m_auroc:.4f} +- {ffn_m_std:.4f} ({ffn_splits}-fold CV)")
    print(f"Attn L{PEAK_ATTN_LAYER}  AUROC = {attn_m_auroc:.4f} +- {attn_m_std:.4f} ({attn_splits}-fold CV)")

    # Round-5 review fix: treating fold-std/sqrt(splits) as a null-hypothesis
    # SE is invalid (5 CV folds are not 5 independent draws under H0: AUROC=0.5).
    # Replace with a label-permutation test: shuffle y_matched N_PERM times,
    # rerun the identical CV pipeline, and report the empirical one-sided
    # p-value as the fraction of permuted AUROCs >= the observed one.
    N_PERM = 500
    perm_results = {}
    for name, X, observed_auc in [("FFN", X_ffn_matched, ffn_m_auroc),
                                    ("Attn", X_attn_matched, attn_m_auroc)]:
        perm_aurocs = []
        rng = np.random.default_rng(RANDOM_STATE)
        for p in range(N_PERM):
            y_perm = rng.permutation(y_matched)
            auc_p, _, _ = cv_auroc(X, y_perm)
            perm_aurocs.append(auc_p)
        perm_aurocs = np.array(perm_aurocs)
        p_value = float((perm_aurocs >= observed_auc).sum() + 1) / (N_PERM + 1)
        perm_results[name] = {
            "observed_auroc": float(observed_auc),
            "n_permutations": N_PERM,
            "perm_mean": float(perm_aurocs.mean()),
            "perm_std": float(perm_aurocs.std()),
            "p_value": p_value,
        }
        print(f"{name}: permutation test vs chance ({N_PERM} shuffles): "
              f"perm_mean={perm_aurocs.mean():.4f}+-{perm_aurocs.std():.4f}, "
              f"observed={observed_auc:.4f}, p={p_value:.4f}")

    return {
        "delta_name": delta_name,
        "n_original": int(n),
        "n_matched": int(n_matched),
        "retention_pct": round(n_matched / n * 100, 1),
        "difficulty_label_corr_before": {"r": float(r_before), "p": float(p_before)},
        "difficulty_label_corr_after": {"r": float(r_after), "p": float(p_after)},
        "original_unmatched": {
            "ffn_auroc": ffn_full_auroc, "ffn_std": ffn_full_std,
            "attn_auroc": attn_full_auroc, "attn_std": attn_full_std,
        },
        "difficulty_matched": {
            "ffn_auroc": ffn_m_auroc, "ffn_std": ffn_m_std, "ffn_splits": ffn_splits,
            "attn_auroc": attn_m_auroc, "attn_std": attn_m_std, "attn_splits": attn_splits,
        },
        "permutation_test": perm_results,
        "bin_report": bin_report,
    }


def main():
    with open(f"{MECH_INT}/activations.pkl", "rb") as f:
        activations = pickle.load(f)
    y = np.load(VENDORED / "labels.npy")
    X_features = np.load(VENDORED / "features.npy")
    n = len(y)
    print(f"Loaded N={n}, correct={y.sum()}, hallucinated={(y==0).sum()}")

    X_ffn_full = np.stack([act["ffn_outputs"][PEAK_FFN_LAYER].mean(axis=0) for act in activations])
    X_attn_full = np.stack([act["attn_outputs"][PEAK_ATTN_LAYER].mean(axis=0) for act in activations])
    ffn_full_auroc, ffn_full_std, _ = cv_auroc(X_ffn_full, y)
    attn_full_auroc, attn_full_std, _ = cv_auroc(X_attn_full, y)
    print(f"\nOriginal (unmatched, N={n}): FFN L{PEAK_FFN_LAYER} AUROC="
          f"{ffn_full_auroc:.4f}+-{ffn_full_std:.4f}  Attn L{PEAK_ATTN_LAYER} AUROC="
          f"{attn_full_auroc:.4f}+-{attn_full_std:.4f}")

    # ── Control 1 (round-4): single-feature entropy proxy ──
    delta_entropy = X_features[:, 0]
    result_entropy = run_matched_control(
        delta_entropy, "entropy_only_feature0", y, X_ffn_full, X_attn_full, n,
        ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std,
    )

    # ── Control 2 (round-5 review fix): full 6-feature composite score ──
    # Round-5 review: matching on entropy alone doesn't control for the
    # other 5 surface features (max_entropy, logit_variance, confidence_gap,
    # attention_entropy, activation_norm) that also feed the 0.576 surface
    # baseline (05_run_surface_baseline.py). Use the out-of-fold predicted
    # probability of a logistic regression on all 6 features as a single
    # composite "generic surface computation" score, and match on that.
    from sklearn.model_selection import cross_val_predict
    composite_pipeline = build_probe()
    skf6 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    delta_composite = cross_val_predict(
        composite_pipeline, X_features, y, cv=skf6, method="predict_proba"
    )[:, 1]
    result_composite = run_matched_control(
        delta_composite, "full_6feature_composite_oof_prob", y, X_ffn_full, X_attn_full, n,
        ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std,
    )

    import json
    out = {
        "entropy_only_control": result_entropy,
        "full_6feature_composite_control": result_composite,
    }
    out_path = ROOT / "results" / "difficulty_matched_control.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- rerun the Sec 3.7 difficulty-matched control (code/06) under the
validated judge label instead of the original Jaccard label, completing
the same GPT-2 judge-label extension as code/29 (Sec 3.1/3.2). Imports
code/06's own `cv_auroc` and `run_matched_control` functions directly (via
importlib, since the filename starts with a digit) rather than
duplicating them, so any future fix to the matching/permutation-test logic
in code/06 stays in sync here automatically.
"""
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "surface_baseline"
MECH_INT = Path("/Users/chakrivijayarao/Desktop/mech-int/data/processed")

spec = importlib.util.spec_from_file_location("dmc06", ROOT / "code" / "06_difficulty_matched_control.py")
dmc06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmc06)

PEAK_FFN_LAYER = dmc06.PEAK_FFN_LAYER
PEAK_ATTN_LAYER = dmc06.PEAK_ATTN_LAYER


def main():
    with open(MECH_INT / "activations.pkl", "rb") as f:
        activations = pickle.load(f)

    with open(ROOT / "results" / "gpt2_full_534_judge_labels.json") as f:
        judge_data = json.load(f)
    judge_labels_raw = np.array(judge_data["judge_labels"])
    valid_mask = judge_labels_raw != -1

    y_jaccard_full = np.load(VENDORED / "labels.npy")
    X_features_full = np.load(VENDORED / "features.npy")

    y = judge_labels_raw[valid_mask]
    X_features = X_features_full[valid_mask]
    activations_valid = [a for a, v in zip(activations, valid_mask) if v]
    n = len(y)
    print(f"Loaded N={n} (of {len(y_jaccard_full)}, judge-valid only), "
          f"correct={y.sum()}, hallucinated={(y==0).sum()}")

    X_ffn_full = np.stack([act["ffn_outputs"][PEAK_FFN_LAYER].mean(axis=0) for act in activations_valid])
    X_attn_full = np.stack([act["attn_outputs"][PEAK_ATTN_LAYER].mean(axis=0) for act in activations_valid])
    ffn_full_auroc, ffn_full_std, _ = dmc06.cv_auroc(X_ffn_full, y)
    attn_full_auroc, attn_full_std, _ = dmc06.cv_auroc(X_attn_full, y)
    print(f"\n[JUDGE LABEL] Original (unmatched, N={n}): FFN L{PEAK_FFN_LAYER} AUROC="
          f"{ffn_full_auroc:.4f}+-{ffn_full_std:.4f}  Attn L{PEAK_ATTN_LAYER} AUROC="
          f"{attn_full_auroc:.4f}+-{attn_full_std:.4f}")

    delta_entropy = X_features[:, 0]
    result_entropy = dmc06.run_matched_control(
        delta_entropy, "entropy_only_feature0_JUDGE_LABEL", y, X_ffn_full, X_attn_full, n,
        ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std,
    )

    composite_pipeline = dmc06.build_probe()
    skf6 = StratifiedKFold(n_splits=5, shuffle=True, random_state=dmc06.RANDOM_STATE)
    delta_composite = cross_val_predict(
        composite_pipeline, X_features, y, cv=skf6, method="predict_proba"
    )[:, 1]
    result_composite = dmc06.run_matched_control(
        delta_composite, "full_6feature_composite_oof_prob_JUDGE_LABEL", y, X_ffn_full, X_attn_full, n,
        ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std,
    )

    out = {
        "n_valid": n,
        "entropy_only_control": result_entropy,
        "full_6feature_composite_control": result_composite,
    }
    out_path = ROOT / "results" / "difficulty_matched_control_judge_label.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

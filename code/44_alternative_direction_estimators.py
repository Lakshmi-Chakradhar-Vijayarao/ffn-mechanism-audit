"""
Paper 1 -- alternative direction estimators for the flagship causal test.

The tier1-validated kernel tests exactly one direction-estimation method
(difference-of-class-means) and finds it fails the direction-validity
gate (held-out AUROC CIs include or sit below 0.5, n=11). Before treating
that as evidence the causal effect itself doesn't exist, this checks
whether a DIFFERENT direction estimator -- L2-regularized logistic
regression weights, and Fisher LDA -- fares any better on the identical
held-out validity split. If none clear chance-level held-out AUROC, that
strengthens "the effect isn't there" over "this one estimator is bad."

Reuses the exact same deterministic train/validity split and activation
extraction as `kaggle_kernels/paper1-causal-patch-tier1-validated/`
(imported directly), so results are apples-to-apples with the diff-of-
means numbers already reported.
"""
import importlib.util
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
OUT_PATH = ROOT / "results" / "alternative_direction_estimators.json"


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("tier1_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lr_direction(vecs, labels):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(vecs, labels)
    w = clf.coef_[0]
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-8 else None


def lda_direction(vecs, labels):
    lda = LinearDiscriminantAnalysis()
    lda.fit(vecs, labels)
    w = lda.coef_[0]
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-8 else None


def bootstrap_ci(direction, test_vecs, test_labels, n_boot=2000, seed=42):
    scores = test_vecs @ direction
    try:
        auroc = roc_auc_score(test_labels, scores)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    n = len(test_labels)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = test_labels[idx]
        if len(np.unique(yy)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yy, scores[idx]))
    ci = (float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))) if boot_aurocs else (None, None)
    return {"auroc": float(auroc), "bootstrap_ci_95": list(ci), "n_test": int(n)}


def main():
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    k = load_kernel_module()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open((Path(os.path.expanduser(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int"))) / "data" / "processed" / "labeled.pkl"), "rb") as f:
        lab = pickle.load(f)
    with open(ROOT / "results" / "gpt2_full_534_judge_labels.json") as f:
        judge_data = json.load(f)
    prompts = lab["prompts"]
    judge_labels = np.array(judge_data["judge_labels"])

    valid_mask = judge_labels >= 0
    valid_idx = np.where(valid_mask)[0]
    correct_idx = [i for i in valid_idx if judge_labels[i] == 1]
    hall_idx = [i for i in valid_idx if judge_labels[i] == 0]
    rng = np.random.default_rng(k.RANDOM_STATE)
    rng.shuffle(correct_idx)
    rng.shuffle(hall_idx)

    n_train_correct = min(k.TRAIN_N_PER_CLASS, int(0.7 * len(correct_idx)) if len(correct_idx) < k.TRAIN_N_PER_CLASS * 2 else k.TRAIN_N_PER_CLASS)
    n_train_hall = min(k.TRAIN_N_PER_CLASS, int(0.7 * len(hall_idx)) if len(hall_idx) < k.TRAIN_N_PER_CLASS * 2 else k.TRAIN_N_PER_CLASS)
    n_train_correct = max(n_train_correct, min(5, len(correct_idx)))
    n_train_hall = max(n_train_hall, min(5, len(hall_idx)))
    train_correct_idx = correct_idx[:n_train_correct]
    train_hall_idx = hall_idx[:n_train_hall]

    n_val_correct = max(1, len(train_correct_idx) // 5)
    n_val_hall = max(1, len(train_hall_idx) // 5)
    direction_fit_idx = train_correct_idx[:-n_val_correct] + train_hall_idx[:-n_val_hall]
    validity_holdout_idx = train_correct_idx[-n_val_correct:] + train_hall_idx[-n_val_hall:]
    print(f"Direction-fit set: {len(direction_fit_idx)}, validity-holdout set: {len(validity_holdout_idx)}", flush=True)

    direction_fit_prompts = [prompts[i] for i in direction_fit_idx]
    direction_fit_labels = np.array([int(judge_labels[i]) for i in direction_fit_idx])
    validity_prompts = [prompts[i] for i in validity_holdout_idx]
    validity_labels = np.array([int(judge_labels[i]) for i in validity_holdout_idx])

    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    out = {}
    for layer_idx in (k.TARGET_LAYER, k.SECOND_LAYER):
        fit_ffn_vecs, fit_attn_vecs, _ = k.cache_train_activations(direction_fit_prompts, direction_fit_labels, model, tok, device, layer_idx)
        val_ffn_vecs, val_attn_vecs, _ = k.cache_train_activations(validity_prompts, validity_labels, model, tok, device, layer_idx)

        layer_out = {}
        for comp_name, fit_vecs, val_vecs in [("ffn", fit_ffn_vecs, val_ffn_vecs), ("attn", fit_attn_vecs, val_attn_vecs)]:
            for est_name, est_fn in [("diff_of_means", k.direction_from_vecs), ("logistic_regression", lr_direction), ("lda", lda_direction)]:
                direction = est_fn(fit_vecs, direction_fit_labels)
                if direction is None:
                    layer_out[f"{comp_name}_{est_name}"] = None
                    continue
                result = bootstrap_ci(direction, val_vecs, validity_labels)
                layer_out[f"{comp_name}_{est_name}"] = result
                print(f"  L{layer_idx} {comp_name} {est_name}: {result}", flush=True)
        out[str(layer_idx)] = layer_out

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- 200-resplit diagnostic for the direction-validity gate
(kaggle_kernels/paper1-causal-patch-tier1-validated/), addressing a
question raised by the same fresh review that found three of the four
n=11 direction-validity cells nominally significant in the
anti-predictive direction: is that a real, stable property of these
directions on this pool, or a chance draw at a single n=11 split that a
different resample would not reproduce?

Uses the exact same 534-item GPT-2/TruthfulQA labeled pool as the
tier1-validated kernel (results/gpt2_full_534_judge_labels.json: n=534,
27 correct, 507 hallucinated, judge-labeled -- index-aligned with the
kernel's own embedded LABELED_DATA_B64 prompts array) and the identical
train-pool construction (TRAIN_N_PER_CLASS=40 -> 18 correct + 40
hallucinated = 58, direction-fit 47 / validity-holdout 11 at an 80/20
split per class), but reruns that construction at 200 different random
seeds instead of the kernel's single RANDOM_STATE=42, to see how much the
held-out AUROC actually varies across resplits of the same underlying
pool.

Activations are extracted ONCE for all 534 prompts (GPT-2 is small enough
to run on CPU) and cached, so all 200 resplits reuse the same activation
matrix -- no repeated model calls.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
OUT_PATH = ROOT / "results" / "direction_validity_resplit_diagnostic.json"

TARGET_LAYER = 8
SECOND_LAYER = 9
TRAIN_N_PER_CLASS = 40
N_RESPLITS = 200


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("tier1_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier1_kernel"] = mod
    # The kernel module runs `load_dataset`/model-loading code only inside
    # main(), so importing it is side-effect-free apart from defining
    # functions/constants -- safe to import without running main().
    spec.loader.exec_module(mod)
    return mod


def main():
    print("Loading kernel module for load_labeled_data/extract_last_token...", flush=True)
    kernel = load_kernel_module()
    labeled = kernel.load_labeled_data()
    prompts = labeled["prompts"]
    print(f"Loaded {len(prompts)} embedded prompts", flush=True)

    judge_data = json.load(open(JUDGE_LABELS_PATH))
    judge_labels = np.array(judge_data["judge_labels"])
    assert len(judge_labels) == len(prompts), (
        f"judge_labels length {len(judge_labels)} != prompts length {len(prompts)} -- "
        f"not index-aligned, cannot proceed"
    )
    n_correct = int((judge_labels == 1).sum())
    n_hall = int((judge_labels == 0).sum())
    print(f"judge_labels: n_correct={n_correct} (expected 27), n_hall={n_hall} (expected 507)", flush=True)
    assert n_correct == judge_data["n_correct"]
    assert n_hall == judge_data["n_hallucinated"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2.eval()

    valid_idx = np.arange(len(prompts))  # all 534 are valid here (n_valid == n from the JSON)
    print("Extracting FFN/Attn activations at L8 and L9 for all prompts (one pass)...", flush=True)
    acts = {}  # acts[layer][sublayer] -> [N, 768]
    for layer_idx in (TARGET_LAYER, SECOND_LAYER):
        for sublayer in ("mlp", "attn"):
            vecs = []
            for i, p in enumerate(prompts):
                vecs.append(kernel.extract_last_token(p, gpt2, gpt2_tok, device, layer_idx, sublayer))
                if (i + 1) % 100 == 0:
                    print(f"  L{layer_idx}/{sublayer}: {i+1}/{len(prompts)}", flush=True)
            acts[(layer_idx, sublayer)] = np.stack(vecs)
    print("Activation extraction complete.", flush=True)

    correct_idx_all = np.where(judge_labels == 1)[0]
    hall_idx_all = np.where(judge_labels == 0)[0]

    results = {"n_resplits": N_RESPLITS, "layers": {}}
    for layer_idx in (TARGET_LAYER, SECOND_LAYER):
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            aurocs = []
            for seed in range(N_RESPLITS):
                rng = np.random.default_rng(seed)
                correct_idx = correct_idx_all.copy()
                hall_idx = hall_idx_all.copy()
                rng.shuffle(correct_idx)
                rng.shuffle(hall_idx)

                n_train_correct = min(TRAIN_N_PER_CLASS, int(0.7 * len(correct_idx)) if len(correct_idx) < TRAIN_N_PER_CLASS * 2 else TRAIN_N_PER_CLASS)
                n_train_hall = min(TRAIN_N_PER_CLASS, int(0.7 * len(hall_idx)) if len(hall_idx) < TRAIN_N_PER_CLASS * 2 else TRAIN_N_PER_CLASS)
                n_train_correct = max(n_train_correct, min(5, len(correct_idx)))
                n_train_hall = max(n_train_hall, min(5, len(hall_idx)))

                train_correct_idx = correct_idx[:n_train_correct]
                train_hall_idx = hall_idx[:n_train_hall]

                n_val_correct = max(1, n_train_correct // 5)
                n_val_hall = max(1, n_train_hall // 5)
                fit_idx = np.concatenate([train_correct_idx[:-n_val_correct], train_hall_idx[:-n_val_hall]])
                holdout_idx = np.concatenate([train_correct_idx[-n_val_correct:], train_hall_idx[-n_val_hall:]])

                vecs = acts[(layer_idx, sublayer)]
                fit_vecs, fit_labels = vecs[fit_idx], judge_labels[fit_idx]
                holdout_vecs, holdout_labels = vecs[holdout_idx], judge_labels[holdout_idx]

                direction = kernel.direction_from_vecs(fit_vecs, fit_labels)
                if direction is None:
                    continue
                scores = holdout_vecs @ direction
                try:
                    auroc = roc_auc_score(holdout_labels, scores)
                except ValueError:
                    continue
                aurocs.append(float(auroc))

            aurocs = np.array(aurocs)
            key = f"L{layer_idx}_{component}"
            results["layers"][key] = {
                "n_valid_resplits": len(aurocs),
                "mean": float(aurocs.mean()),
                "sd": float(aurocs.std(ddof=1)),
                "median": float(np.median(aurocs)),
                "min": float(aurocs.min()),
                "max": float(aurocs.max()),
                "pct_below_0.5": float((aurocs < 0.5).mean()),
                "pct_at_or_below_original_seed42": None,  # filled below
                "all_aurocs": aurocs.tolist(),
            }
            print(f"{key}: mean={aurocs.mean():.4f} sd={aurocs.std(ddof=1):.4f} "
                  f"median={np.median(aurocs):.4f} range=[{aurocs.min():.4f},{aurocs.max():.4f}] "
                  f"pct_below_0.5={100*(aurocs < 0.5).mean():.1f}%", flush=True)

    # Compare against the kernel's own single seed=42 split (its reported AUROCs)
    original_aurocs = {"L8_ffn": 0.0833333333333, "L8_attn": 0.0833333333333, "L9_ffn": 0.0, "L9_attn": 0.125}
    for key, orig in original_aurocs.items():
        arr = np.array(results["layers"][key]["all_aurocs"])
        pct = float((arr <= orig + 1e-9).mean())
        results["layers"][key]["original_seed42_auroc"] = orig
        results["layers"][key]["pct_of_resplits_at_or_below_original"] = pct
        print(f"{key}: original seed=42 AUROC={orig:.4f} is at or below {100*pct:.1f}% of the "
              f"{N_RESPLITS} resplit AUROCs", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

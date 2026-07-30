"""
Paper 1 -- low-dose alpha sweep {2.5, 5, 10} on the Tier-1-validated
common-site condition (block-level patch, L8), the one Tier-2 follow-up
from the original plan that was never run. Reuses the exact same
deterministic train/test split, direction-fitting, and patched-generation
code as `kaggle_kernels/paper1-causal-patch-tier1-validated/`
(imported directly, not reimplemented), on the same 60-prompt
ensemble subset that kernel already established as a fast stress-test
convention. GPT-2 is small enough to run this locally (no Kaggle needed)
since it's restricted to one layer, one site, three alphas.

Flip rate is scored against the Jaccard word-overlap label (reference
correct/incorrect answers from TruthfulQA) rather than the LLM judge --
this keeps the sweep local and fast (no 3B judge model needed) at the
cost of the judge label's higher fidelity; this is disclosed in the
output and in the paper text as a scope trade-off, not silently assumed
equivalent.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
OUT_PATH = ROOT / "results" / "low_dose_alpha_sweep.json"

LOW_DOSE_ALPHAS = [2.5, 5.0, 10.0]


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("tier1_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import pickle
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch

    k = load_kernel_module()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Reuse the already-computed, saved validated judge labels rather than
    # reloading the 3B judge model (the kernel's own load_labeled_data() +
    # fresh judging loop) -- same prompts, same judge, same result, much
    # faster for this local, low-priority follow-up sweep.
    with open((Path(os.path.expanduser(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int"))) / "data" / "processed" / "labeled.pkl"), "rb") as f:
        lab = pickle.load(f)
    with open(ROOT / "results" / "gpt2_full_534_judge_labels.json") as f:
        judge_data = json.load(f)
    prompts = lab["prompts"]
    judge_labels = judge_data["judge_labels"]
    questions = [k.question_from_prompt(p) for p in prompts]
    print(f"Loaded {len(prompts)} labeled prompts (reusing saved validated judge labels)", flush=True)

    valid_mask = np.array(judge_labels) >= 0
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
    train_idx = train_correct_idx + train_hall_idx
    hallucinated_test_idx = hall_idx[n_train_hall:]
    print(f"Train: {len(train_idx)}, hallucinated test pool: {len(hallucinated_test_idx)}", flush=True)

    train_prompts = [prompts[i] for i in train_idx]
    train_labels_judge = np.array([int(judge_labels[i]) for i in train_idx])

    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    ffn_vecs, attn_vecs, _ = k.cache_train_activations(train_prompts, train_labels_judge, model, tok, device, k.TARGET_LAYER)
    ffn_dir = k.direction_from_vecs(ffn_vecs, train_labels_judge)
    attn_dir = k.direction_from_vecs(attn_vecs, train_labels_judge)
    print(f"Directions fit at L{k.TARGET_LAYER}. FFN norm-ok={ffn_dir is not None}, Attn norm-ok={attn_dir is not None}", flush=True)

    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {item["question"].strip(): (item["correct_answers"], item["incorrect_answers"]) for item in ds}

    ensemble_subset_idx = hallucinated_test_idx[:k.ENSEMBLE_TEST_N]
    print(f"Running low-dose sweep on {len(ensemble_subset_idx)} prompts, alphas={LOW_DOSE_ALPHAS}, common site (block), L{k.TARGET_LAYER}", flush=True)

    out = {"alphas": LOW_DOSE_ALPHAS, "layer": k.TARGET_LAYER, "n_prompts": len(ensemble_subset_idx), "results": {}}
    for direction, dname in [(ffn_dir, "ffn"), (attn_dir, "attn")]:
        for alpha in LOW_DOSE_ALPHAS:
            n_flip = 0
            for j, i in enumerate(ensemble_subset_idx):
                prompt = prompts[i]
                question = questions[i]
                correct_answers, incorrect_answers = qmap[question.strip()]
                completion = k.patched_generate(prompt, model, tok, device, k.TARGET_LAYER, direction, alpha, "block")
                label = k.jaccard_label(completion, correct_answers, incorrect_answers)
                if label == 1:
                    n_flip += 1
            flip_rate = n_flip / len(ensemble_subset_idx)
            out["results"][f"{dname}_common_alpha{alpha}"] = {"n_flip": n_flip, "flip_rate": flip_rate}
            print(f"  {dname} common-site alpha={alpha}: flip_rate={flip_rate:.4f} ({n_flip}/{len(ensemble_subset_idx)})", flush=True)

    out["note"] = ("Flip rate scored against Jaccard word-overlap label (not the LLM judge), "
                    "to keep this sweep local/fast; a scope trade-off, disclosed in the paper text.")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

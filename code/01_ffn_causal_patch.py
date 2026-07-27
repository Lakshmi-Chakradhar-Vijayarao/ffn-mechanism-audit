"""
Causal FFN-Sublayer Patching Experiment (Paper 1, new experiment).

QUESTION
--------
mech-int's existing steering.py additively steers the WHOLE residual stream
(a mix of attention + FFN + embedding contributions) and only checks whether
a FROZEN probe's AUROC changes on held-out representations. It never touches
generation. mech-int's component_probe.py separately shows FFN output is a
better *correlational* predictor of hallucination than attention output at
L8 -- but correlational, not causal.

Neither experiment asks the question a mechanistic-interpretability reviewer
will ask first: if you intervene on the FFN sublayer specifically, does the
model's ACTUAL GENERATED ANSWER change? This script closes that gap, and
is the load-bearing new result needed before Paper 1 can be submitted --
it also directly engages arXiv 2604.13068 ("Detection Without Correction"),
which found that residual-stream steering flips 0/7 tested models' generated
answers toward correct on GPT-2-scale models. This script tests whether a
*component-targeted* (FFN-only) intervention does any better than that
generic negative result, using a random-direction and an attention-sublayer
control to isolate the FFN locus specifically.

PROTOCOL (no leakage)
----------------------
1. Reproduce mech-int/steering.py's 70/30 stratified split (seed=42) on the
   534 labeled TruthfulQA samples.
2. Compute a "found direction" per layer L from TRAIN-split FFN outputs only:
       dir_L = unit( mean(ffn_out[correct, L, last_token])
                    - mean(ffn_out[hallucinated, L, last_token]) )
3. Compute a random orthogonal control direction (same recipe as steering.py).
4. For each hallucinated TEST-split prompt, patch generation by adding
   alpha * direction to the target sublayer's output at every decoding step
   (works transparently under HF's KV-cache since each step re-invokes the
   hooked module), then re-label the new completion with the SAME Jaccard
   word-overlap scheme used to build the dataset (prepare_data.py),
   re-pulling correct_answers/incorrect_answers fresh from TruthfulQA.
5. Compare flip-to-correct rates: FFN+found-direction vs. FFN+random-direction
   vs. Attn+found-direction vs. no-patch baseline, with an exact (binomial)
   McNemar test on the discordant pairs between FFN+found and FFN+random.

This is intentionally a first, tractable-scope validation pass (one primary
layer, two alpha values, three intervention arms) to get real numbers fast.
Scale-up (full layer sweep, full 534 samples, more alphas) is a follow-up
run once this pipeline is confirmed working and the direction of the effect
is known.
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset

# Fixed post-final-audit: was hardcoded to a personal absolute path to
# the mech-int sibling project (this imports live code, src.model.load_model,
# not just data, so it cannot be fully vendored -- env var override added).
ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
MECH_INT_ROOT = Path(os.environ.get("MECH_INT_ROOT", "/Users/chakrivijayarao/Desktop/mech-int"))
sys.path.insert(0, str(MECH_INT_ROOT))

from src.model.load_model import load_gpt2  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET_LAYER = 8          # peak FFN component-probe layer per component_probe.py
SECOND_LAYER = 9          # peak dense-probe / steering layer, for robustness
ALPHAS = [20.0, 40.0]     # representative mid/high steering strengths
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12    # same threshold as prepare_data.py


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labeled():
    labeled_path = VENDORED / "labeled.pkl"
    if not labeled_path.exists():
        labeled_path = MECH_INT_ROOT / "data/processed/labeled.pkl"
    with open(labeled_path, "rb") as f:
        return pickle.load(f)


def build_question_answer_map():
    """question -> (correct_answers, incorrect_answers), fresh from HF."""
    print("Loading TruthfulQA (generation, validation) for answer lookup...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {}
    for item in ds:
        qmap[item["question"].strip()] = (item["correct_answers"], item["incorrect_answers"])
    print(f"  {len(qmap)} questions indexed")
    return qmap


def question_from_prompt(prompt: str) -> str:
    # prompts are formatted "Q: {question}\nA:"
    q = prompt.split("Q:", 1)[1].split("\nA:", 1)[0]
    return q.strip()


# ── Labeling (identical recipe to prepare_data.py) ────────────────────────────

def _word_overlap(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def label_completion(completion: str, correct_answers, incorrect_answers) -> int:
    best_correct = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_incorrect = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_correct > LABEL_THRESHOLD or best_incorrect > LABEL_THRESHOLD:
        return 1 if best_correct >= best_incorrect else 0
    return -1


# ── Split (must match steering.py exactly: same seed, same test_size) ────────

def get_split(labels):
    y = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, test_idx


# ── FFN direction computation (train split only, last-token) ─────────────────

def extract_ffn_last_token(prompt, model, tokenizer, device, layer_idx):
    """Single forward pass; return the FFN sublayer output at layer_idx, last token."""
    captured = {}

    def hook(module, inp, out):
        captured["ffn"] = out[0, -1, :].detach().cpu().numpy().copy()

    h = model.transformer.h[layer_idx].mlp.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        h.remove()
    return captured["ffn"]


def compute_ffn_direction(train_prompts, train_labels, model, tokenizer, device, layer_idx):
    correct_vecs, hall_vecs = [], []
    for p, y in zip(train_prompts, train_labels):
        vec = extract_ffn_last_token(p, model, tokenizer, device, layer_idx)
        (correct_vecs if y == 1 else hall_vecs).append(vec)
    direction = np.mean(correct_vecs, axis=0) - np.mean(hall_vecs, axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("FFN direction near zero.")
    return direction / norm


def random_orthogonal_direction(direction, seed=42):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(direction.shape)
    v -= np.dot(v, direction) * direction
    v /= np.linalg.norm(v)
    return v


# ── Causal patch + generate ───────────────────────────────────────────────────

def patched_generate(prompt, model, tokenizer, device, layer_idx, direction, alpha, sublayer):
    """Greedy-generate with `alpha * direction` added to a sublayer's output
    at every decoding step. sublayer in {'mlp', 'attn'}."""
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        if sublayer == "mlp":
            return out + alpha * direction_t
        else:
            # attn block returns a tuple; out[0] is the hidden-state tensor
            modified = out[0] + alpha * direction_t
            return (modified,) + out[1:]

    module = (
        model.transformer.h[layer_idx].mlp
        if sublayer == "mlp"
        else model.transformer.h[layer_idx].attn
    )
    handle = module.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    finally:
        handle.remove()


def unpatched_generate(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Main experiment ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    labeled = load_labeled()
    prompts, labels = labeled["prompts"], labeled["labels"]
    train_idx, test_idx = get_split(labels)

    train_prompts = [prompts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    hallucinated_test_idx = [i for i in test_idx if labels[i] == 0]
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}  "
          f"Hallucinated-in-test: {len(hallucinated_test_idx)}")

    model, tokenizer, device = load_gpt2()
    qmap = build_question_answer_map()

    print(f"\nComputing FFN found-direction at L{TARGET_LAYER} (train split only)...")
    direction_l8 = compute_ffn_direction(train_prompts, train_labels, model, tokenizer, device, TARGET_LAYER)
    random_l8 = random_orthogonal_direction(direction_l8, seed=RANDOM_STATE)
    print(f"Computing FFN found-direction at L{SECOND_LAYER} (train split only)...")
    direction_l9 = compute_ffn_direction(train_prompts, train_labels, model, tokenizer, device, SECOND_LAYER)
    random_l9 = random_orthogonal_direction(direction_l9, seed=RANDOM_STATE)

    directions = {TARGET_LAYER: (direction_l8, random_l8), SECOND_LAYER: (direction_l9, random_l9)}

    results = []
    n = len(hallucinated_test_idx)
    for progress_i, i in enumerate(hallucinated_test_idx):
        prompt = prompts[i]
        question = question_from_prompt(prompt)
        if question not in qmap:
            continue
        correct_answers, incorrect_answers = qmap[question]

        row = {"idx": int(i), "prompt": prompt}

        baseline_completion = unpatched_generate(prompt, model, tokenizer, device)
        row["baseline_label"] = label_completion(baseline_completion, correct_answers, incorrect_answers)
        row["baseline_completion"] = baseline_completion

        for layer_idx in (TARGET_LAYER, SECOND_LAYER):
            found_dir, rand_dir = directions[layer_idx]
            for alpha in ALPHAS:
                ffn_found_c = patched_generate(prompt, model, tokenizer, device, layer_idx, found_dir, alpha, "mlp")
                ffn_rand_c = patched_generate(prompt, model, tokenizer, device, layer_idx, rand_dir, alpha, "mlp")
                attn_found_c = patched_generate(prompt, model, tokenizer, device, layer_idx, found_dir, alpha, "attn")

                key = f"L{layer_idx}_a{int(alpha)}"
                row[f"{key}_ffn_found_label"] = label_completion(ffn_found_c, correct_answers, incorrect_answers)
                row[f"{key}_ffn_rand_label"] = label_completion(ffn_rand_c, correct_answers, incorrect_answers)
                row[f"{key}_attn_found_label"] = label_completion(attn_found_c, correct_answers, incorrect_answers)

        results.append(row)
        if (progress_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{progress_i + 1}/{n}] elapsed={elapsed:.0f}s")

    # ── Aggregate flip-to-correct rates + McNemar exact test ──────────────────
    summary = {}
    for layer_idx in (TARGET_LAYER, SECOND_LAYER):
        for alpha in ALPHAS:
            key = f"L{layer_idx}_a{int(alpha)}"
            ffn_found_flip = [r[f"{key}_ffn_found_label"] == 1 for r in results]
            ffn_rand_flip = [r[f"{key}_ffn_rand_label"] == 1 for r in results]
            attn_found_flip = [r[f"{key}_attn_found_label"] == 1 for r in results]

            # McNemar exact (binomial) test: FFN-found vs FFN-random, discordant pairs only
            b = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if f and not r)  # found flips, random doesn't
            c = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if not f and r)  # random flips, found doesn't
            if b + c > 0:
                mcnemar_p = binomtest(b, b + c, 0.5).pvalue
            else:
                mcnemar_p = 1.0

            summary[key] = {
                "n": len(results),
                "ffn_found_flip_rate": round(float(np.mean(ffn_found_flip)), 4),
                "ffn_random_flip_rate": round(float(np.mean(ffn_rand_flip)), 4),
                "attn_found_flip_rate": round(float(np.mean(attn_found_flip)), 4),
                "mcnemar_discordant_b_found_only": b,
                "mcnemar_discordant_c_random_only": c,
                "mcnemar_exact_p": round(float(mcnemar_p), 4),
            }

    baseline_hall_confirmed = sum(1 for r in results if r["baseline_label"] == 0)
    print(f"\nBaseline re-labeling sanity check: {baseline_hall_confirmed}/{len(results)} "
          f"confirmed hallucinated under fresh relabeling (should be close to {len(results)})")

    print("\n=== Flip-to-correct rate summary ===")
    for key, s in summary.items():
        print(f"  {key}: FFN-found={s['ffn_found_flip_rate']:.3f}  "
              f"FFN-random={s['ffn_random_flip_rate']:.3f}  "
              f"Attn-found={s['attn_found_flip_rate']:.3f}  "
              f"McNemar p={s['mcnemar_exact_p']:.4f}  "
              f"(n={s['n']}, b={s['mcnemar_discordant_b_found_only']}, c={s['mcnemar_discordant_c_random_only']})")

    out = {
        "config": {
            "target_layer": TARGET_LAYER,
            "second_layer": SECOND_LAYER,
            "alphas": ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        "baseline_hall_confirmed": baseline_hall_confirmed,
        "n_hallucinated_test": len(results),
        "summary": summary,
        "per_sample": results,
    }
    out_path = OUT_DIR / "ffn_causal_patch_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full results to {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

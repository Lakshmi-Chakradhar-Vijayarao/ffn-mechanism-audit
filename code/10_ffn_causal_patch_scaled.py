"""
Paper 1 -- Tier 2 stretch goal (final-audit pass): scales up the original
GPT-2 dense-direction causal-patching test (01_ffn_causal_patch.py,
n=81 hallucinated-in-test prompts) toward the paper's own disclosed
"n~300-400" properly-powered target.

HONEST SCOPE, stated before any result: this target is not fully
reachable on this fixed dataset. The entire labeled pool
(`results/vendored_mech_int/labeled.pkl`) has exactly 534 examples (268
hallucinated, 266 correct) -- this is GPT-2's own labeled generations
over ALL 817 TruthfulQA validation questions, with the ~283 remaining
questions dropped as ambiguous under the paper's own Jaccard
word-overlap threshold (LABEL_THRESHOLD=0.12). 268 hallucinated examples
is therefore a hard ceiling on how many hallucinated-in-test prompts
this specific dataset can ever supply, regardless of train/test split
ratio, since a train split of size 0 would leave the found-direction
estimate degenerate. We disclose this rather than silently reporting a
smaller number as if it met the target.

Design: shrink the train split to the minimum needed for a stable
per-class mean-difference direction (15%, ~40 examples per class) and
use the remaining ~85% (up to 228 hallucinated prompts) as test --
maximizing n within the dataset's real ceiling, a 2.8x increase over the
original n=81. Everything else is identical to
01_ffn_causal_patch.py: L8/L9 FFN+Attn found-vs-random-direction
patching, alphas {20, 40}, exact (binomial) McNemar test on discordant
found-vs-random flip-to-correct pairs. The only methodological change
(besides train/test ratio) is loading GPT-2 directly via `transformers`
instead of importing the sibling mech-int project's `load_gpt2`, to
remove an external local-path dependency -- the model, tokenizer, and
weights are identical (`gpt2`, the 117M checkpoint).
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
OUT_PATH = ROOT / "results" / "ffn_causal_patch_scaled_results.json"

RANDOM_STATE = 42
TRAIN_SIZE = 0.15         # minimum needed for a stable per-class direction; maximizes test n
TARGET_LAYER = 8
SECOND_LAYER = 9
ALPHAS = [20.0, 40.0]
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12


def load_labeled():
    with open(VENDORED / "labeled.pkl", "rb") as f:
        return pickle.load(f)


def build_question_answer_map():
    print("Loading TruthfulQA (generation, validation) for answer lookup...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {}
    for item in ds:
        qmap[item["question"].strip()] = (item["correct_answers"], item["incorrect_answers"])
    print(f"  {len(qmap)} questions indexed")
    return qmap


def question_from_prompt(prompt: str) -> str:
    q = prompt.split("Q:", 1)[1].split("\nA:", 1)[0]
    return q.strip()


def _word_overlap(a: str, b: str) -> float:
    words_a, words_b = set(a.lower().split()), set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def label_completion(completion: str, correct_answers, incorrect_answers) -> int:
    best_correct = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_incorrect = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_correct > LABEL_THRESHOLD or best_incorrect > LABEL_THRESHOLD:
        return 1 if best_correct >= best_incorrect else 0
    return -1


def get_split(labels):
    y = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, test_idx


def extract_ffn_last_token(prompt, model, tokenizer, device, layer_idx):
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
    return direction / norm, len(correct_vecs), len(hall_vecs)


def random_orthogonal_direction(direction, seed=42):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(direction.shape)
    v -= np.dot(v, direction) * direction
    v /= np.linalg.norm(v)
    return v


def patched_generate(prompt, model, tokenizer, device, layer_idx, direction, alpha, sublayer):
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        if sublayer == "mlp":
            return out + alpha * direction_t
        modified = out[0] + alpha * direction_t
        return (modified,) + out[1:]

    module = (model.transformer.h[layer_idx].mlp if sublayer == "mlp"
              else model.transformer.h[layer_idx].attn)
    handle = module.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()


def unpatched_generate(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    labeled = load_labeled()
    prompts, labels = labeled["prompts"], labeled["labels"]
    train_idx, test_idx = get_split(labels)

    train_prompts = [prompts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    hallucinated_test_idx = [i for i in test_idx if labels[i] == 0]
    print(f"Total labeled: {len(labels)} (hallucinated={sum(1 for l in labels if l == 0)}, "
          f"correct={sum(1 for l in labels if l == 1)})")
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}  "
          f"Hallucinated-in-test: {len(hallucinated_test_idx)}  "
          f"(vs. n=81 in the original 70/30-split run)")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    qmap = build_question_answer_map()

    print(f"\nComputing FFN found-direction at L{TARGET_LAYER} (train split only, n={len(train_idx)})...")
    direction_l8, n_correct_l8, n_hall_l8 = compute_ffn_direction(
        train_prompts, train_labels, model, tokenizer, device, TARGET_LAYER)
    random_l8 = random_orthogonal_direction(direction_l8, seed=RANDOM_STATE)
    print(f"Computing FFN found-direction at L{SECOND_LAYER} (train split only)...")
    direction_l9, n_correct_l9, n_hall_l9 = compute_ffn_direction(
        train_prompts, train_labels, model, tokenizer, device, SECOND_LAYER)
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
        if (progress_i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (progress_i + 1)) * (n - progress_i - 1) / 60
            print(f"  [{progress_i + 1}/{n}] elapsed={elapsed:.0f}s ETA={eta:.0f}min", flush=True)

    summary = {}
    for layer_idx in (TARGET_LAYER, SECOND_LAYER):
        for alpha in ALPHAS:
            key = f"L{layer_idx}_a{int(alpha)}"
            ffn_found_flip = [r[f"{key}_ffn_found_label"] == 1 for r in results]
            ffn_rand_flip = [r[f"{key}_ffn_rand_label"] == 1 for r in results]
            attn_found_flip = [r[f"{key}_attn_found_label"] == 1 for r in results]

            b = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if f and not r)
            c = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if not f and r)
            mcnemar_p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0

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
          f"confirmed hallucinated under fresh relabeling")

    print("\n=== Flip-to-correct rate summary (scaled-up run) ===")
    for key, s in summary.items():
        print(f"  {key}: FFN-found={s['ffn_found_flip_rate']:.3f}  "
              f"FFN-random={s['ffn_random_flip_rate']:.3f}  "
              f"Attn-found={s['attn_found_flip_rate']:.3f}  "
              f"McNemar p={s['mcnemar_exact_p']:.4f}  "
              f"(n={s['n']}, b={s['mcnemar_discordant_b_found_only']}, c={s['mcnemar_discordant_c_random_only']})")

    out = {
        "config": {
            "target_layer": TARGET_LAYER, "second_layer": SECOND_LAYER, "alphas": ALPHAS,
            "max_new_tokens": MAX_NEW_TOKENS, "random_state": RANDOM_STATE, "train_size": TRAIN_SIZE,
            "n_train_correct_l8": n_correct_l8, "n_train_hall_l8": n_hall_l8,
        },
        "total_labeled": len(labels), "total_hallucinated": sum(1 for l in labels if l == 0),
        "baseline_hall_confirmed": baseline_hall_confirmed,
        "n_hallucinated_test": len(results),
        "summary": summary,
        "per_sample": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full results to {OUT_PATH}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

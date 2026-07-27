"""
Paper 1 -- elite-review follow-up: the n=228 scaled causal-patch test
(code/10_ffn_causal_patch_scaled.py) inherits the same repetition-loop
construct-validity issue flagged for the original n=81 test
(code/04_degeneration_check.py: 42/81 baseline "hallucinated" completions
are degenerate repetition loops, not confident false claims) but was
never itself re-filtered. This regenerates ONLY the 228 baseline
completions (identical seed/split/greedy-decode as 10_*.py -- no patched
generations needed, since those labels are already saved), applies the
same is_repetitive() criterion, and recomputes McNemar on the
non-degenerate subset only, reusing the already-computed found/random
labels from results/ffn_causal_patch_scaled_results.json.
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"

RANDOM_STATE = 42
TRAIN_SIZE = 0.15
MAX_NEW_TOKENS = 40


def is_repetitive(text, min_repeat=3):
    words = text.split()
    if len(words) < 12:
        return False
    for chunk_len in (4, 5, 6, 8):
        seen = {}
        for i in range(len(words) - chunk_len + 1):
            chunk = " ".join(words[i:i + chunk_len])
            seen[chunk] = seen.get(chunk, 0) + 1
        if max(seen.values(), default=0) >= min_repeat:
            return True
    return False


def wilson_ci(k, n, z=1.96):
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return center - half, center + half


def get_split(labels):
    y = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, test_idx


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    with open(VENDORED / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    prompts, labels = labeled["prompts"], labeled["labels"]
    train_idx, test_idx = get_split(labels)
    hallucinated_test_idx = [i for i in test_idx if labels[i] == 0]
    print(f"Hallucinated-in-test: {len(hallucinated_test_idx)} (should match n=228 scaled run)")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    idx_to_completion = {}
    for i in hallucinated_test_idx:
        prompt = prompts[i]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        idx_to_completion[i] = completion

    degenerate_idx = {i for i, c in idx_to_completion.items() if is_repetitive(c)}
    n_total = len(hallucinated_test_idx)
    n_degenerate = len(degenerate_idx)
    lo, hi = wilson_ci(n_degenerate, n_total)
    print(f"\n{n_degenerate}/{n_total} = {n_degenerate/n_total*100:.1f}% degenerate repetition loops "
          f"(Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")

    with open(ROOT / "results" / "ffn_causal_patch_scaled_results.json") as f:
        scaled = json.load(f)
    per_sample = scaled["per_sample"]
    idx_set = set(hallucinated_test_idx) - degenerate_idx
    filtered_samples = [s for s in per_sample if s["idx"] in idx_set]
    print(f"Filtered (non-degenerate) subset: {len(filtered_samples)}/{len(per_sample)}")

    configs = [("L8", 20), ("L8", 40), ("L9", 20), ("L9", 40)]
    summary = {}
    for layer, alpha in configs:
        key = f"{layer}_a{alpha}"
        ffn_found_flip = [s[f"{key}_ffn_found_label"] == 1 for s in filtered_samples]
        ffn_rand_flip = [s[f"{key}_ffn_rand_label"] == 1 for s in filtered_samples]
        attn_found_flip = [s[f"{key}_attn_found_label"] == 1 for s in filtered_samples]

        b = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if f and not r)
        c = sum(1 for f, r in zip(ffn_found_flip, ffn_rand_flip) if not f and r)
        mcnemar_p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0

        summary[key] = {
            "n": len(filtered_samples),
            "ffn_found_flip_rate": round(float(np.mean(ffn_found_flip)), 4),
            "ffn_random_flip_rate": round(float(np.mean(ffn_rand_flip)), 4),
            "attn_found_flip_rate": round(float(np.mean(attn_found_flip)), 4),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(float(mcnemar_p), 4),
        }
        print(f"  {key}: FFN-found={summary[key]['ffn_found_flip_rate']:.3f}  "
              f"FFN-random={summary[key]['ffn_random_flip_rate']:.3f}  "
              f"Attn-found={summary[key]['attn_found_flip_rate']:.3f}  "
              f"McNemar p={summary[key]['mcnemar_p']:.4f}  (b={b}, c={c})")

    out = {
        "n_total_hallucinated_test": n_total,
        "n_degenerate_repetition_loops": n_degenerate,
        "degenerate_pct": round(n_degenerate / n_total * 100, 1),
        "degenerate_wilson_ci": [round(lo * 100, 1), round(hi * 100, 1)],
        "n_filtered_nondegenerate": len(filtered_samples),
        "summary_filtered": summary,
    }
    out_path = ROOT / "results" / "ffn_causal_patch_scaled_degeneration_filtered.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

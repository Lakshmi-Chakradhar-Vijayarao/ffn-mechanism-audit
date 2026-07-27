"""
Construct-validity check flagged by round-4 review: what fraction of the
81 "hallucinated" baseline completions used in the causal-patching test
(Sec 3.4) are degenerate repetition loops rather than confident false
claims? A repetition-loop-dominated test set means the "flip-to-correct"
metric partly measures "did the intervention break a repetition loop,"
not "did it correct a factual error."

Result: 42/81 (51.9%, Wilson 95% CI [41.1%, 62.4%]) are degenerate
repetition loops by the criterion below (a 4-8 word phrase repeated
3+ times).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path


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


def main():
    with open(ROOT / "results" / "ffn_causal_patch_results.json") as f:
        d = json.load(f)
    hallucinated_baseline = [s for s in d["per_sample"] if s["baseline_label"] == 0]
    degenerate = [s for s in hallucinated_baseline if is_repetitive(s["baseline_completion"])]
    n, k = len(hallucinated_baseline), len(degenerate)
    lo, hi = wilson_ci(k, n)
    print(f"{k}/{n} = {k/n*100:.1f}% degenerate repetition loops")
    print(f"Wilson 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")


if __name__ == "__main__":
    main()

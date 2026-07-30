"""
Paper 1 -- formal power / minimum-detectable-effect analysis for the
direction-validity AUROC gate (kaggle_kernels/paper1-causal-patch-tier1-
validated/), rather than asserting "n=11 is small" without quantifying
what it means. The validity-holdout split there is 3 correct + 8
hallucinated (n=11 total) at both L8 and L9, bottlenecked by GPT-2's
judge-validated correct rate on TruthfulQA (27/534, 5.1%).

Monte Carlo power analysis: for a range of TRUE held-out AUROC values,
simulate binormal scores matching that AUROC at n_pos=3/n_neg=8 (and
separately at the enlarged pool's split, once available), compute the
same 2000-resample bootstrap CI the kernel itself uses, and estimate the
fraction of simulated replicates whose 95% CI excludes 0.5 (statistical
power to detect that true AUROC as different from chance). Reports the
smallest true AUROC clearing 80% power -- the number this test could
have detected, not just "n is small."
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT.parent / "results" / "direction_validity_power_analysis.json"

N_POS_ORIG, N_NEG_ORIG = 3, 8  # validity holdout at n=11 (3 correct, 8 hallucinated)
N_SIM = 2000
N_BOOT = 2000
TRUE_AUROCS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
RNG = np.random.default_rng(2026)


def fast_auroc_batch(pos_scores, neg_scores):
    """Vectorized AUROC (Mann-Whitney U / n_pos*n_neg) for many resamples at
    once. pos_scores: [B, n_pos], neg_scores: [B, n_neg] -> [B] AUROCs."""
    n_pos, n_neg = pos_scores.shape[1], neg_scores.shape[1]
    # count(pos > neg) + 0.5*count(pos == neg), summed over all pairs, per batch row
    diff = pos_scores[:, :, None] - neg_scores[:, None, :]  # [B, n_pos, n_neg]
    wins = (diff > 0).sum(axis=(1, 2)) + 0.5 * (diff == 0).sum(axis=(1, 2))
    return wins / (n_pos * n_neg)


def power_at(true_auroc, n_pos, n_neg, n_sim=N_SIM, n_boot=N_BOOT, rng=RNG):
    d_prime = np.sqrt(2) * norm.ppf(true_auroc)
    hits = 0
    for _ in range(n_sim):
        pos = rng.standard_normal(n_pos) + d_prime
        neg = rng.standard_normal(n_neg)
        # bootstrap resample (with replacement) within each class, n_boot draws at once
        boot_pos = pos[rng.integers(0, n_pos, size=(n_boot, n_pos))]
        boot_neg = neg[rng.integers(0, n_neg, size=(n_boot, n_neg))]
        boot_aurocs = fast_auroc_batch(boot_pos, boot_neg)
        ci_low = np.percentile(boot_aurocs, 2.5)
        if ci_low > 0.5:
            hits += 1
    return hits / n_sim


def main():
    results = {"n_pos": N_POS_ORIG, "n_neg": N_NEG_ORIG, "n_total": N_POS_ORIG + N_NEG_ORIG, "power_by_true_auroc": {}}
    print(f"Power analysis at n_pos={N_POS_ORIG}, n_neg={N_NEG_ORIG} (the actual validity-holdout split):")
    mde_80 = None
    for auroc in TRUE_AUROCS:
        p = power_at(auroc, N_POS_ORIG, N_NEG_ORIG)
        results["power_by_true_auroc"][str(auroc)] = p
        print(f"  true AUROC={auroc:.2f}: power={p:.3f}", flush=True)
        if p >= 0.80 and mde_80 is None:
            mde_80 = auroc
    results["minimum_detectable_auroc_80pct_power"] = mde_80
    results["interpretation"] = (
        f"At n=11 ({N_POS_ORIG} correct + {N_NEG_ORIG} hallucinated), this test only reaches "
        f"80% power to distinguish from chance (AUROC=0.5) once the true held-out AUROC is "
        f"approximately {mde_80 if mde_80 else '>0.99'}. The observed point estimates in the "
        f"kernel output (0.0-0.125) are far below any AUROC this test could reliably confirm OR "
        f"rule out -- the null result is uninformative by construction at this n, not just "
        f"'small sample size' in a vague sense."
    )
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMDE at 80% power: {mde_80}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

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

CORRECTION (independent adversarial review found this): the power
criterion was one-sided (`ci_low > 0.5` only), structurally blind to
detecting a true AUROC below chance -- exactly the direction three of
the four observed cells in the kernel output actually fall in (nominally
significant anti-predictive, per the corrected §4 text). Fixed to a
two-sided criterion (`ci_low > 0.5 or ci_high < 0.5`), and TRUE_AUROCS
extended to include below-chance values so the MDE is reported
symmetrically in both directions.
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
TRUE_AUROCS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
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
        ci_high = np.percentile(boot_aurocs, 97.5)
        if ci_low > 0.5 or ci_high < 0.5:
            hits += 1
    return hits / n_sim


def main():
    results = {"n_pos": N_POS_ORIG, "n_neg": N_NEG_ORIG, "n_total": N_POS_ORIG + N_NEG_ORIG, "power_by_true_auroc": {}}
    print(f"Power analysis at n_pos={N_POS_ORIG}, n_neg={N_NEG_ORIG} (the actual validity-holdout split), two-sided criterion:")
    mde_80_high = None  # smallest true AUROC > 0.5 clearing 80% power
    mde_80_low = None   # largest true AUROC < 0.5 clearing 80% power
    for auroc in TRUE_AUROCS:
        p = power_at(auroc, N_POS_ORIG, N_NEG_ORIG)
        results["power_by_true_auroc"][str(auroc)] = p
        print(f"  true AUROC={auroc:.2f}: power={p:.3f}", flush=True)
        if auroc > 0.5 and p >= 0.80 and mde_80_high is None:
            mde_80_high = auroc
        if auroc < 0.5 and p >= 0.80:
            mde_80_low = auroc  # keep updating -> ends at the value closest to 0.5 clearing 80%
    results["minimum_detectable_auroc_80pct_power_high_side"] = mde_80_high
    results["minimum_detectable_auroc_80pct_power_low_side"] = mde_80_low
    results["interpretation"] = (
        f"At n=11 ({N_POS_ORIG} correct + {N_NEG_ORIG} hallucinated), this test only reaches "
        f"80% power to distinguish from chance (AUROC=0.5) in the helpful direction once the true "
        f"held-out AUROC is approximately {mde_80_high if mde_80_high else '>0.99'}, or in the "
        f"anti-predictive direction once it is approximately "
        f"{mde_80_low if mde_80_low else '<0.01'}. The observed point estimates in the kernel "
        f"output (0.0-0.125) sit inside the well-powered anti-predictive region at this n, which "
        f"is exactly why the exact Mann-Whitney tests on the same data are nominally significant "
        f"(anti-predictive) at 3 of 4 cells -- this test is well-powered to detect a true AUROC "
        f"this far below chance, so that nominal significance is not itself surprising or "
        f"informative about a real anti-predictive relationship; it does not, however, resolve "
        f"whether the true held-out AUROC is actually anti-predictive or simply appears so due to "
        f"chance fluctuation at n=11, since a single realization cannot distinguish the two "
        f"without more data. The test remains uninformative about whether the direction carries "
        f"any *helpful* signal (AUROC > 0.5), which would require a true AUROC of "
        f"{mde_80_high if mde_80_high else '>0.99'} to detect at 80% power -- far above anything "
        f"plausible here."
    )
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMDE (high side, 80% power): {mde_80_high}")
    print(f"MDE (low side, 80% power): {mde_80_low}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

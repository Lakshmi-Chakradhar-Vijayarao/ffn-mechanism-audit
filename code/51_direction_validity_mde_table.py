"""
Paper 1 -- a formal minimum-detectable-effect / power table for the
direction-validity gate proposed in the Discussion checklist.

The gate is: before patching an estimated direction into a model, check
that the direction's scalar projection separates the two classes on a
genuinely held-out split. This paper's own instance of that gate ran at
n=11 (3 positives, 8 negatives), and "n is small" was previously the
only characterization offered. This script replaces that with an exact
calculation.

Under the null hypothesis that the direction carries no information, the
held-out AUROC is a rescaled Mann-Whitney U statistic whose distribution
is EXACT and enumerable: all C(n_pos+n_neg, n_pos) label arrangements
are equally likely. We compute that distribution exactly (via the
standard U-statistic recursion, which is equivalent to enumeration but
tractable at any n), and report, for each (n_pos, n_neg):

  * n_arrangements  -- the size of the discrete sample space, i.e. how
    coarse the achievable p-value grid is;
  * auroc_crit_one_sided_05 -- the smallest OBSERVED held-out AUROC that
    could ever be declared significant at one-sided alpha=0.05. This is a
    property of the observable statistic, not a bound on the true effect:
    a true AUROC below this value can still produce an observation above
    it, just infrequently (see the power columns, which quantify exactly
    how infrequently);
  * attainable_alpha -- the exact size of that test (discreteness means
    it is usually well below 0.05);
  * false_accept_naive_rule -- the exact probability that a pure-noise
    direction passes the naive gate "held-out AUROC > 0.5";
  * false_accept_rule_p75 / _p80 -- same for the stricter thresholds
    AUROC >= 0.75 and >= 0.80, which is the quantity a reader needs in
    order to interpret a gate that is passed;
  * power at true AUROCs 0.60-0.95 under a binormal alternative
    (positives ~ N(mu,1), negatives ~ N(0,1), mu = sqrt(2)*Phi^-1(A)),
    20,000 Monte Carlo draws.

Class ratio is held at this paper's own gate (3:8 = 27% positive), so
the table reads as "what would this gate have been able to do with a
larger held-out set drawn the same way."
"""
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "direction_validity_mde_table.json"
N_MC = 2000000
TRUE_AUROCS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RNG = np.random.default_rng(7)


def exact_u_null(n_pos, n_neg):
    """Exact null pmf of the Mann-Whitney U statistic (0..n_pos*n_neg).
    counts[u] = number of label arrangements giving U = u; total = C(n,n_pos)."""
    max_u = n_pos * n_neg
    # dp over positives: number of ways to get rank-sum offset u
    dp = np.zeros(max_u + 1, dtype=object)
    dp[0] = 1
    # standard recursion: f(m,n,u) = f(m-1,n,u-n) + f(m,n-1,u)
    f = [[None] * (n_neg + 1) for _ in range(n_pos + 1)]
    for m in range(n_pos + 1):
        for n in range(n_neg + 1):
            arr = np.zeros(m * n + 1, dtype=object)
            if m == 0 or n == 0:
                arr[0] = 1
            else:
                a = f[m - 1][n]
                b = f[m][n - 1]
                for u, c in enumerate(a):
                    if c:
                        arr[u + n] += c
                for u, c in enumerate(b):
                    if c:
                        arr[u] += c
            f[m][n] = arr
    counts = f[n_pos][n_neg]
    total = comb(n_pos + n_neg, n_pos)
    assert int(sum(counts)) == total, (int(sum(counts)), total)
    pmf = np.array([float(c) / total for c in counts])
    return pmf


def auroc_grid(n_pos, n_neg):
    return np.arange(n_pos * n_neg + 1) / (n_pos * n_neg)


def mc_auroc(n_pos, n_neg, true_auroc, n_mc=N_MC):
    mu = sqrt(2.0) * norm.ppf(true_auroc)
    # Chunked over draws so peak memory is bounded regardless of n_mc: the
    # naive (n_mc, n_pos, n_neg) broadcast tensor is infeasible for the
    # larger Table C cells (e.g. n_pos=10, n_neg=475 at n_mc=2e6 is ~9.5e9
    # elements). Cap each chunk's element count instead of its draw count.
    max_elems_per_chunk = 200_000_000
    chunk = max(1, min(n_mc, max_elems_per_chunk // max(1, n_pos * n_neg)))
    out = np.empty(n_mc, dtype=np.float64)
    start = 0
    while start < n_mc:
        end = min(n_mc, start + chunk)
        pos = RNG.normal(mu, 1.0, size=(end - start, n_pos))
        neg = RNG.normal(0.0, 1.0, size=(end - start, n_neg))
        # U = number of (pos, neg) pairs with pos > neg (ties measure-zero)
        u = (pos[:, :, None] > neg[:, None, :]).sum(axis=(1, 2))
        out[start:end] = u / (n_pos * n_neg)
        start = end
    return out


def characterize(n_pos, n_neg):
    """Exact operating characteristics of the gate at a given (n_pos, n_neg)."""
    n_tot = n_pos + n_neg
    pmf = exact_u_null(n_pos, n_neg)
    grid = auroc_grid(n_pos, n_neg)
    # one-sided upper-tail p at each attainable AUROC
    tail = np.cumsum(pmf[::-1])[::-1]      # P(U >= u)
    ok = np.where(tail <= 0.05)[0]
    crit_idx = int(ok[0]) if len(ok) else None
    crit = float(grid[crit_idx]) if crit_idx is not None else None
    alpha_att = float(tail[crit_idx]) if crit_idx is not None else None
    # lower tail (anti-predictive direction)
    ltail = np.cumsum(pmf)
    lok = np.where(ltail <= 0.05)[0]
    crit_lo = float(grid[int(lok[-1])]) if len(lok) else None
    row = {
        "n_pos": n_pos, "n_neg": n_neg, "n_total": n_tot,
        "n_arrangements": comb(n_tot, n_pos),
        "auroc_crit_one_sided_05": crit,
        "attainable_alpha_at_crit": alpha_att,
        "auroc_crit_lower_tail_05": crit_lo,
        "false_accept_naive_gt_half": float(tail[np.searchsorted(grid, 0.5, side="right")]),
        "false_accept_rule_auroc_ge_0p75": float(tail[np.searchsorted(grid, 0.75 - 1e-12)]),
        "false_accept_rule_auroc_ge_0p80": float(tail[np.searchsorted(grid, 0.80 - 1e-12)]),
        "power": {},
    }
    if crit is not None:
        for A in TRUE_AUROCS:
            sim = mc_auroc(n_pos, n_neg, A)
            row["power"][f"{A:.2f}"] = float((sim >= crit - 1e-12).mean())
    print(f"n={n_tot} ({n_pos}+{n_neg}): arrangements={row['n_arrangements']:.3g}, "
          f"MDE(AUROC crit, one-sided .05)={crit:.4f}, exact alpha={alpha_att:.4f}, "
          f"P(noise passes 'AUROC>0.5')={row['false_accept_naive_gt_half']:.3f}, "
          f"P(noise passes 'AUROC>=0.75')={row['false_accept_rule_auroc_ge_0p75']:.4f}, "
          f"P(noise passes 'AUROC>=0.80')={row['false_accept_rule_auroc_ge_0p80']:.4f}, "
          f"power@0.75={row['power'].get('0.75'):.3f}, power@0.90={row['power'].get('0.90'):.3f}",
          flush=True)
    return row


def main():
    # --- Table A: both classes grow together, at this paper's 3:8 ratio.
    print("=== Table A: n_neg = round(n_pos * 8/3) (both classes grow) ===", flush=True)
    rows = [characterize(n_pos, int(round(n_pos * 8 / 3))) for n_pos in (3, 5, 8, 10, 15, 20, 30)]

    # --- Table B: POSITIVES HELD AT 3, negatives varied.
    # The 3:8 holdout this paper actually ran was not a data limit. The
    # tier-1 kernel caps the training pool at TRAIN_N_PER_CLASS=40 per class
    # and takes the last 20% of each class as the validity holdout, giving 8
    # negatives -- but the judge-labeled pool contains 507 hallucinated
    # items, of which only 40 are used at all. The remaining 467 are the
    # causal test pool and were never eligible to be direction-fit data, so
    # 475 negatives (507 minus the 32 spent on direction-fitting) are
    # available for the holdout at zero additional data-collection cost.
    # Positives are the genuinely binding constraint: there are only 27
    # judge-correct items in the entire pool.
    print("\n=== Table B: n_pos held at 3, n_neg varied (negatives are not scarce) ===",
          flush=True)
    neg_rows = [characterize(3, n_neg) for n_neg in (8, 20, 40, 60, 100, 200, 475, 507)]

    # --- Table C: a small 2-D grid, so the gate can be read as a function of
    # BOTH class counts rather than of a single "held-out size".
    print("\n=== Table C: 2-D grid over (n_pos, n_neg) ===", flush=True)
    grid_rows = [characterize(np_, nn) for np_ in (3, 5, 10) for nn in (8, 60, 475)]

    # This paper's own gate cell, spelled out
    pmf = exact_u_null(3, 8)
    grid = auroc_grid(3, 8)
    tail = np.cumsum(pmf[::-1])[::-1]     # P(U >= u)
    ltail = np.cumsum(pmf)                # P(U <= u)

    def exact_p(auroc):
        """Exact one- and two-sided Mann-Whitney p for an observed AUROC at
        this (3, 8) split. Two-sided uses the doubled smaller tail, the
        convention scipy.stats.mannwhitneyu(method='exact') reports."""
        idx = int(round(auroc * 24))
        lo, hi = float(ltail[idx]), float(tail[idx])
        return {"auroc": float(grid[idx]), "U": idx,
                "p_one_sided_lower": lo, "p_one_sided_upper": hi,
                "p_two_sided": float(min(1.0, 2 * min(lo, hi)))}

    observed = {"L8_ffn": 1 / 12, "L8_attn": 1 / 12, "L9_ffn": 0.0, "L9_attn": 0.125}
    paper_cell = {
        "n_pos": 3, "n_neg": 8, "n_total": 11, "n_arrangements": comb(11, 3),
        "n_attainable_auroc_values": len(grid),
        "attainable_auroc_values": [round(float(g), 4) for g in grid],
        "null_upper_tail_p": [round(float(t), 5) for t in tail],
        "observed_holdout_aurocs": {k: round(v, 4) for k, v in observed.items()},
        "observed_exact_mannwhitney": {k: exact_p(v) for k, v in observed.items()},
    }
    print("\nExact Mann-Whitney p for the four observed held-out AUROCs (n_pos=3, n_neg=8):")
    for k, v in paper_cell["observed_exact_mannwhitney"].items():
        print(f"  {k}: AUROC={v['auroc']:.4f}  one-sided(lower)={v['p_one_sided_lower']:.4f}  "
              f"two-sided={v['p_two_sided']:.4f}")
    with open(OUT_PATH, "w") as f:
        json.dump({"class_ratio": "n_neg = round(n_pos * 8/3), matching this paper's 3:8 gate",
                   "n_monte_carlo": N_MC, "true_aurocs": TRUE_AUROCS,
                   "rows": rows,
                   "negative_sweep_note":
                       "n_pos fixed at 3 (the binding constraint: only 27 judge-correct items "
                       "exist in the whole pool, and 15 of them are spent on direction-fitting). "
                       "n_neg is NOT a data limit: the pool has 507 judge-hallucinated items, of "
                       "which the tier-1 kernel uses only 40 (TRAIN_N_PER_CLASS=40), leaving 475 "
                       "eligible for the holdout after the 32 spent on direction-fitting.",
                   "negative_sweep_rows": neg_rows,
                   "two_dim_grid_rows": grid_rows,
                   "paper_gate_cell_n11": paper_cell}, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

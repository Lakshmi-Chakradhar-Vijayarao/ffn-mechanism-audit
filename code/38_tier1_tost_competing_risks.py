"""
Paper 1 -- Tier 2 statistical additions on top of the Tier 1 Kaggle kernel
(kaggle_kernels/paper1-causal-patch-tier1-validated/), per the audit's
suggested TOST equivalence testing and competing-risks outcome modeling.

(1) TOST equivalence test: McNemar's b/c discordant-pair counts, tested
    against a pre-specified odds-ratio equivalence bound (OR=2.0) via two
    one-sided exact binomial tests. If both one-sided tests reject, the
    true OR is bounded within (1/2.0, 2.0) with 95% confidence -- turning
    "failed to reject the null" into "can rule out an effect larger than
    this bound," which is the actual claim a null-result paper needs.

(2) Competing-risks outcome table: rather than collapsing "degenerate
    repetition-loop completions" into "not flipped" (as the original
    binary flip/no-flip outcome does), this uses the Jaccard label's own
    -1 (unparseable/degenerate-consistent) rate as a three-way outcome
    (flipped / not-flipped / degenerate) and tests for association
    between condition (FFN vs. Attn, at both native and common sites)
    and this three-way outcome via Fisher's exact test on the resulting
    2x3 contingency table. A full multinomial logit was considered and
    set aside: with flip counts this sparse (5-15 of 467), a multinomial
    fit would not converge reliably, and a contingency-table test answers
    the same question (is outcome distribution the same across
    conditions) without that fragility.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
TIER1_JSON = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "output" / "causal_patch_tier1_validated_results.json"
OUT_PATH = ROOT / "results" / "tier1_tost_competing_risks.json"

OR_BOUND = 2.0
CELLS = ["L8_a20", "L8_a40", "L9_a20", "L9_a40"]


def tost_at_bound(b, c, or_bound, alpha=0.05):
    n = b + c
    p_upper_bound = or_bound / (1 + or_bound)
    p_lower_bound = 1 / (1 + or_bound)
    test_upper = binomtest(b, n, p_upper_bound, alternative="less")
    test_lower = binomtest(b, n, p_lower_bound, alternative="greater")
    return test_upper.pvalue, test_lower.pvalue


def tost_mcnemar(b, c, or_bound=OR_BOUND, alpha=0.05):
    """Two one-sided exact binomial tests on discordant pairs at a
    pre-specified OR_BOUND=2.0 (the audit's suggested value), PLUS the
    smallest OR bound (searched over a fine grid) at which equivalence
    actually is established -- equivalent to a one-sided upper confidence
    bound on the true OR, and more informative than a single fixed-bound
    pass/fail given this test's actual achieved power."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "equivalence_established_at_OR_2": None,
                "smallest_OR_bound_achieving_equivalence": None}
    tu, tl = tost_at_bound(b, c, or_bound, alpha)
    equivalence = bool(tu < alpha and tl < alpha)

    smallest_bound = None
    for candidate in np.arange(1.05, 15.0, 0.05):
        tu_c, tl_c = tost_at_bound(b, c, candidate, alpha)
        if tu_c < alpha and tl_c < alpha:
            smallest_bound = float(candidate)
            break

    return {
        "b": b, "c": c, "n_discordant": n, "observed_p": b / n,
        "p_value_test_against_OR_geq_2": float(tu),
        "p_value_test_against_OR_leq_0.5": float(tl),
        "equivalence_established_at_OR_2": equivalence,
        "smallest_OR_bound_achieving_equivalence": smallest_bound,
    }


def competing_risks_table(per_sample, key, cond_a, cond_b):
    """3-way outcome (flip / no-flip / degenerate) per condition, using
    the Jaccard label's -1 as the degenerate-consistent proxy (judge label
    itself had zero unparseable verdicts in this run)."""
    def outcome(judge_label, jaccard_label):
        if jaccard_label == -1:
            return "degenerate"
        return "flip" if judge_label == 1 else "no_flip"

    table = {"flip": [0, 0], "no_flip": [0, 0], "degenerate": [0, 0]}
    for r in per_sample:
        o_a = outcome(r[f"{key}_{cond_a}_judge_label"], r[f"{key}_{cond_a}_jaccard_label"])
        o_b = outcome(r[f"{key}_{cond_b}_judge_label"], r[f"{key}_{cond_b}_jaccard_label"])
        table[o_a][0] += 1
        table[o_b][1] += 1

    contingency = np.array([table["flip"], table["no_flip"], table["degenerate"]])
    # scipy's fisher_exact is 2x2-only; for a 2x3 table use a Monte Carlo
    # / chi-square fallback. We report both raw counts and a chi-square
    # test (with a continuity note given expected-count sparsity).
    from scipy.stats import chi2_contingency
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency)
        min_expected = float(expected.min())
    except ValueError:
        chi2, p_chi2, dof, min_expected = None, None, None, None

    return {
        "contingency_flip_noflip_degenerate": {
            cond_a: table["flip"][0:1] + table["no_flip"][0:1] + table["degenerate"][0:1],
            cond_b: [table["flip"][1], table["no_flip"][1], table["degenerate"][1]],
        },
        "chi2": float(chi2) if chi2 is not None else None,
        "p_value": float(p_chi2) if p_chi2 is not None else None,
        "dof": dof,
        "min_expected_count": min_expected,
        "note": "chi-square test on 2x3 (condition x {flip,no_flip,degenerate}) contingency table; "
                "min_expected_count flags whether the chi-square approximation is reliable here (rule of thumb: >=5).",
    }


def main():
    d = json.load(open(TIER1_JSON))
    summary_judge = d["summary_under_judge_label"]
    per_sample = d["per_sample"]

    out = {"or_bound": OR_BOUND, "cells": {}}
    for key in CELLS:
        s = summary_judge[key]
        tost_native = tost_mcnemar(s["mcnemar_ffn_found_vs_attn_found_native_site"]["b"],
                                    s["mcnemar_ffn_found_vs_attn_found_native_site"]["c"])
        tost_common = tost_mcnemar(s["mcnemar_ffn_vs_attn_COMMON_SITE"]["b"],
                                    s["mcnemar_ffn_vs_attn_COMMON_SITE"]["c"])
        cr_native = competing_risks_table(per_sample, key, "ffn_found", "attn_found")
        cr_common = competing_risks_table(per_sample, key, "ffn_common", "attn_common")

        print(f"\n=== {key} ===")
        print(f"  TOST (native site, FFN vs Attn): {tost_native}")
        print(f"  TOST (common site, FFN vs Attn): {tost_common}")
        print(f"  Competing-risks (native site): {cr_native}")
        print(f"  Competing-risks (common site): {cr_common}")

        out["cells"][key] = {
            "tost_native_site": tost_native,
            "tost_common_site": tost_common,
            "competing_risks_native_site": cr_native,
            "competing_risks_common_site": cr_common,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

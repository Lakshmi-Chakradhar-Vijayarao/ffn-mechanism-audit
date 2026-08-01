"""
Paper 1 -- rerunning the direction-validity gate with the negatives the
testbed actually has, instead of the 8 the kernel's arbitrary cap left it.

WHAT WAS WRONG
--------------
Section 4.3 described its 3-positive / 8-negative holdout as "the n=11
holdout this testbed supports." That is a property of a constant, not of
the data. The tier-1 kernel sets TRAIN_N_PER_CLASS = 40 and builds its
train pool as (up to) 40 items per class; the validity holdout is the
last 20% of each class in that pool, i.e. 18//5 = 3 correct and 40//5 = 8
hallucinated. But the judge-labeled pool contains 507 hallucinated items.
Only 40 of them are ever used for anything upstream of the gate; the
other 467 sit in the causal test pool, and none of the 475 outside the
direction-fit set contributed to estimating the direction. They were
therefore eligible to be gate negatives all along, at zero additional
data-collection or compute cost.

Positives really are scarce: the entire pool has 27 judge-correct items,
of which the seed-42 split spends 15 on direction-fitting, so 3 in the
holdout is close to what a 3:1 fit/holdout split of an 18-item positive
train pool allows. Positives, not "held-out size," are the binding
constraint.

WHAT THIS SCRIPT DOES
---------------------
Reproduces the tier-1 kernel's seed-42 split exactly (verified: it
returns the kernel's own held-out AUROCs of 0.0833 / 0.0833 / 0.0 /
0.125 at the 3:8 holdout), then re-scores the SAME three held-out
positives against ALL negatives outside the direction-fit set, and
reports the exact Mann-Whitney p under the enlarged holdout. It also
reruns the 200-resplit diagnostic under the enlarged holdout, so the
enlarged-gate result is not itself a single-draw claim.

Nothing about the direction changes: it is fit on exactly the same 47
items as before. Only the set of negatives it is scored against grows.

Activations come from results/gpt2_lasttoken_L8L9.npz (written by
code/48), so this runs with no model calls when that cache is present;
*.npz is excluded from the released repository, and the script rebuilds
the cache via code/48's extraction if it is missing.
"""
import importlib.util
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
CACHE_PATH = ROOT / "results" / "gpt2_lasttoken_L8L9.npz"
OUT_PATH = ROOT / "results" / "enlarged_negative_holdout_gate.json"

LAYERS = (8, 9)
TRAIN_N_PER_CLASS = 40
KERNEL_SEED = 42
N_RESPLITS = 200
KERNEL_REPORTED = {"L8_ffn": 1 / 12, "L8_attn": 1 / 12, "L9_ffn": 0.0, "L9_attn": 0.125}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def exact_u_null(n_pos, n_neg):
    """Exact null pmf of the Mann-Whitney U statistic (same recursion as
    code/51)."""
    f = [[None] * (n_neg + 1) for _ in range(n_pos + 1)]
    for m in range(n_pos + 1):
        for n in range(n_neg + 1):
            arr = np.zeros(m * n + 1, dtype=object)
            if m == 0 or n == 0:
                arr[0] = 1
            else:
                for u, c in enumerate(f[m - 1][n]):
                    if c:
                        arr[u + n] += c
                for u, c in enumerate(f[m][n - 1]):
                    if c:
                        arr[u] += c
            f[m][n] = arr
    counts = f[n_pos][n_neg]
    total = comb(n_pos + n_neg, n_pos)
    assert int(sum(counts)) == total
    return np.array([float(c) / total for c in counts])


def auroc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def split_at_seed(labels, seed):
    """The tier-1 kernel's split construction, byte-identical to code/46's
    replication of it. Returns (fit_idx, holdout_pos_idx, holdout_neg_small,
    holdout_neg_all)."""
    correct_idx = np.where(labels == 1)[0].copy()
    hall_idx = np.where(labels == 0)[0].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(correct_idx)
    rng.shuffle(hall_idx)

    n_train_correct = min(TRAIN_N_PER_CLASS,
                          int(0.7 * len(correct_idx)) if len(correct_idx) < TRAIN_N_PER_CLASS * 2
                          else TRAIN_N_PER_CLASS)
    n_train_hall = min(TRAIN_N_PER_CLASS,
                       int(0.7 * len(hall_idx)) if len(hall_idx) < TRAIN_N_PER_CLASS * 2
                       else TRAIN_N_PER_CLASS)
    n_train_correct = max(n_train_correct, min(5, len(correct_idx)))
    n_train_hall = max(n_train_hall, min(5, len(hall_idx)))

    train_correct_idx = correct_idx[:n_train_correct]
    train_hall_idx = hall_idx[:n_train_hall]
    n_val_correct = max(1, n_train_correct // 5)
    n_val_hall = max(1, n_train_hall // 5)

    fit_pos = train_correct_idx[:-n_val_correct]
    fit_neg = train_hall_idx[:-n_val_hall]
    hold_pos = train_correct_idx[-n_val_correct:]
    hold_neg_small = train_hall_idx[-n_val_hall:]
    # every negative NOT used to fit the direction is an eligible gate negative
    hold_neg_all = np.setdiff1d(hall_idx, fit_neg, assume_unique=False)
    return (np.concatenate([fit_pos, fit_neg]), hold_pos, hold_neg_small, hold_neg_all)


def gate_auroc(vecs, labels, fit_idx, pos_idx, neg_idx):
    fit_lab = labels[fit_idx]
    d = vecs[fit_idx][fit_lab == 1].mean(axis=0) - vecs[fit_idx][fit_lab == 0].mean(axis=0)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return None
    d = d / n
    idx = np.concatenate([pos_idx, neg_idx])
    return auroc(labels[idx], vecs[idx] @ d)

def load_activations(prompts):
    """Read the L8/L9 last-token cache written by code/48; if it is absent
    (the released repository excludes *.npz), rebuild it by calling code/48's
    own extraction so this script is standalone."""
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        return {(l, s): z[f"L{l}_{s}"] for l in LAYERS for s in ("mlp", "attn")}
    c48 = load_module(ROOT / "code" / "48_permuted_pseudocategory_control.py", "c48")
    return c48.get_activations(prompts)


def main():
    kernel = load_module(KERNEL_PATH, "tier1_kernel")
    prompts = kernel.load_labeled_data()["prompts"]
    y = np.array(json.load(open(JUDGE_LABELS_PATH))["judge_labels"])
    assert len(y) == len(prompts)
    acts = {k: v.astype(np.float64) for k, v in load_activations(prompts).items()}

    fit_idx, hp, hn_small, hn_all = split_at_seed(y, KERNEL_SEED)
    print(f"seed {KERNEL_SEED}: direction-fit n={len(fit_idx)} "
          f"({int(y[fit_idx].sum())} correct, {int((y[fit_idx] == 0).sum())} hallucinated)")
    print(f"holdout positives: {len(hp)} | kernel holdout negatives: {len(hn_small)} | "
          f"all eligible negatives: {len(hn_all)} (of {int((y == 0).sum())} in the pool)")

    pmf_small = exact_u_null(len(hp), len(hn_small))
    pmf_big = exact_u_null(len(hp), len(hn_all))

    def exact_p(pmf, n_pos, n_neg, a):
        idx = int(round(a * n_pos * n_neg))
        lo = float(np.cumsum(pmf)[idx])
        hi = float(np.cumsum(pmf[::-1])[::-1][idx])
        return {"p_one_sided_lower": lo, "p_one_sided_upper": hi,
                "p_two_sided": float(min(1.0, 2 * min(lo, hi)))}

    cells = {}
    for layer_idx in LAYERS:
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            key = f"L{layer_idx}_{component}"
            X = acts[(layer_idx, sublayer)]
            a_small = gate_auroc(X, y, fit_idx, hp, hn_small)
            a_big = gate_auroc(X, y, fit_idx, hp, hn_all)
            assert abs(a_small - KERNEL_REPORTED[key]) < 1e-9, (key, a_small)

            # the same gate over 200 resplits, at both holdout sizes
            small_re, big_re = [], []
            for s in range(N_RESPLITS):
                f2, p2, ns2, na2 = split_at_seed(y, s)
                v1 = gate_auroc(X, y, f2, p2, ns2)
                v2 = gate_auroc(X, y, f2, p2, na2)
                if v1 is not None:
                    small_re.append(v1)
                if v2 is not None:
                    big_re.append(v2)
            small_re, big_re = np.array(small_re), np.array(big_re)

            cells[key] = {
                "kernel_holdout_3_8": {
                    "n_pos": len(hp), "n_neg": len(hn_small), "auroc": a_small,
                    **exact_p(pmf_small, len(hp), len(hn_small), a_small),
                    "resplit_mean": float(small_re.mean()),
                    "resplit_sd": float(small_re.std(ddof=1)),
                },
                "enlarged_holdout_all_eligible_negatives": {
                    "n_pos": len(hp), "n_neg": len(hn_all), "auroc": a_big,
                    **exact_p(pmf_big, len(hp), len(hn_all), a_big),
                    "resplit_mean": float(big_re.mean()),
                    "resplit_sd": float(big_re.std(ddof=1)),
                },
            }
            c = cells[key]
            print(f"{key}: 3:8 AUROC={a_small:.4f} (two-sided exact p="
                  f"{c['kernel_holdout_3_8']['p_two_sided']:.4f}, resplit mean="
                  f"{c['kernel_holdout_3_8']['resplit_mean']:.4f}) -> "
                  f"3:{len(hn_all)} AUROC={a_big:.4f} (two-sided exact p="
                  f"{c['enlarged_holdout_all_eligible_negatives']['p_two_sided']:.4f}, "
                  f"resplit mean={c['enlarged_holdout_all_eligible_negatives']['resplit_mean']:.4f})",
                  flush=True)

    out = {
        "n_items": int(len(y)),
        "n_judge_correct": int(y.sum()),
        "n_judge_hallucinated": int((y == 0).sum()),
        "kernel_seed": KERNEL_SEED,
        "train_n_per_class_cap": TRAIN_N_PER_CLASS,
        "n_direction_fit": int(len(fit_idx)),
        "n_holdout_positives": int(len(hp)),
        "n_holdout_negatives_kernel": int(len(hn_small)),
        "n_holdout_negatives_eligible": int(len(hn_all)),
        "note": "The direction is identical in both columns -- fit on the same 47 items. Only the "
                "set of negatives it is scored against changes. The 8-negative holdout was a "
                "consequence of TRAIN_N_PER_CLASS=40, not of data availability.",
        "n_resplits": N_RESPLITS,
        "cells": cells,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

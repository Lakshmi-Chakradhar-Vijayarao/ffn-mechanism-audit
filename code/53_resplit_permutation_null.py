"""
Paper 1 -- label-permutation null for the 200-resplit direction-validity
diagnostic (code/46_direction_validity_resplit_diagnostic.py).

WHY THIS TEST IS NEEDED
-----------------------
code/46 reports that, redrawing which items land in the direction-fit and
held-out roles at 200 random seeds, the held-out AUROC has mean
0.54-0.58 -- "centered at or slightly above chance." Section 4.3 then
moved on without testing that statement.

A reader who treats the 200 resplits as independent draws would compute
z = (mean - 0.5) / (sd / sqrt(200)) and get z = 2.65-5.62, i.e. would
conclude the direction DOES carry signal -- the opposite of this paper's
conclusion. That reading is wrong, because the 200 resplits are not
independent: every resplit draws its positives from the same 27
judge-correct items and its negatives from the same 507 judge-hallucinated
items, so the resplit means are strongly dependent and the naive standard
error is far too small.

THE CORRECT NULL
----------------
Permute the 534 judge labels (which preserves the 27/507 class counts and
therefore the entire split structure), rerun the WHOLE 200-resplit
procedure under each permutation, and record the resulting resplit mean.
The distribution of those permuted means is the correct null for the
observed resplit mean, because it inherits exactly the same dependence
structure. We report the null mean and SD, the naive (wrong) z computed
against sd/sqrt(200), the correct z against the permutation SD, and a
one-sided empirical p.

Activations come from results/gpt2_lasttoken_L8L9.npz (written by
code/48; same 534 prompts, same last-token extraction, index-aligned), so
this runs with no model calls when that cache is present; *.npz is
excluded from the released repository, and the script rebuilds the cache
via code/48's extraction if it is missing. The split construction is
byte-identical to code/46's, which is itself byte-identical to the
tier-1 kernel's.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
CACHE_PATH = ROOT / "results" / "gpt2_lasttoken_L8L9.npz"
OUT_PATH = ROOT / "results" / "resplit_permutation_null.json"

LAYERS = (8, 9)
TRAIN_N_PER_CLASS = 40
N_RESPLITS = 200
N_PERM = 500
PERM_SEED = 20260801


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fast_auroc(labels, scores):
    """Mann-Whitney AUROC with mid-ranks for ties. Small-n, no sklearn
    overhead -- this is called ~400k times."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def resplit_mean(vecs, labels, n_resplits=N_RESPLITS):
    """code/46's exact 200-resplit procedure, returning the mean held-out
    AUROC (and the per-resplit values)."""
    correct_idx_all = np.where(labels == 1)[0]
    hall_idx_all = np.where(labels == 0)[0]
    aurocs = []
    for seed in range(n_resplits):
        rng = np.random.default_rng(seed)
        correct_idx = correct_idx_all.copy()
        hall_idx = hall_idx_all.copy()
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
        fit_idx = np.concatenate([train_correct_idx[:-n_val_correct], train_hall_idx[:-n_val_hall]])
        holdout_idx = np.concatenate([train_correct_idx[-n_val_correct:], train_hall_idx[-n_val_hall:]])

        fit_vecs, fit_labels = vecs[fit_idx], labels[fit_idx]
        direction = fit_vecs[fit_labels == 1].mean(axis=0) - fit_vecs[fit_labels == 0].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        a = fast_auroc(labels[holdout_idx], vecs[holdout_idx] @ direction)
        if a is not None:
            aurocs.append(a)
    aurocs = np.array(aurocs)
    return float(aurocs.mean()), float(aurocs.std(ddof=1)), len(aurocs), aurocs

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    kernel = load_module(KERNEL_PATH, "tier1_kernel")
    prompts = kernel.load_labeled_data()["prompts"]
    y = np.array(json.load(open(JUDGE_LABELS_PATH))["judge_labels"])
    assert len(y) == len(prompts)
    print(f"{len(y)} items, {int(y.sum())} judge-correct, {int((y == 0).sum())} judge-hallucinated",
          flush=True)

    acts = {k: v.astype(np.float64) for k, v in load_activations(prompts).items()}

    rng = np.random.default_rng(PERM_SEED)
    perms = [rng.permutation(y) for _ in range(args.n_perm)]

    cells = {}
    for layer_idx in LAYERS:
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            key = f"L{layer_idx}_{component}"
            X = acts[(layer_idx, sublayer)]

            obs_mean, obs_sd, obs_n, obs_all = resplit_mean(X, y)
            print(f"{key}: observed resplit mean={obs_mean:.4f} (sd={obs_sd:.4f}, n={obs_n})",
                  flush=True)

            res = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(resplit_mean)(X, p) for p in perms)
            null_means = np.array([r[0] for r in res])

            naive_se = obs_sd / np.sqrt(obs_n)
            naive_z = (obs_mean - 0.5) / naive_se
            perm_sd = float(null_means.std(ddof=1))
            perm_z = (obs_mean - float(null_means.mean())) / perm_sd
            p_one = float((null_means.sum() * 0 + (null_means >= obs_mean).sum() + 1) /
                          (len(null_means) + 1))

            cells[key] = {
                "observed_resplit_mean": obs_mean,
                "observed_resplit_sd_across_resplits": obs_sd,
                "n_resplits": obs_n,
                "naive_independent_se": float(naive_se),
                "naive_independent_z_vs_0p5": float(naive_z),
                "null_mean": float(null_means.mean()),
                "null_sd": perm_sd,
                "null_min": float(null_means.min()),
                "null_max": float(null_means.max()),
                "null_p2p5": float(np.percentile(null_means, 2.5)),
                "null_p97p5": float(np.percentile(null_means, 97.5)),
                "se_inflation_factor_vs_naive": float(perm_sd / naive_se),
                "permutation_z": float(perm_z),
                "permutation_p_one_sided": p_one,
                "n_permutations": int(len(null_means)),
            }
            print(f"  null mean={null_means.mean():.4f} sd={perm_sd:.4f} | naive z={naive_z:.2f} "
                  f"| permutation z={perm_z:.2f} | one-sided empirical p={p_one:.3f} "
                  f"| SE inflation={perm_sd / naive_se:.2f}x", flush=True)

    out = {
        "n_items": int(len(y)),
        "n_correct": int(y.sum()),
        "n_resplits_per_run": N_RESPLITS,
        "n_permutations": int(args.n_perm),
        "permutation_seed": PERM_SEED,
        "procedure": "permute the 534 judge labels (class counts preserved), rerun the entire "
                     "200-resplit direction-fit/holdout procedure of code/46, record the mean "
                     "held-out AUROC; the distribution of those means is the null for the "
                     "observed resplit mean and inherits the same between-resplit dependence.",
        "cells": cells,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

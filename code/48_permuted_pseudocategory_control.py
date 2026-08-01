"""
Paper 1 -- permuted-pseudo-category control for the leave-one-category-out
(LOGO) CV diagnostic in code/47_category_leakage_diagnostic.py.

code/47 reported that standard 5-fold CV AUROC (0.616-0.663) "collapses"
to 0.479-0.491 under leave-one-category-out CV, and an earlier draft of
this paper read that as evidence of category-clustering leakage. This
script tests that interpretation two ways, and both refute it:

(1) TOPIC-ONLY AUROC CEILING. If category identity really carried the
    probe's signal, then a classifier that sees ONLY the category label
    (and nothing else) should be able to reproduce a large chunk of the
    0.62-0.66 standard-CV AUROC. We compute the best possible such
    classifier three ways -- in-sample (per-category correct rate fit on
    all data, an optimistic upper bound), leave-one-out, and under this
    paper's own 5-fold CV protocol (per-category rate fit on the train
    fold only, applied to the test fold) -- on the exact
    `category_correct_rates` / per-item categories the diagnostic used.

(2) PERMUTED PSEUDO-CATEGORY CONTROL, in two forms. We rerun the
    identical LOGO diagnostic (code/47::probe_leave_one_category_out,
    reproduced unmodified) on group assignments that are random by
    construction:

      (2a) SIZE-MATCHED: permute the category-assignment vector across
           items. Preserves the category-size distribution exactly while
           destroying any relationship between group membership and topic
           content.
      (2b) SIZE- AND CLASS-MATCHED (the stricter control): permute
           positives among positive slots and negatives among negative
           slots, so every pseudo-category has EXACTLY the same n and the
           same number of correct-labeled items as the real category it
           stands in for. This makes the set of LOGO-usable folds
           identical in size and class composition to the real one,
           removing the alternative explanation that the 16 usable real
           categories are simply an unusual subset.

    If random groupings reproduce the "collapse," it is a property of the
    LOGO estimator at this sample size rather than a leakage measurement.
    If they do not, the collapse is specific to real topic structure.

The LOGO estimator averages PER-CATEGORY AUROCs, which is a different
estimand from the pooled/marginal AUROC that standard CV reports; with
22 of 38 categories having zero positive-class items, the ~16 surviving
folds average ~1.7 positives each, so each fold's AUROC is noisy. Whether
that noise also biases the MEAN downward is exactly what the permutation
controls settle empirically.

Activations: same last-token FFN/Attn extraction at L8/L9 as code/47
(cached to results/gpt2_lasttoken_L8L9.npz on first run so repeat runs
need no model calls).
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
CACHE_PATH = ROOT / "results" / "gpt2_lasttoken_L8L9.npz"
DIAG_PATH = ROOT / "results" / "category_leakage_diagnostic.json"
OUT_PATH = ROOT / "results" / "permuted_pseudocategory_control.json"

LAYERS = (8, 9)
N_PERM = 1000  # was 100; at 100 the attainable two-sided p floor (0.0198) is
               # larger than the smallest Holm threshold across the four cells
               # (0.05/4 = 0.0125), so no cell could ever survive correction.
SEED = 2026


def holm_bonferroni(pvals):
    """Holm-Bonferroni adjusted p-values, order-preserving (the same
    correction used in sections 4.4 and 4.7)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def question_from_prompt(prompt: str) -> str:
    m = re.match(r"^Q: (.*)\nA:$", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


# ---------------------------------------------------------------- (1) ceiling
def topic_only_auroc_in_sample(y, cats):
    """Optimistic upper bound: score each item by its own category's
    correct rate computed on ALL data (including the item itself)."""
    rates = {c: y[cats == c].mean() for c in set(cats.tolist())}
    scores = np.array([rates[c] for c in cats])
    return float(roc_auc_score(y, scores))


def topic_only_auroc_loo(y, cats):
    """Leave-one-out: score each item by its category's correct rate with
    that item removed (global mean as fallback for singleton categories)."""
    global_rate = float(y.mean())
    scores = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        mask = (cats == cats[i])
        mask[i] = False
        scores[i] = y[mask].mean() if mask.sum() > 0 else global_rate
    return float(roc_auc_score(y, scores))


def topic_only_auroc_kfold(y, cats, n_splits=5, seed=42):
    """This paper's own 5-fold CV protocol, but with a topic-only rule:
    per-category correct rate is fit on the training fold and applied to
    the test fold (global train rate for categories unseen in train).
    Returns (pooled-OOF AUROC, mean-of-folds AUROC)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=float)
    fold_aurocs = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        g = float(y[tr].mean())
        rates = {}
        for c in set(cats[tr].tolist()):
            rates[c] = float(y[tr][cats[tr] == c].mean())
        s = np.array([rates.get(c, g) for c in cats[te]])
        oof[te] = s
        if len(np.unique(y[te])) > 1:
            fold_aurocs.append(roc_auc_score(y[te], s))
    return float(roc_auc_score(y, oof)), float(np.mean(fold_aurocs))


# ------------------------------------------------------- (2) permuted control
def probe_leave_one_category_out(X, y, groups):
    """Byte-identical logic to code/47::probe_leave_one_category_out."""
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    fold_aurocs = []
    n_skipped = 0
    for tr_idx, te_idx in logo.split(X, y, groups):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            n_skipped += 1
            continue
        probe = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
        probe.fit(X[tr_idx], y_tr)
        proba = probe.predict_proba(X[te_idx])[:, 1]
        fold_aurocs.append(roc_auc_score(y_te, proba))
    fold_aurocs = np.array(fold_aurocs)
    return (float(fold_aurocs.mean()) if len(fold_aurocs) else None,
            float(fold_aurocs.std(ddof=1)) if len(fold_aurocs) > 1 else None,
            len(fold_aurocs), n_skipped)


def within_between_topic_auroc(y, cats, scores):
    """Decompose a pooled AUROC into the pairs it is actually built from.

    AUROC is the probability that a random positive outranks a random
    negative. Standard random K-fold CV pools ALL (pos, neg) pairs, most of
    which straddle two different categories. Leave-one-category-out CV, by
    construction, only ever compares a positive and a negative drawn from
    the SAME category. So the two protocols differ in the estimand even
    before any question of leakage arises. This function separates the two
    pair sets on the identical out-of-fold scores, which is the direct way
    to see where a probe's discriminative power lives."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    w_num = w_den = b_num = b_den = 0.0
    for i in pos:
        si, ci = scores[i], cats[i]
        same = cats[neg] == ci
        for arr, is_same in ((neg[same], True), (neg[~same], False)):
            if len(arr) == 0:
                continue
            wins = float((scores[arr] < si).sum() + 0.5 * (scores[arr] == si).sum())
            if is_same:
                w_num += wins
                w_den += len(arr)
            else:
                b_num += wins
                b_den += len(arr)
    return {
        "within_topic_auroc": (w_num / w_den) if w_den else None,
        "within_topic_n_pairs": int(w_den),
        "between_topic_auroc": (b_num / b_den) if b_den else None,
        "between_topic_n_pairs": int(b_den),
        "pooled_auroc": float(roc_auc_score(y, scores)),
        "frac_pairs_within_topic": (w_den / (w_den + b_den)) if (w_den + b_den) else None,
    }


def oof_scores(X, y, n_splits=5, seed=42):
    """Out-of-fold predicted probabilities under code/47's standard CV."""
    from sklearn.model_selection import cross_val_predict
    probe = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(probe, X, y, cv=skf, method="predict_proba")[:, 1]


def stratified_permuted_groups(y, cats, rng):
    """Pseudo-categories with EXACTLY the same size and the same number of
    positive-class items as each real category, but random membership."""
    new = np.empty(len(y), dtype=object)
    pos_idx = np.where(y == 1)[0].copy()
    neg_idx = np.where(y == 0)[0].copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pi = ni = 0
    for c in sorted(set(cats.tolist())):
        m = cats == c
        npos = int(y[m].sum())
        nneg = int(m.sum()) - npos
        new[pos_idx[pi:pi + npos]] = c
        pi += npos
        new[neg_idx[ni:ni + nneg]] = c
        ni += nneg
    assert pi == len(pos_idx) and ni == len(neg_idx)
    return new.astype(str)


def get_activations(prompts):
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        return {(l, s): z[f"L{l}_{s}"] for l in LAYERS for s in ("mlp", "attn")}
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    kernel = load_module(KERNEL_PATH, "tier1_kernel")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    acts = {}
    for layer_idx in LAYERS:
        for sublayer in ("mlp", "attn"):
            vecs = [kernel.extract_last_token(p, model, tok, device, layer_idx, sublayer) for p in prompts]
            acts[(layer_idx, sublayer)] = np.stack(vecs)
            print(f"  extracted L{layer_idx}/{sublayer}", flush=True)
    np.savez_compressed(CACHE_PATH, **{f"L{l}_{s}": acts[(l, s)] for l in LAYERS for s in ("mlp", "attn")})
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    n_perm = args.n_perm

    kernel = load_module(KERNEL_PATH, "tier1_kernel")
    labeled = kernel.load_labeled_data()
    prompts = labeled["prompts"]
    y = np.array(json.load(open(JUDGE_LABELS_PATH))["judge_labels"])
    assert len(y) == len(prompts)

    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {item["question"].strip(): item["category"] for item in ds}
    cats = np.array([qmap.get(question_from_prompt(p), "UNKNOWN") for p in prompts])
    keep = cats != "UNKNOWN"
    prompts = [p for p, k in zip(prompts, keep) if k]
    y, cats = y[keep], cats[keep]
    print(f"{len(y)} items, {len(set(cats.tolist()))} categories, {int(y.sum())} correct", flush=True)

    # ---- (1) topic-only ceiling
    in_samp = topic_only_auroc_in_sample(y, cats)
    loo = topic_only_auroc_loo(y, cats)
    kf_pooled, kf_mof = topic_only_auroc_kfold(y, cats)
    print(f"Topic-only AUROC ceiling: in-sample={in_samp:.4f}  LOO={loo:.4f}  "
          f"5-fold-CV pooled={kf_pooled:.4f} mean-of-folds={kf_mof:.4f}", flush=True)

    # ---- (2) permuted pseudo-category control
    acts = get_activations(prompts)
    diag = json.load(open(DIAG_PATH))["layers"]
    rng = np.random.default_rng(SEED)
    perms_size = [cats[rng.permutation(len(cats))] for _ in range(n_perm)]
    perms_strat = [stratified_permuted_groups(y, cats, rng) for _ in range(n_perm)]

    def summarize(perm_means, perm_folds, real_mean, tag):
        perm_means = np.array(perm_means)
        pct = float((perm_means <= real_mean).mean())
        # exact two-sided empirical p with the standard +1 correction
        p_two = float(min(1.0, 2 * (min((perm_means <= real_mean).sum(),
                                        (perm_means >= real_mean).sum()) + 1) / (len(perm_means) + 1)))
        return {
            f"{tag}_logo_auroc_mean": float(perm_means.mean()),
            f"{tag}_logo_auroc_sd": float(perm_means.std(ddof=1)),
            f"{tag}_logo_auroc_min": float(perm_means.min()),
            f"{tag}_logo_auroc_max": float(perm_means.max()),
            f"{tag}_logo_auroc_p2p5": float(np.percentile(perm_means, 2.5)),
            f"{tag}_logo_auroc_p97p5": float(np.percentile(perm_means, 97.5)),
            f"{tag}_mean_n_folds_used": float(np.mean(perm_folds)),
            f"{tag}_n_permutations": int(len(perm_means)),
            f"{tag}_real_percentile_in_null": pct,
            f"{tag}_empirical_two_sided_p": p_two,
        }

    cells = {}
    for layer_idx in LAYERS:
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            key = f"L{layer_idx}_{component}"
            X = acts[(layer_idx, sublayer)]
            real_mean, real_sd, real_folds, real_skip = probe_leave_one_category_out(X, y, cats)
            std_cv = diag[key]["standard_5fold_cv_auroc_mean"]
            entry = {
                "standard_5fold_cv_auroc": std_cv,
                "real_logo_auroc_mean": real_mean,
                "real_logo_auroc_sd": real_sd,
                "real_logo_n_folds": real_folds,
                "real_logo_n_skipped": real_skip,
            }
            for tag, plist in (("permuted_size_matched", perms_size),
                               ("permuted_size_and_class_matched", perms_strat)):
                print(f"  {key}/{tag}: running {n_perm} permutations...", flush=True)
                res = Parallel(n_jobs=args.n_jobs, verbose=1)(
                    delayed(probe_leave_one_category_out)(X, y, g) for g in plist)
                pm = [r[0] for r in res if r[0] is not None]
                pf = [r[2] for r in res if r[0] is not None]
                entry.update(summarize(pm, pf, real_mean, tag))
                entry[f"{tag}_collapse_vs_standard_cv"] = float(std_cv - float(np.mean(pm)))
            entry["collapse_real_vs_standard_cv"] = float(std_cv - real_mean)
            entry["pair_decomposition"] = within_between_topic_auroc(y, cats, oof_scores(X, y))
            pd_ = entry["pair_decomposition"]
            print(f"  {key} pair decomposition: pooled={pd_['pooled_auroc']:.4f} | "
                  f"within-topic={pd_['within_topic_auroc']:.4f} ({pd_['within_topic_n_pairs']} pairs) | "
                  f"between-topic={pd_['between_topic_auroc']:.4f} ({pd_['between_topic_n_pairs']} pairs)",
                  flush=True)
            cells[key] = entry
            print(f"{key}: standardCV={std_cv:.4f} | real LOGO={real_mean:.4f} ({real_folds} folds) | "
                  f"size-matched null={entry['permuted_size_matched_logo_auroc_mean']:.4f}"
                  f"+/-{entry['permuted_size_matched_logo_auroc_sd']:.4f} "
                  f"(p={entry['permuted_size_matched_empirical_two_sided_p']:.3f}) | "
                  f"size+class-matched null={entry['permuted_size_and_class_matched_logo_auroc_mean']:.4f}"
                  f"+/-{entry['permuted_size_and_class_matched_logo_auroc_sd']:.4f} "
                  f"(p={entry['permuted_size_and_class_matched_empirical_two_sided_p']:.3f})", flush=True)

    # Holm-Bonferroni across the four cells, separately within each control
    # family -- the same correction sections 4.4 and 4.7 apply to their own
    # four-test families. Without it, four nominal p-values around the
    # permutation floor read as four independent confirmations.
    keys = list(cells)
    holm = {}
    for tag in ("permuted_size_matched", "permuted_size_and_class_matched"):
        raw = [cells[k][f"{tag}_empirical_two_sided_p"] for k in keys]
        adj = holm_bonferroni(raw)
        holm[tag] = {k: {"raw_p": raw[i], "holm_p": adj[i], "survives_holm_05": bool(adj[i] < 0.05)}
                     for i, k in enumerate(keys)}
        for i, k in enumerate(keys):
            cells[k][f"{tag}_holm_p"] = adj[i]
        print(f"Holm-Bonferroni ({tag}, m=4): " +
              ", ".join(f"{k} {raw[i]:.4f}->{adj[i]:.4f}" for i, k in enumerate(keys)), flush=True)

    out = {
        "n_items": int(len(y)),
        "n_categories": int(len(set(cats.tolist()))),
        "n_correct": int(y.sum()),
        "n_permutations": n_perm,
        "attainable_two_sided_p_floor": float(2.0 / (n_perm + 1)),
        "holm_bonferroni_across_four_cells": holm,
        "seed": SEED,
        "category_size_distribution": Counter(cats.tolist()).most_common(),
        "topic_only_auroc_ceiling": {
            "in_sample_upper_bound": in_samp,
            "leave_one_out": loo,
            "kfold_cv_pooled_oof": kf_pooled,
            "kfold_cv_mean_of_folds": kf_mof,
        },
        "cells": cells,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

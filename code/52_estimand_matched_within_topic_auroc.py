"""
Paper 1 -- estimand-matched within-topic AUROC, correcting the
training-overlap decomposition previously reported in section 4.10.

THE PROBLEM THIS SCRIPT FIXES
-----------------------------
An earlier version of section 4.10 subtracted two numbers that are not
the same estimand and read the difference as "the part of the LOGO
collapse attributable to training-set topic overlap":

  * the "within-topic" column came from
    code/48::within_between_topic_auroc, which is a PAIR-WEIGHTED POOLED
    AUROC over all 675 within-topic (positive, negative) pairs; and
  * the "LOGO" column came from
    code/47::probe_leave_one_category_out, which is an UNWEIGHTED MEAN
    of 16 PER-CATEGORY AUROCs.

Pair-weighted pooling gives a large category (Misconceptions, n=69) many
more pairs than a small one (Science, n=4); per-category averaging gives
them equal weight. Subtracting one from the other therefore mixes the
quantity of interest (what removing same-topic training data does) with
a purely mechanical re-weighting.

WHAT THIS SCRIPT COMPUTES
-------------------------
The estimand-matched comparison. Holding the AVERAGING CONVENTION fixed
at LOGO's own (unweighted mean of per-category AUROCs over exactly the
same usable categories), we compute the within-topic AUROC from the
STANDARD 5-fold CV out-of-fold scores:

  matched_within_topic = mean over usable categories c of
                         AUROC( y[c], oof_scores[c] )

which differs from the LOGO number in one and only one respect: whether
same-topic items were present in the probe's training data. The
difference between them is then a decomposition rather than an artifact.

We also report, for completeness, the pair-weighted pooled within-topic
value the paper previously used, so the size of the estimand-mismatch
artifact is visible directly.

Paired significance is tested across the 16 usable categories
(Wilcoxon signed-rank, exact; paired t-test as a secondary check), which
is the correct pairing: each category contributes one LOGO AUROC and one
standard-CV AUROC computed on the same items.

Activations come from results/gpt2_lasttoken_L8L9.npz, the cache written
by code/48 (same 534 prompts, same last-token extraction, index-aligned;
code/48 drops 0 items for unknown category, so the cache row order is the
prompt order). *.npz files are excluded from the released repository; if
the cache is absent this script rebuilds it by calling code/48's own
extraction, so it is standalone.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
CACHE_PATH = ROOT / "results" / "gpt2_lasttoken_L8L9.npz"
OUT_PATH = ROOT / "results" / "estimand_matched_within_topic_auroc.json"

LAYERS = (8, 9)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def question_from_prompt(prompt: str) -> str:
    m = re.match(r"^Q: (.*)\nA:$", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


def make_probe():
    """Leak-free: the scaler is fit inside the pipeline, per fold."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])


def logo_per_category(X, y, groups):
    """code/47::probe_leave_one_category_out, but returning the per-category
    AUROCs and the category names rather than only their mean."""
    logo = LeaveOneGroupOut()
    out = {}
    n_skipped = 0
    for tr_idx, te_idx in logo.split(X, y, groups):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            n_skipped += 1
            continue
        probe = make_probe()
        probe.fit(X[tr_idx], y_tr)
        proba = probe.predict_proba(X[te_idx])[:, 1]
        out[str(groups[te_idx][0])] = float(roc_auc_score(y_te, proba))
    return out, n_skipped


def oof_scores(X, y, n_splits=5, seed=42):
    """Standard 5-fold CV out-of-fold predicted probabilities -- code/48's
    protocol, and the protocol every passive AUROC in this paper uses."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_probe(), X, y, cv=skf, method="predict_proba")[:, 1]


def per_category_auroc_from_scores(y, cats, scores, usable):
    """LOGO's own averaging convention applied to standard-CV OOF scores."""
    return {c: float(roc_auc_score(y[cats == c], scores[cats == c])) for c in usable}


def pair_weighted_within_topic(y, cats, scores):
    """The pair-weighted pooled within-topic AUROC the paper previously
    reported (code/48::within_between_topic_auroc), reproduced here so the
    size of the estimand mismatch is visible."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    num = den = 0.0
    for i in pos:
        same = neg[cats[neg] == cats[i]]
        if len(same) == 0:
            continue
        num += float((scores[same] < scores[i]).sum() + 0.5 * (scores[same] == scores[i]).sum())
        den += len(same)
    return (num / den) if den else None, int(den)

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

    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {item["question"].strip(): item["category"] for item in ds}
    cats = np.array([qmap.get(question_from_prompt(p), "UNKNOWN") for p in prompts])
    keep = cats != "UNKNOWN"
    assert keep.all(), f"{(~keep).sum()} unmatched categories -- cache alignment assumption broken"
    print(f"{len(y)} items, {len(set(cats.tolist()))} categories, {int(y.sum())} correct", flush=True)

    acts = load_activations(prompts)

    cells = {}
    for layer_idx in LAYERS:
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            key = f"L{layer_idx}_{component}"
            X = acts[(layer_idx, sublayer)]

            logo_by_cat, n_skipped = logo_per_category(X, y, cats)
            usable = sorted(logo_by_cat)
            scores = oof_scores(X, y)
            matched_by_cat = per_category_auroc_from_scores(y, cats, scores, usable)

            logo_vec = np.array([logo_by_cat[c] for c in usable])
            matched_vec = np.array([matched_by_cat[c] for c in usable])
            diff = matched_vec - logo_vec

            pw, pw_pairs = pair_weighted_within_topic(y, cats, scores)

            try:
                w_stat, w_p = wilcoxon(matched_vec, logo_vec)
                w_stat, w_p = float(w_stat), float(w_p)
            except ValueError:
                w_stat, w_p = None, None
            t_stat, t_p = ttest_rel(matched_vec, logo_vec)

            cells[key] = {
                "n_usable_categories": len(usable),
                "n_skipped_categories": n_skipped,
                "usable_categories": usable,
                "logo_auroc_by_category": logo_by_cat,
                "logo_auroc_mean": float(logo_vec.mean()),
                "matched_within_topic_auroc_by_category": matched_by_cat,
                "matched_within_topic_auroc_mean": float(matched_vec.mean()),
                "matched_residual_mean": float(diff.mean()),
                "matched_residual_sd": float(diff.std(ddof=1)),
                "paired_wilcoxon_stat": w_stat,
                "paired_wilcoxon_p_two_sided": w_p,
                "paired_t_stat": float(t_stat),
                "paired_t_p_two_sided": float(t_p),
                "pair_weighted_within_topic_auroc": float(pw),
                "pair_weighted_within_topic_n_pairs": pw_pairs,
                "pair_weighted_residual_mean": float(pw - logo_vec.mean()),
                "pooled_oof_auroc": float(roc_auc_score(y, scores)),
            }
            print(f"{key}: LOGO={logo_vec.mean():.4f} | matched within-topic (category-averaged)="
                  f"{matched_vec.mean():.4f} | matched residual={diff.mean():+.4f} "
                  f"(Wilcoxon p={w_p:.3f}, paired t p={t_p:.3f}) || pair-weighted within-topic="
                  f"{pw:.4f} ({pw_pairs} pairs) -> misleading residual={pw - logo_vec.mean():+.4f}",
                  flush=True)

    out = {
        "n_items": int(len(y)),
        "n_correct": int(y.sum()),
        "n_categories": int(len(set(cats.tolist()))),
        "standard_cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), "
                       "Pipeline(StandardScaler, LogisticRegression(C=1.0, max_iter=1000))",
        "note": "matched_within_topic uses LOGO's own averaging convention (unweighted mean of "
                "per-category AUROCs over the same usable categories) applied to standard-CV "
                "out-of-fold scores, so LOGO-vs-matched differs only in whether same-topic items "
                "were in training. pair_weighted_within_topic is the pooled-over-pairs quantity "
                "the paper previously (incorrectly) differenced against LOGO.",
        "cells": cells,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

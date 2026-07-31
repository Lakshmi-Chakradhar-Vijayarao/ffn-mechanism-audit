"""
Paper 1 -- leave-one-category-out CV diagnostic for the FFN/Attention
component probe, addressing the single highest-priority missing check a
fresh adversarial review identified: TruthfulQA questions cluster into
38 topical categories (Misconceptions, Law, Health, ...), and standard
random K-fold CV can place near-duplicate, same-category questions in
both train and test folds. If the component probe's above-chance AUROC
is actually a category classifier riding along with the hallucination
label (categories are not independent of hallucination rate), a random
K-fold CV would not reveal this, while a CV that holds out entire
categories at once would show the AUROC collapse toward chance.

This uses the exact same real, judge-labeled GPT-2/TruthfulQA pool as
the flagship causal-patching test (534 items, prompts embedded in
kaggle_kernels/paper1-causal-patch-tier1-validated/run_causal_patch_tier1_validated.py,
judge labels in results/gpt2_full_534_judge_labels.json -- both already
verified index-aligned by code/46), plus each item's TruthfulQA category
(re-fetched from the HF dataset; category is dataset metadata, not a
model output, so no re-generation or re-judging is needed to obtain it).
Activations (last-token FFN/Attn sublayer output, same convention as
code/02_cross_arch_component_probe.py and the causal-patching kernel) are
extracted fresh at L8 and L9 -- GPT-2 is small enough to run on CPU.

Two CV protocols are compared at each layer/component:
  (1) Standard 5-fold stratified CV (identical to
      code/02::probe_component_at_layer) -- the protocol never checked
      for category leakage.
  (2) Leave-one-category-out CV (sklearn LeaveOneGroupOut on the 38
      TruthfulQA categories) -- each fold trains on 37 categories and
      tests on the 1 held-out category, so no question sharing a topic
      with a test item can appear in that fold's training data.
If (2) collapses toward chance relative to (1), this is direct evidence
the standard CV's AUROC is (at least partly) a category-leakage artifact,
not genuine hallucination signal.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "kaggle_kernels" / "paper1-causal-patch-tier1-validated" / "run_causal_patch_tier1_validated.py"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
OUT_PATH = ROOT / "results" / "category_leakage_diagnostic.json"

LAYERS = (8, 9)


def load_kernel_module():
    spec = importlib.util.spec_from_file_location("tier1_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier1_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod


def question_from_prompt(prompt: str) -> str:
    m = re.match(r"^Q: (.*)\nA:$", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


def probe_standard_cv(X, y, n_splits=5):
    actual_splits = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std()), actual_splits


def probe_leave_one_category_out(X, y, groups):
    logo = LeaveOneGroupOut()
    fold_aurocs = []
    n_skipped = 0
    for tr_idx, te_idx in logo.split(X, y, groups):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            n_skipped += 1
            continue
        probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
        probe.fit(X[tr_idx], y_tr)
        proba = probe.predict_proba(X[te_idx])[:, 1]
        fold_aurocs.append(roc_auc_score(y_te, proba))
    fold_aurocs = np.array(fold_aurocs)
    return (float(fold_aurocs.mean()) if len(fold_aurocs) else None,
            float(fold_aurocs.std(ddof=1)) if len(fold_aurocs) > 1 else None,
            len(fold_aurocs), n_skipped)


def main():
    print("Loading kernel module and labeled pool...", flush=True)
    kernel = load_kernel_module()
    labeled = kernel.load_labeled_data()
    prompts = labeled["prompts"]
    judge_data = json.load(open(JUDGE_LABELS_PATH))
    judge_labels = np.array(judge_data["judge_labels"])
    assert len(judge_labels) == len(prompts)
    print(f"Loaded {len(prompts)} prompts, {int((judge_labels==1).sum())} correct, "
          f"{int((judge_labels==0).sum())} hallucinated", flush=True)

    print("Loading TruthfulQA for category metadata...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap_category = {item["question"].strip(): item["category"] for item in ds}

    questions = [question_from_prompt(p) for p in prompts]
    categories = np.array([qmap_category.get(q, "UNKNOWN") for q in questions])
    n_unknown = int((categories == "UNKNOWN").sum())
    print(f"Category lookup: {n_unknown}/{len(categories)} unmatched (dropped below)", flush=True)
    keep_mask = categories != "UNKNOWN"

    prompts_k = [p for p, k in zip(prompts, keep_mask) if k]
    labels_k = judge_labels[keep_mask]
    categories_k = categories[keep_mask]
    n_categories = len(set(categories_k))
    print(f"Kept {len(prompts_k)} items across {n_categories} categories", flush=True)
    from collections import Counter
    print("Category counts:", Counter(categories_k.tolist()).most_common(10), flush=True)
    print("Hallucination rate by category (top 10 by count):", flush=True)
    for cat, cnt in Counter(categories_k.tolist()).most_common(10):
        rate = labels_k[categories_k == cat].mean()
        print(f"  {cat}: n={cnt}, correct_rate={rate:.3f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2.eval()

    print("Extracting FFN/Attn last-token activations at L8, L9...", flush=True)
    acts = {}
    for layer_idx in LAYERS:
        for sublayer in ("mlp", "attn"):
            vecs = []
            for i, p in enumerate(prompts_k):
                vecs.append(kernel.extract_last_token(p, gpt2, gpt2_tok, device, layer_idx, sublayer))
                if (i + 1) % 100 == 0:
                    print(f"  L{layer_idx}/{sublayer}: {i+1}/{len(prompts_k)}", flush=True)
            acts[(layer_idx, sublayer)] = np.stack(vecs)

    results = {"n_items": len(prompts_k), "n_categories": n_categories, "n_unknown_dropped": n_unknown,
               "category_counts": Counter(categories_k.tolist()).most_common(), "layers": {}}
    for layer_idx in LAYERS:
        for component, sublayer in (("ffn", "mlp"), ("attn", "attn")):
            X = acts[(layer_idx, sublayer)]
            y = labels_k
            std_mean, std_sd, std_splits = probe_standard_cv(X, y)
            logo_mean, logo_sd, logo_n_folds, logo_skipped = probe_leave_one_category_out(X, y, categories_k)
            key = f"L{layer_idx}_{component}"
            results["layers"][key] = {
                "standard_5fold_cv_auroc_mean": std_mean, "standard_5fold_cv_auroc_sd": std_sd,
                "standard_cv_n_splits": std_splits,
                "leave_one_category_out_auroc_mean": logo_mean, "leave_one_category_out_auroc_sd": logo_sd,
                "logo_n_folds_used": logo_n_folds, "logo_n_folds_skipped_single_class": logo_skipped,
            }
            print(f"{key}: standard 5-fold CV AUROC={std_mean:.4f} (sd={std_sd:.4f}) | "
                  f"leave-one-category-out AUROC={logo_mean:.4f} (sd={logo_sd:.4f}, "
                  f"{logo_n_folds} folds used, {logo_skipped} skipped)", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

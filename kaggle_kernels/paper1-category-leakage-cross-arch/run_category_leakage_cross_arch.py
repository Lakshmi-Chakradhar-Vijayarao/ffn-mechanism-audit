"""
Paper 1 -- extending the category-leakage diagnostic (found via GPT-2,
code/47_category_leakage_diagnostic.py: standard 5-fold CV AUROC of
0.62-0.66 collapses to 0.48-0.49 under leave-one-category-out CV) to the
other two architectures already in this paper's cross-architecture
component-probe claim (Sec 3.3): Pythia-410M and Qwen2.5-0.5B-Instruct.

Uses the identical generation/labeling/extraction methodology as
code/02_cross_arch_component_probe.py (Jaccard word-overlap labeling,
mean-pooled per-layer FFN/Attention sublayer activations, bare
"Q: {question}\\nA:" prompts, no chat template for Qwen -- matching the
"qwen05" config whose peak_ffn_layer=8/peak_attn_layer=17 this script
targets, not "qwen05chat"), generating fresh over the FULL 817-item
TruthfulQA validation split (not the capped subset any earlier run may
have used) so every item's category is available, then extracts
activations only at each architecture's own already-reported peak FFN
and peak Attn layers (Pythia: peak_ffn=L11, peak_attn=L4;
Qwen0.5B: peak_ffn=L8, peak_attn=L17) to keep this tractable on a single
Kaggle GPU session, rather than repeating the full 24-layer sweep.

For each architecture x layer x component cell, compares:
  (1) standard 5-fold stratified CV AUROC (identical protocol to
      code/02::probe_component_at_layer)
  (2) leave-one-category-out CV AUROC (LeaveOneGroupOut over TruthfulQA's
      38 categories, skipping any held-out category whose slice is
      single-class)
If (2) collapses toward chance relative to (1), as it did for GPT-2, this
confirms the category-leakage finding is a property of the evaluation
protocol and TruthfulQA itself, not specific to one probe/architecture.
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path("/kaggle/working")

LABEL_THRESHOLD = 0.12
MAX_NEW_TOKENS = 40

MODEL_REGISTRY = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "layers_path": lambda m: m.gpt_neox.layers,
        "mlp_attr": "mlp",
        "attn_attr": "attention",
        "chat_template": False,
        "peak_ffn_layer": 11,
        "peak_attn_layer": 4,
    },
    "qwen05": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_path": lambda m: m.model.layers,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
        "chat_template": False,
        "peak_ffn_layer": 8,
        "peak_attn_layer": 17,
    },
}


def _word_overlap(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def label_completion(completion, correct_answers, incorrect_answers) -> int:
    best_correct = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_incorrect = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_correct > LABEL_THRESHOLD or best_incorrect > LABEL_THRESHOLD:
        return 1 if best_correct >= best_incorrect else 0
    return -1


def load_model_and_tokenizer(model_key):
    cfg = MODEL_REGISTRY[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    print(f"Loaded {cfg['hf_id']} on {device}  layers={len(cfg['layers_path'](model))}  "
          f"hidden={model.config.hidden_size}", flush=True)
    return model, tokenizer, device, cfg


def generate_label_and_categorize(model, tokenizer, device, dataset, cfg):
    prompts, labels, categories = [], [], []
    use_chat_template = cfg.get("chat_template", False)
    for i, item in enumerate(dataset):
        question = item["question"]
        if use_chat_template:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        label = label_completion(completion, item["correct_answers"], item["incorrect_answers"])
        if label == -1:
            continue
        prompts.append(prompt)
        labels.append(label)
        categories.append(item["category"])
        if (i + 1) % 100 == 0:
            print(f"  generated+labeled {i + 1}/{len(dataset)}  (kept {len(labels)})", flush=True)
    print(f"Kept {len(labels)}/{len(dataset)}  correct={sum(labels)}  "
          f"hallucinated={len(labels) - sum(labels)}  categories={len(set(categories))}", flush=True)
    return prompts, np.array(labels), np.array(categories)


def extract_component_at_layer(prompts, model, tokenizer, device, cfg, layer_idx, component):
    layers = cfg["layers_path"](model)
    attr = cfg["mlp_attr"] if component == "ffn" else cfg["attn_attr"]
    vecs = []
    for prompt in prompts:
        captured = {}

        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured["v"] = t[0].mean(dim=0).detach().cpu().numpy()

        h = getattr(layers[layer_idx], attr).register_forward_hook(hook)
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                model(**inputs)
        finally:
            h.remove()
        vecs.append(captured["v"])
    return np.stack(vecs)


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
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Loaded {len(ds)} TruthfulQA validation items", flush=True)

    out = {"n_categories_total": len(set(item["category"] for item in ds)), "architectures": {}}

    for model_key in ("pythia", "qwen05"):
        print(f"\n{'='*70}\n{model_key}\n{'='*70}", flush=True)
        model, tokenizer, device, cfg = load_model_and_tokenizer(model_key)
        prompts, labels, categories = generate_label_and_categorize(model, tokenizer, device, ds, cfg)

        arch_out = {
            "n_items": len(prompts), "n_correct": int(labels.sum()),
            "n_categories_kept": len(set(categories.tolist())),
            "layers": {},
        }
        for component, layer_idx in (("ffn", cfg["peak_ffn_layer"]), ("attn", cfg["peak_attn_layer"])):
            print(f"Extracting {component} activations at L{layer_idx}...", flush=True)
            X = extract_component_at_layer(prompts, model, tokenizer, device, cfg, layer_idx, component)
            std_mean, std_sd, std_splits = probe_standard_cv(X, labels)
            logo_mean, logo_sd, logo_n_folds, logo_skipped = probe_leave_one_category_out(X, labels, categories)
            key = f"L{layer_idx}_{component}"
            arch_out["layers"][key] = {
                "standard_5fold_cv_auroc_mean": std_mean, "standard_5fold_cv_auroc_sd": std_sd,
                "standard_cv_n_splits": std_splits,
                "leave_one_category_out_auroc_mean": logo_mean, "leave_one_category_out_auroc_sd": logo_sd,
                "logo_n_folds_used": logo_n_folds, "logo_n_folds_skipped_single_class": logo_skipped,
            }
            print(f"{model_key} {key}: standard CV AUROC={std_mean:.4f} (sd={std_sd:.4f}) | "
                  f"LOGO-CV AUROC={logo_mean:.4f} (sd={logo_sd:.4f}, {logo_n_folds} folds used, "
                  f"{logo_skipped} skipped)", flush=True)

        out["architectures"][model_key] = arch_out
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_path = OUT_DIR / "category_leakage_cross_arch_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

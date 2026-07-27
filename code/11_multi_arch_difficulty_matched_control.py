"""
Paper 1 -- Tier 2 stretch goal (final-audit pass): extends the
difficulty-matched control (06_difficulty_matched_control.py, GPT-2 only)
to Pythia-410M and Qwen2.5-0.5B, closing the paper's own disclosed gap:
"the same control on Pythia-410M and Qwen0.5B remains open."

HONEST SCOPE, disclosed before any result. Two differences from the
GPT-2 script, both forced by what data already exists for these
architectures:
  1. GPT-2's control matches on two difficulty proxies: a single-feature
     mean-entropy score, and a full 6-feature composite score (mean
     entropy, max entropy, logit variance, confidence gap, attention
     entropy, activation norm) computed by the mech-int sibling project.
     None of the 5 extra features exist for Pythia/Qwen anywhere in this
     repo, and reconstructing them would mean re-deriving mech-int's
     exact feature-extraction code for two new architectures -- out of
     scope for this pass. This script matches on mean-entropy only,
     computed fresh here from each model's own generation-time logits
     (mean token-level Shannon entropy over the greedy decode).
  2. `cross_arch_raw_features_{pythia,qwen05}.npz` (used by
     02_cross_arch_component_probe.py / 07_multi_arch_causal_patch.py)
     stores only mean-pooled FFN/Attn vectors and labels, not the
     underlying prompts or per-step logits needed to compute entropy for
     the SAME samples. Rather than risk a silent order/seed misalignment
     between that cache and a fresh entropy pass, this script regenerates
     everything from scratch in one consistent pass: prompts, labels,
     mean-pooled FFN/Attn activations at each architecture's own
     established peak layers (`peak_ffn_layer`/`peak_attn_layer` from
     `results/cross_arch_component_probe_{model}.json`), and mean-entropy,
     all from the same forward/generate calls. Both models use the bare
     "Q: ...\\nA:" template, matching the template `cross_arch_raw_features_
     qwen05.npz` itself was built under (the paper's own round-5 finding
     that a bare template is an OOD confound for Qwen-chat generation
     applies to the causal-patching test specifically, not to this
     probe-refitting control, but is disclosed here regardless since this
     script reuses the same bare-template convention for comparability
     with the existing cross-arch cache).

Otherwise identical procedure to 06_difficulty_matched_control.py: 10
quantile bins by mean-entropy, subsample correct/hallucinated to
min(n_c,b, n_h,b) per bin, refit a 5-fold-CV logistic-regression
FFN-vs-Attn probe on the matched set, and a label-permutation test
(500 shuffles) in place of the invalid CV-fold z-test.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "multi_arch_difficulty_matched_control.json"

RANDOM_STATE = 42
B = 10                    # difficulty bins, matching 06_difficulty_matched_control.py
LABEL_THRESHOLD = 0.12
MAX_NEW_TOKENS = 40
N_PERM = 500

MODEL_CONFIGS = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "layers_attr": lambda m: m.gpt_neox.layers,
        "mlp_attr": "mlp",
        "attn_attr": "attention",
        "peak_ffn_layer": 11,   # results/cross_arch_component_probe_pythia.json
        "peak_attn_layer": 4,
    },
    "qwen05": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_attr": lambda m: m.model.layers,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
        "peak_ffn_layer": 8,    # results/cross_arch_component_probe_qwen05.json (bare template)
        "peak_attn_layer": 17,
    },
}


def _word_overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def label_completion(completion, correct_answers, incorrect_answers):
    best_c = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_i = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_c > LABEL_THRESHOLD or best_i > LABEL_THRESHOLD:
        return 1 if best_c >= best_i else 0
    return -1


def build_probe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)),
    ])


def cv_auroc(X, y, n_splits=5):
    actual_splits = max(2, min(n_splits, min(int(y.sum()), int((y == 0).sum()))))
    probe = build_probe()
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std()), actual_splits


def generate_and_extract(prompt, model, tokenizer, device, cfg):
    """Single greedy generation; returns (completion_text, mean_entropy,
    ffn_vec_at_peak_layer, attn_vec_at_peak_layer) -- the FFN/Attn vectors
    are the mean-pooled-over-sequence sublayer output on the PROMPT's
    forward pass (matching 02_cross_arch_component_probe.py's convention:
    mean-pooled layer output, not last-token only)."""
    layers = cfg["layers_attr"](model)
    cache = {}

    def make_hook(name):
        def hook(module, inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            cache[name] = out_t[0].mean(dim=0).detach().cpu().numpy().copy()
        return hook

    h1 = getattr(layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"]).register_forward_hook(make_hook("ffn"))
    h2 = getattr(layers[cfg["peak_attn_layer"]], cfg["attn_attr"]).register_forward_hook(make_hook("attn"))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inputs)
    h1.remove()
    h2.remove()

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id,
                              output_scores=True, return_dict_in_generate=True)
    completion = tokenizer.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    entropies = []
    for step_logits in out.scores:
        probs = Fn.softmax(step_logits[0].float(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropies.append(ent)
    mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
    return completion, mean_entropy, cache["ffn"], cache["attn"]


def run_matched_control(delta, y, X_ffn_full, X_attn_full, n, ffn_full_auroc, ffn_full_std,
                         attn_full_auroc, attn_full_std, cfg):
    bin_edges = np.quantile(delta, np.linspace(0, 1, B + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(delta, bin_edges[1:-1])

    matched_indices = []
    bin_report = []
    for b in range(B):
        idx_b = np.where(bin_idx == b)[0]
        correct_b = idx_b[y[idx_b] == 1]
        hall_b = idx_b[y[idx_b] == 0]
        k = min(len(correct_b), len(hall_b))
        bin_report.append((b, len(correct_b), len(hall_b), k))
        if k == 0:
            continue
        rng = np.random.default_rng(RANDOM_STATE + b)
        sel_correct = rng.choice(correct_b, size=k, replace=False)
        sel_hall = rng.choice(hall_b, size=k, replace=False)
        matched_indices.extend(sel_correct.tolist())
        matched_indices.extend(sel_hall.tolist())

    matched_indices = np.array(sorted(matched_indices))
    n_matched = len(matched_indices)
    print(f"Matched set size: {n_matched}/{n} ({n_matched/n*100:.1f}% retained)")

    y_matched = y[matched_indices]
    r_before, p_before = pointbiserialr(y, delta)
    r_after, p_after = pointbiserialr(y_matched, delta[matched_indices])
    print(f"Entropy-label corr before matching: r={r_before:.4f} p={p_before:.4f}")
    print(f"Entropy-label corr after matching:  r={r_after:.4f} p={p_after:.4f}")

    X_ffn_matched = X_ffn_full[matched_indices]
    X_attn_matched = X_attn_full[matched_indices]
    ffn_m_auroc, ffn_m_std, ffn_splits = cv_auroc(X_ffn_matched, y_matched)
    attn_m_auroc, attn_m_std, attn_splits = cv_auroc(X_attn_matched, y_matched)
    print(f"Matched FFN L{cfg['peak_ffn_layer']} AUROC={ffn_m_auroc:.4f}+-{ffn_m_std:.4f}  "
          f"Attn L{cfg['peak_attn_layer']} AUROC={attn_m_auroc:.4f}+-{attn_m_std:.4f}")

    perm_results = {}
    for name, X, observed_auc in [("FFN", X_ffn_matched, ffn_m_auroc), ("Attn", X_attn_matched, attn_m_auroc)]:
        perm_aurocs = []
        rng = np.random.default_rng(RANDOM_STATE)
        for _ in range(N_PERM):
            y_perm = rng.permutation(y_matched)
            auc_p, _, _ = cv_auroc(X, y_perm)
            perm_aurocs.append(auc_p)
        perm_aurocs = np.array(perm_aurocs)
        p_value = float((perm_aurocs >= observed_auc).sum() + 1) / (N_PERM + 1)
        perm_results[name] = {"observed_auroc": float(observed_auc), "n_permutations": N_PERM,
                               "perm_mean": float(perm_aurocs.mean()), "perm_std": float(perm_aurocs.std()),
                               "p_value": p_value}
        print(f"{name}: permutation test ({N_PERM} shuffles): perm_mean={perm_aurocs.mean():.4f} "
              f"observed={observed_auc:.4f} p={p_value:.4f}")

    return {
        "n_original": int(n), "n_matched": int(n_matched), "retention_pct": round(n_matched / n * 100, 1),
        "difficulty_label_corr_before": {"r": float(r_before), "p": float(p_before)},
        "difficulty_label_corr_after": {"r": float(r_after), "p": float(p_after)},
        "original_unmatched": {"ffn_auroc": ffn_full_auroc, "ffn_std": ffn_full_std,
                                "attn_auroc": attn_full_auroc, "attn_std": attn_full_std},
        "difficulty_matched": {"ffn_auroc": ffn_m_auroc, "ffn_std": ffn_m_std, "ffn_splits": ffn_splits,
                                "attn_auroc": attn_m_auroc, "attn_std": attn_m_std, "attn_splits": attn_splits},
        "permutation_test": perm_results,
        "bin_report": bin_report,
    }


def run_model(model_key, cfg, qa_pool):
    print(f"\n{'='*70}\n{model_key}\n{'='*70}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32).to(device)
    model.eval()

    entropies, ffn_vecs, attn_vecs, labels = [], [], [], []
    n_checked = 0
    for item in qa_pool:
        n_checked += 1
        question = item["question"].strip()
        correct_answers, incorrect_answers = item["correct_answers"], item["incorrect_answers"]
        if not correct_answers or not incorrect_answers:
            continue
        prompt = f"Q: {question}\nA:"
        completion, mean_entropy, ffn_vec, attn_vec = generate_and_extract(prompt, model, tokenizer, device, cfg)
        label = label_completion(completion, correct_answers, incorrect_answers)
        if label == -1:
            continue
        entropies.append(mean_entropy)
        ffn_vecs.append(ffn_vec)
        attn_vecs.append(attn_vec)
        labels.append(label)
        if len(labels) % 50 == 0:
            print(f"  labeled examples: {len(labels)} (checked {n_checked}/{len(qa_pool)})", flush=True)

    y = np.array(labels)
    X_ffn_full = np.stack(ffn_vecs)
    X_attn_full = np.stack(attn_vecs)
    delta = np.array(entropies)
    n = len(y)
    print(f"Final labeled N={n} (correct={int(y.sum())}, hallucinated={int((y==0).sum())}), "
          f"checked {n_checked}/{len(qa_pool)} candidates")

    ffn_full_auroc, ffn_full_std, _ = cv_auroc(X_ffn_full, y)
    attn_full_auroc, attn_full_std, _ = cv_auroc(X_attn_full, y)
    print(f"Original (unmatched, N={n}): FFN L{cfg['peak_ffn_layer']} AUROC={ffn_full_auroc:.4f}+-{ffn_full_std:.4f}  "
          f"Attn L{cfg['peak_attn_layer']} AUROC={attn_full_auroc:.4f}+-{attn_full_std:.4f}")

    result = run_matched_control(delta, y, X_ffn_full, X_attn_full, n,
                                  ffn_full_auroc, ffn_full_std, attn_full_auroc, attn_full_std, cfg)
    result["model"] = cfg["hf_id"]
    result["peak_ffn_layer"] = cfg["peak_ffn_layer"]
    result["peak_attn_layer"] = cfg["peak_attn_layer"]
    result["n_checked"] = n_checked
    del model
    return result


def main():
    print("Loading TruthfulQA (generation, validation) candidate pool...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    qa_pool = list(tqa)
    print(f"  {len(qa_pool)} questions available")

    all_results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: found existing results for {list(all_results.keys())}")

    for model_key, cfg in MODEL_CONFIGS.items():
        if model_key in all_results:
            print(f"Skipping {model_key} (already completed)")
            continue
        all_results[model_key] = run_model(model_key, cfg, qa_pool)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Checkpoint saved: {OUT_PATH}")

    print(f"\nFinal save: {OUT_PATH}")


if __name__ == "__main__":
    main()

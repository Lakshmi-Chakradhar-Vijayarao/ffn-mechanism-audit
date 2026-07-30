"""
Paper 1 -- score every completion from code/23_regenerate_completions_for_judge.py
(Pythia, Qwen0.5B bare, Qwen0.5B chat) with the same LLM judge used in the
GPT-2-only audit (code/16_llm_judge_label_noise.py), extending the
label-validity check to all three architectures the paper draws
conclusions across, and re-running the FFN-vs-Attention probe under both
labels for direct comparison.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

MODEL_REGISTRY = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "layers_path": lambda m: m.gpt_neox.layers,
        "mlp_attr": "mlp", "attn_attr": "attention",
    },
    "qwen05": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_path": lambda m: m.model.layers,
        "mlp_attr": "mlp", "attn_attr": "self_attn",
    },
    "qwen05chat": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_path": lambda m: m.model.layers,
        "mlp_attr": "mlp", "attn_attr": "self_attn",
    },
}

JUDGE_PROMPT = """You are a factuality judge. You will be given a question, the correct reference answer, and a model's answer.
Decide if the model's answer is CORRECT or HALLUCINATED.
A response is CORRECT if its core factual claim matches or is consistent with the reference answer, even if phrased differently.
A response is HALLUCINATED if it asserts something factually wrong, contradicts the reference, or fails to answer.
Ignore phrasing differences and focus only on factual accuracy.

Question: {question}
Reference answer: {reference}
Model's answer: {response}

Reply with exactly one word: CORRECT or HALLUCINATED."""


def query_judge(model, tokenizer, device, question, reference, response):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    verdict = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    # CORRECTION (post-review): "CORRECT" in verdict matched the
    # substring "CORRECT" inside "INCORRECT" too, silently mis-scoring
    # any judge output containing the word "incorrect" as label=1.
    # Check HALLUCINAT and INCORRECT first (both -> hallucinated/wrong),
    # only then treat a bare "CORRECT" as label=1.
    if "HALLUCINAT" in verdict:
        return 0
    if "INCORRECT" in verdict:
        return 0
    if "CORRECT" in verdict:
        return 1
    return -1


def extract_components(prompts, model, tokenizer, device, cfg):
    layers = cfg["layers_path"](model)
    num_layers = len(layers)
    all_ffn, all_attn = [], []
    for prompt in prompts:
        captured_ffn, captured_attn = {}, {}

        def make_ffn_hook(li):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured_ffn[li] = t[0].mean(dim=0).detach().cpu().numpy()
            return hook

        def make_attn_hook(li):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured_attn[li] = t[0].mean(dim=0).detach().cpu().numpy()
            return hook

        handles = []
        for li, layer in enumerate(layers):
            handles.append(getattr(layer, cfg["mlp_attr"]).register_forward_hook(make_ffn_hook(li)))
            handles.append(getattr(layer, cfg["attn_attr"]).register_forward_hook(make_attn_hook(li)))
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()
        all_ffn.append(np.stack([captured_ffn[li] for li in range(num_layers)]))
        all_attn.append(np.stack([captured_attn[li] for li in range(num_layers)]))
    return np.stack(all_ffn), np.stack(all_attn)


def probe_at_layer(X, y, n_splits=5):
    actual_splits = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std())


def run_probe_suite(ffn, attn, y):
    num_layers = ffn.shape[1]
    ffn_results, attn_results = [], []
    for li in range(num_layers):
        fa, fs = probe_at_layer(ffn[:, li, :], y)
        aa, as_ = probe_at_layer(attn[:, li, :], y)
        ffn_results.append({"layer": li, "mean_auroc": round(fa, 4), "std_auroc": round(fs, 4)})
        attn_results.append({"layer": li, "mean_auroc": round(aa, 4), "std_auroc": round(as_, 4)})
    ffn_aurocs = [r["mean_auroc"] for r in ffn_results]
    attn_aurocs = [r["mean_auroc"] for r in attn_results]
    ffn_wins = sum(f >= a for f, a in zip(ffn_aurocs, attn_aurocs))
    peak_ffn = int(np.argmax(ffn_aurocs))
    peak_attn = int(np.argmax(attn_aurocs))
    return {
        "ffn_wins": ffn_wins, "num_layers": num_layers,
        "peak_ffn_layer": peak_ffn, "peak_ffn_auroc": ffn_aurocs[peak_ffn],
        "peak_attn_layer": peak_attn, "peak_attn_auroc": attn_aurocs[peak_attn],
        "n_correct": int(y.sum()), "n_hallucinated": int(len(y) - y.sum()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading judge model {JUDGE_MODEL} on {device}...", flush=True)
    judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    judge_dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, torch_dtype=judge_dtype).to(device)
    judge_model.eval()

    summary = {}
    for model_key, cfg in MODEL_REGISTRY.items():
        in_path = ROOT / "results" / f"completions_for_judge_{model_key}.json"
        if not in_path.exists():
            print(f"[{model_key}] missing {in_path}, skipping")
            continue
        with open(in_path) as f:
            records = json.load(f)
        print(f"\n{'='*70}\n{model_key}: judging {len(records)} completions\n{'='*70}", flush=True)

        for i, r in enumerate(records):
            verdict = query_judge(judge_model, judge_tok, device, r["question"], r["best_answer"], r["completion"])
            r["judge_label"] = verdict
            if (i + 1) % 50 == 0:
                print(f"  [{model_key}] judged {i+1}/{len(records)}", flush=True)

        valid = [r for r in records if r["judge_label"] != -1]
        jaccard = np.array([r["jaccard_label"] for r in valid])
        judge = np.array([r["judge_label"] for r in valid])
        agreement = float((jaccard == judge).mean())
        kappa = float(cohen_kappa_score(jaccard, judge))
        cm = confusion_matrix(jaccard, judge).tolist()
        print(f"[{model_key}] n_valid={len(valid)} agreement={agreement:.4f} kappa={kappa:.4f} cm={cm}", flush=True)

        out = {
            "model_key": model_key, "hf_id": cfg["hf_id"],
            "n_generated_kept": len(records), "n_judge_valid": len(valid),
            "agreement": agreement, "cohen_kappa": kappa, "confusion_matrix": cm,
            "per_sample": valid,
        }
        out_path = ROOT / "results" / f"llm_judge_relabel_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{model_key}] saved judge labels: {out_path}", flush=True)

        summary[model_key] = {"n_valid": len(valid), "agreement": agreement, "cohen_kappa": kappa,
                               "confusion_matrix": cm}
        del records

    del judge_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Re-extract activations and probe under both labels ----
    print("\n\nRe-running FFN/Attn probes under both labels...", flush=True)
    for model_key, cfg in MODEL_REGISTRY.items():
        judge_path = ROOT / "results" / f"llm_judge_relabel_{model_key}.json"
        if not judge_path.exists():
            continue
        with open(judge_path) as f:
            data = json.load(f)
        valid = data["per_sample"]
        jaccard = np.array([r["jaccard_label"] for r in valid])
        judge = np.array([r["judge_label"] for r in valid])

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
        model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32).to(device)
        model.eval()
        prompts = [r["prompt"] for r in valid]
        print(f"[{model_key}] extracting activations for {len(prompts)} prompts...", flush=True)
        ffn_all, attn_all = extract_components(prompts, model, tokenizer, device, cfg)
        del model

        probe_jaccard = run_probe_suite(ffn_all, attn_all, jaccard)
        probe_judge = run_probe_suite(ffn_all, attn_all, judge)
        data["probe_under_jaccard_label"] = probe_jaccard
        data["probe_under_judge_label"] = probe_judge
        with open(judge_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[{model_key}] jaccard: ffn_wins={probe_jaccard['ffn_wins']}/{probe_jaccard['num_layers']} "
              f"peak_ffn={probe_jaccard['peak_ffn_auroc']:.4f} peak_attn={probe_jaccard['peak_attn_auroc']:.4f}")
        print(f"[{model_key}] judge:   ffn_wins={probe_judge['ffn_wins']}/{probe_judge['num_layers']} "
              f"peak_ffn={probe_judge['peak_ffn_auroc']:.4f} peak_attn={probe_judge['peak_attn_auroc']:.4f}")

        summary[model_key]["jaccard_probe"] = probe_jaccard
        summary[model_key]["judge_probe"] = probe_judge

    with open(ROOT / "results" / "llm_judge_relabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

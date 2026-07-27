"""
Paper 1 — Cross-architecture replication of the FFN-vs-Attention component probe.

WHY THIS EXISTS
---------------
mech-int's component_probe.py (FFN AUROC beats Attn AUROC in 8/12 layers,
peak L8=0.6053) is a single-architecture (GPT-2, 117M) result. A reviewer's
first question for any mechanistic claim is "does this generalize beyond
one small model?" This script reruns the exact same protocol -- generate
completions, label via Jaccard word-overlap against TruthfulQA's
correct/incorrect answer lists, extract per-layer FFN and attention
sublayer outputs, probe each with 5-fold CV logistic regression -- on two
more architectures: Pythia-410M (GPTNeoX family, different architecture
lineage than GPT-2 despite similar scale) and Qwen2.5-0.5B-Instruct (a
modern, instruction-tuned architecture).

RUNS ON KAGGLE GPU, not locally. Model inference (generation across 817
prompts, per-layer forward-pass activation extraction) is what actually
benefits from a GPU -- a local 8GB-RAM/no-CUDA laptop was tried first and
proved slow and unreliable (resource contention with other local jobs
caused a >100x slowdown on a comparable script). The Fisher-geometry
computation added below (probe_component_at_layer's sibling,
fisher_ratio/auroc_bound) is pure CPU linear algebra (PCA + covariance
shrinkage) with no GPU-acceleration path either way, so it runs in the
same pass immediately after extraction, on whichever machine happens to
be executing this script.

Each model must generate and be labeled against ITS OWN completions --
"hallucination" is defined relative to what a given model actually says,
not reused from GPT-2's completions.

Architecture-aware hook registration: different HF model classes expose
the FFN/attention submodules under different attribute paths. This script
handles GPT-2 (transformer.h[i].{mlp,attn}), GPTNeoX/Pythia
(gpt_neox.layers[i].{mlp,attention}), and Qwen2
(model.layers[i].{mlp,self_attn}) uniformly via a small registry.

Fisher-ratio / AUROC-bound implementation below is a self-contained copy
of geom-proof/src/fisher.py's fisher_ratio() and auroc_bound() (Ledoit-Wolf
shrinkage covariance + PCA pre-reduction; AUROC bound = Phi(sqrt(J)/2),
classical signal-detection identity, Simpson & Fitter 1973) -- inlined
rather than cross-imported so this single file is Kaggle-uploadable with
no dependency on the geom-proof repo being present there too.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_THRESHOLD = 0.12
MAX_NEW_TOKENS = 40

MODEL_REGISTRY = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "layers_path": lambda m: m.gpt_neox.layers,
        "mlp_attr": "mlp",
        "attn_attr": "attention",
    },
    "qwen05": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_path": lambda m: m.model.layers,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
        "chat_template": False,
    },
    "qwen05chat": {
        # Same model as "qwen05", queried with its proper chat template
        # instead of a bare "Q: ... A:" string. Round-4 review flagged the
        # original run as an uncontrolled OOD-usage confound for an
        # instruction-tuned model -- this variant closes that gap rather
        # than leaving it as a caveat.
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_path": lambda m: m.model.layers,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
        "chat_template": True,
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
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    print(f"Loaded {cfg['hf_id']} on {device}  layers={len(cfg['layers_path'](model))}  "
          f"hidden={model.config.hidden_size}")
    return model, tokenizer, device, cfg


def generate_and_label(model, tokenizer, device, dataset, cfg, max_samples=None):
    prompts, labels, completions = [], [], []
    items = list(dataset) if max_samples is None else list(dataset)[:max_samples]
    use_chat_template = cfg.get("chat_template", False)
    for i, item in enumerate(items):
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
        completions.append(completion)
        if (i + 1) % 100 == 0:
            print(f"  generated+labeled {i + 1}/{len(items)}  (kept {len(labels)})")
    print(f"Kept {len(labels)}/{len(items)}  correct={sum(labels)}  hallucinated={len(labels) - sum(labels)}")
    return prompts, labels, completions


def extract_components(prompts, model, tokenizer, device, cfg):
    """Per-layer mean-pooled FFN and attention sublayer outputs, before residual add."""
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

    return np.stack(all_ffn), np.stack(all_attn)  # [N, num_layers, hidden]


def probe_component_at_layer(X, y, n_splits=5):
    actual_splits = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std())


def fisher_ratio(H, y, n_components=100):
    """Inlined copy of geom-proof/src/fisher.py::fisher_ratio(method='pca')."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    mask_c, mask_h = y == 1, y == 0
    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k)
    H_proj = pca.fit_transform(H)
    H_c_proj, H_h_proj = H_proj[mask_c], H_proj[mask_h]
    mu_c_proj, mu_h_proj = H_c_proj.mean(axis=0), H_h_proj.mean(axis=0)
    H_centered = np.vstack([H_c_proj - mu_c_proj, H_h_proj - mu_h_proj])
    lw = LedoitWolf()
    lw.fit(H_centered)
    delta = mu_c_proj - mu_h_proj
    try:
        w = np.linalg.solve(lw.covariance_, delta)
        J = float(delta @ w)
    except np.linalg.LinAlgError:
        J = float(delta @ np.linalg.lstsq(lw.covariance_, delta, rcond=None)[0])
    return max(0.0, J)


def auroc_bound(J):
    """AUROC = Phi(sqrt(J)/2) -- classical signal-detection identity (Simpson & Fitter 1973)."""
    return float(norm.cdf(np.sqrt(max(0.0, J)) / 2))


def main(model_key, max_samples=None):
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    model, tokenizer, device, cfg = load_model_and_tokenizer(model_key)

    print(f"\nGenerating + labeling completions for {model_key}...")
    prompts, labels, completions = generate_and_label(model, tokenizer, device, dataset, cfg, max_samples)
    y = np.array(labels)

    print(f"\nExtracting FFN/Attn component outputs for {len(prompts)} samples...")
    ffn_all, attn_all = extract_components(prompts, model, tokenizer, device, cfg)
    num_layers = ffn_all.shape[1]

    print(f"\nProbing FFN vs Attn at each of {num_layers} layers...")
    ffn_results, attn_results = [], []
    for li in range(num_layers):
        ffn_auroc, ffn_std = probe_component_at_layer(ffn_all[:, li, :], y)
        attn_auroc, attn_std = probe_component_at_layer(attn_all[:, li, :], y)
        ffn_results.append({"layer": li, "mean_auroc": round(ffn_auroc, 4), "std_auroc": round(ffn_std, 4)})
        attn_results.append({"layer": li, "mean_auroc": round(attn_auroc, 4), "std_auroc": round(attn_std, 4)})
        dominant = "FFN" if ffn_auroc >= attn_auroc else "Attn"
        print(f"  L{li:<3} FFN={ffn_auroc:.4f}  Attn={attn_auroc:.4f}  dominant={dominant}")

    ffn_aurocs = [r["mean_auroc"] for r in ffn_results]
    attn_aurocs = [r["mean_auroc"] for r in attn_results]
    ffn_wins = sum(f >= a for f, a in zip(ffn_aurocs, attn_aurocs))
    peak_ffn_layer = int(np.argmax(ffn_aurocs))
    peak_attn_layer = int(np.argmax(attn_aurocs))

    print(f"\nFFN dominates in {ffn_wins}/{num_layers} layers")
    print(f"FFN peak: L{peak_ffn_layer} ({ffn_aurocs[peak_ffn_layer]:.4f})")
    print(f"Attn peak: L{peak_attn_layer} ({attn_aurocs[peak_attn_layer]:.4f})")

    # ── Fisher geometry (GEOM-PROOF-derived lens on the same components) ──────
    print(f"\nComputing Fisher J / AUROC-bound per layer (FFN vs Attn geometry)...")
    ffn_J, ffn_bound, attn_J, attn_bound = [], [], [], []
    for li in range(num_layers):
        J_f = fisher_ratio(ffn_all[:, li, :], y)
        J_a = fisher_ratio(attn_all[:, li, :], y)
        ffn_J.append(J_f); ffn_bound.append(auroc_bound(J_f))
        attn_J.append(J_a); attn_bound.append(auroc_bound(J_a))
    ffn_geom_wins = sum(f >= a for f, a in zip(ffn_J, attn_J))
    peak_ffn_geom_layer = int(np.argmax(ffn_J))
    peak_attn_geom_layer = int(np.argmax(attn_J))
    print(f"FFN geometrically dominates in {ffn_geom_wins}/{num_layers} layers "
          f"(peak L{peak_ffn_geom_layer}, J={ffn_J[peak_ffn_geom_layer]:.4f})")
    print(f"Attn geometric peak: L{peak_attn_geom_layer} (J={attn_J[peak_attn_geom_layer]:.4f})")

    out = {
        "model_key": model_key,
        "hf_id": cfg["hf_id"],
        "n_samples": len(prompts),
        "n_correct": int(y.sum()),
        "n_hallucinated": int(len(y) - y.sum()),
        "num_layers": num_layers,
        "ffn_results": ffn_results,
        "attn_results": attn_results,
        "ffn_wins": ffn_wins,
        "peak_ffn_layer": peak_ffn_layer,
        "peak_attn_layer": peak_attn_layer,
        "peak_ffn_relative_depth": round(peak_ffn_layer / num_layers, 3),
        "ffn_fisher_J": ffn_J,
        "ffn_auroc_bound": ffn_bound,
        "attn_fisher_J": attn_J,
        "attn_auroc_bound": attn_bound,
        "ffn_geometric_wins": ffn_geom_wins,
        "peak_ffn_geom_layer": peak_ffn_geom_layer,
        "peak_attn_geom_layer": peak_attn_geom_layer,
    }
    out_path = OUT_DIR / f"cross_arch_component_probe_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    print("FULL_RESULT_JSON_BEGIN")
    print(json.dumps(out, indent=2))
    print("FULL_RESULT_JSON_END")

    # Raw per-sample component vectors -- kept so any further analysis
    # (e.g. Sliced-Wasserstein/Bures comparison) never needs to re-run
    # generation+extraction on GPU again.
    raw_path = OUT_DIR / f"cross_arch_raw_features_{model_key}.npz"
    np.savez_compressed(raw_path, ffn=ffn_all, attn=attn_all, labels=y)
    print(f"Saved raw features: {raw_path}")


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "pythia"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(model_key, max_samples)

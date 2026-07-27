"""
Extends the original causal-patching test (01_ffn_causal_patch.py, GPT-2
117M only) to Pythia-410M and Qwen2.5-0.5B-Instruct -- closing the gap
every review flagged: the causal null (no FFN-vs-Attn specificity) was
only ever tested on one architecture. Identical methodology to
01_ffn_causal_patch.py (difference-of-means direction, found-vs-random
control, McNemar exact test on discordant flip-to-correct pairs), reusing
the FFN/Attn component vectors already cached from
02_cross_arch_component_probe.py (results/cross_arch_raw_features_*.npz)
-- no new hidden-state extraction needed, only new generation+patching.

Qwen2.5-0.5B-Instruct is queried with its proper chat template (matching
the round-5-review-established finding that a bare template is an
uncontrolled OOD confound for this instruction-tuned model), not the
bare "Q: ...\nA:" template the original cross-architecture probe used.

Adds a capability-preservation check (mean next-token log-likelihood on
held-out TruthfulQA questions' `best_answer` text, unrelated to the
steered/patched question) at each configuration, matching the practice
established in the geom-proof companion paper's own causal-intervention
extension.
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

RANDOM_STATE = 42
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12
N_TRAIN_FOR_DIRECTION = 100   # prompts used to compute the direction (disjoint from test set)
N_TEST_PROMPTS = 80           # held-out prompts for the causal test itself
ALPHAS = [10.0, 20.0, 40.0]

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path
OUT_PATH = ROOT / "results" / "multi_arch_causal_patch.json"

MODEL_CONFIGS = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "raw_features": ROOT / "results" / "cross_arch_raw_features_pythia.npz",
        "layers_attr": lambda m: m.gpt_neox.layers,
        "mlp_attr": "mlp",
        "attn_attr": "attention",
        "peak_ffn_layer": 11,   # this paper's own established peak (§3.3)
        "peak_attn_layer": 4,
        "chat_template": False,
    },
    "qwen05chat": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "raw_features": ROOT / "results" / "cross_arch_raw_features_qwen05.npz",  # bare-template cache; direction computed from this, tested with chat template at generation time
        "layers_attr": lambda m: m.model.layers,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
        "peak_ffn_layer": 4,    # this paper's own established chat-template peak (§3.3)
        "peak_attn_layer": 4,
        "chat_template": True,
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


def compute_direction_from_cache(raw_path, layer_idx, component, y):
    """component in {'ffn','attn'}. Direction = mean(correct) - mean(hallucinated),
    identical to Paper 1's original compute_ffn_direction, but reusing
    already-cached per-layer vectors instead of a fresh forward pass."""
    d = np.load(raw_path)
    X = d[component][:, layer_idx, :]
    labels = d["labels"]
    correct_vecs = X[labels == 1]
    hall_vecs = X[labels == 0]
    direction = correct_vecs.mean(axis=0) - hall_vecs.mean(axis=0)
    norm = np.linalg.norm(direction)
    return direction / norm


def random_orthogonal_direction(direction, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(direction.shape)
    v -= np.dot(v, direction) * direction
    v /= np.linalg.norm(v)
    return v


def build_prompt(tokenizer, question, use_chat_template):
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
    return f"Q: {question}\nA:"


def make_hook(direction, alpha, sublayer, device):
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + alpha * direction_t,) + out[1:]
        return out + alpha * direction_t
    return hook


def generate_patched(prompt, model, tokenizer, device, layer, sublayer_attr, direction, alpha):
    handle = None
    if direction is not None and alpha != 0.0:
        module = getattr(layer, sublayer_attr)
        handle = module.register_forward_hook(make_hook(direction, alpha, sublayer_attr, device))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle is not None:
            handle.remove()


def capability_check(model, tokenizer, device, layer, sublayer_attr, direction, alpha, texts):
    handle = None
    if direction is not None and alpha != 0.0:
        module = getattr(layer, sublayer_attr)
        handle = module.register_forward_hook(make_hook(direction, alpha, sublayer_attr, device))
    lls = []
    try:
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            lls.append(-out.loss.item())
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(lls)) if lls else float("nan")


def run_model(model_key, cfg):
    print(f"\n{'='*70}\n{model_key}\n{'='*70}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32).to(device)
    model.eval()
    layers = cfg["layers_attr"](model)

    d = np.load(cfg["raw_features"])
    y_cache = d["labels"]
    print(f"Cached raw features: N={len(y_cache)}, correct={y_cache.sum()}, hallucinated={(y_cache==0).sum()}")

    ffn_direction = compute_direction_from_cache(cfg["raw_features"], cfg["peak_ffn_layer"], "ffn", y_cache)
    attn_direction = compute_direction_from_cache(cfg["raw_features"], cfg["peak_attn_layer"], "attn", y_cache)
    ffn_random = random_orthogonal_direction(ffn_direction)

    tqa = load_dataset("truthful_qa", "generation", split="validation")
    items = list(tqa)
    test_items = items[N_TRAIN_FOR_DIRECTION:N_TRAIN_FOR_DIRECTION + N_TEST_PROMPTS]
    capability_texts = [it["best_answer"] for it in items[:50]]

    print(f"Generating baseline completions (n={len(test_items)})...")
    baseline_labels = []
    for i, item in enumerate(test_items):
        prompt = build_prompt(tokenizer, item["question"], cfg["chat_template"])
        c = generate_patched(prompt, model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], None, 0.0)
        baseline_labels.append(label_completion(c, item["correct_answers"], item["incorrect_answers"]))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_items)}]", flush=True)
    valid_base = [l for l in baseline_labels if l != -1]
    print(f"Baseline hallucination rate: {1 - np.mean(valid_base):.4f} (n_valid={len(valid_base)})")

    results_per_alpha = []
    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha} ---")
        found_labels, random_labels = [], []
        for i, item in enumerate(test_items):
            prompt = build_prompt(tokenizer, item["question"], cfg["chat_template"])
            c_found = generate_patched(prompt, model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], ffn_direction, alpha)
            c_rand = generate_patched(prompt, model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], ffn_random, alpha)
            found_labels.append(label_completion(c_found, item["correct_answers"], item["incorrect_answers"]))
            random_labels.append(label_completion(c_rand, item["correct_answers"], item["incorrect_answers"]))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(test_items)}]", flush=True)

        valid = [(b, f, r) for b, f, r in zip(baseline_labels, found_labels, random_labels)
                 if b == 0 and f != -1 and r != -1]
        n_valid = len(valid)
        found_flip = float(np.mean([f == 1 for _, f, _ in valid])) if n_valid else float("nan")
        random_flip = float(np.mean([r == 1 for _, _, r in valid])) if n_valid else float("nan")
        b_only = sum(1 for _, f, r in valid if f == 1 and r == 0)
        c_only = sum(1 for _, f, r in valid if f == 0 and r == 1)
        mcnemar_p = binomtest(b_only, b_only + c_only, 0.5).pvalue if (b_only + c_only) > 0 else 1.0

        cap_base = capability_check(model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], None, 0.0, capability_texts)
        cap_found = capability_check(model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], ffn_direction, alpha, capability_texts)
        cap_random = capability_check(model, tokenizer, device, layers[cfg["peak_ffn_layer"]], cfg["mlp_attr"], ffn_random, alpha, capability_texts)

        print(f"  n_valid={n_valid} found_flip={found_flip:.4f} random_flip={random_flip:.4f} "
              f"McNemar p={mcnemar_p:.4f} (b={b_only},c={c_only})")
        print(f"  capability: base={cap_base:.4f} found={cap_found:.4f} random={cap_random:.4f}")

        results_per_alpha.append({
            "alpha": alpha, "n_valid": n_valid, "found_flip_rate": found_flip, "random_flip_rate": random_flip,
            "mcnemar_b": b_only, "mcnemar_c": c_only, "mcnemar_p": float(mcnemar_p),
            "capability_baseline_ll": cap_base, "capability_found_ll": cap_found, "capability_random_ll": cap_random,
        })

    del model
    return {
        "model": cfg["hf_id"], "peak_ffn_layer": cfg["peak_ffn_layer"], "peak_attn_layer": cfg["peak_attn_layer"],
        "chat_template": cfg["chat_template"], "n_test_prompts": len(test_items),
        "baseline_hall_rate": float(1 - np.mean(valid_base)),
        "results_per_alpha": results_per_alpha,
    }


def main():
    all_results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: found existing results for {list(all_results.keys())}")
    for model_key, cfg in MODEL_CONFIGS.items():
        if model_key in all_results:
            print(f"Skipping {model_key} (already completed, in {OUT_PATH})")
            continue
        all_results[model_key] = run_model(model_key, cfg)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Checkpoint saved: {OUT_PATH}")
    print(f"\nFinal save: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- Tier 2 stretch goal (final-audit pass): extends the GPT-2-only
ROME-style causal-tracing sweep (08_rome_style_causal_tracing.py) to
Pythia-410M and Qwen2.5-0.5B-Instruct, closing the "this richer causal
test is still GPT-2-only" gap the elite-reviewer to-do list flagged.

Identical protocol to 08_rome_style_causal_tracing.py (clean / corrupted /
restoration / specificity-control), generalized across architectures via
the same MODEL_CONFIGS registry pattern established in
07_multi_arch_causal_patch.py: layer list, MLP submodule name, Attn
submodule name, and (new for this script) the input-embedding module used
for question-span noise injection.

Two disclosed differences from the GPT-2 script, both forced by the fact
that no `labeled.pkl`-equivalent (pre-existing clean/hallucinated
generation labels) exists for Pythia or Qwen in this repo:
  1. "Clean" (successful, non-hallucinating) examples are identified
     fresh here, not loaded from a cache: we greedily generate a baseline
     completion for each candidate TruthfulQA question and label it via
     this paper's own established Jaccard word-overlap scheme
     (LABEL_THRESHOLD=0.12, identical to 07_multi_arch_causal_patch.py),
     keeping only completions labeled "correct" (matching the intent of
     the GPT-2 script's `label==1` filter on its own cached labels).
  2. Qwen2.5-0.5B-Instruct is queried with its proper chat template, not
     a bare "Q: ...\nA:" template -- matching this paper's own
     round-5-review finding (07_multi_arch_causal_patch.py) that a bare
     template is an uncontrolled OOD confound for this instruction-tuned
     model. Pythia-410M (a base, non-chat model) uses the bare template,
     as GPT-2 did.

Everything else -- Gaussian question-span corruption at 3x embedding std,
per-(layer, component) restoration scoring clipped to [-1, 2], a
shuffled-activation specificity control, paired Wilcoxon tests, and
Holm-Bonferroni correction across all (component x layer) tests -- is
unchanged from 08_rome_style_causal_tracing.py.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "multi_arch_rome_style_causal_tracing.json"

RANDOM_STATE = 42
N_EXAMPLES = 100
NOISE_SCALE = 3.0        # x empirical embedding std, matching ROME's convention and 08's GPT-2 run
LABEL_THRESHOLD = 0.12   # identical to 07_multi_arch_causal_patch.py
MAX_NEW_TOKENS = 40      # identical to 07_multi_arch_causal_patch.py

MODEL_CONFIGS = {
    "pythia": {
        "hf_id": "EleutherAI/pythia-410m",
        "layers_attr": lambda m: m.gpt_neox.layers,
        "embed_attr": lambda m: m.gpt_neox.embed_in,
        "mlp_attr": "mlp",
        "attn_attr": "attention",
        "chat_template": False,
    },
    "qwen05chat": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers_attr": lambda m: m.model.layers,
        "embed_attr": lambda m: m.model.embed_tokens,
        "mlp_attr": "mlp",
        "attn_attr": "self_attn",
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


def build_prompt(tokenizer, question, use_chat_template):
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
    return f"Q: {question}\nA:"


def first_content_token_id(tokenizer, answer: str):
    ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def find_question_span(tokenizer, prompt: str, question: str):
    """Token index range [start, end) covering the question content only."""
    prefix = prompt.split(question)[0]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    start = len(prefix_ids)
    end = start + len(q_ids)
    return start, end


def generate_baseline(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_model(model_key, cfg, qa_pool):
    print(f"\n{'='*70}\n{model_key}\n{'='*70}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32).to(device)
    model.eval()

    layers = cfg["layers_attr"](model)
    n_layers = len(layers)
    embed = cfg["embed_attr"](model)

    rng = random.Random(RANDOM_STATE)
    pool = list(qa_pool)
    rng.shuffle(pool)

    # Stage 1: greedily generate + label each candidate; keep "clean" (label==1)
    # examples with a valid, distinct correct/incorrect first-content-token pair,
    # until N_EXAMPLES are collected or the pool is exhausted.
    candidates = []
    n_checked = 0
    for item in pool:
        if len(candidates) >= N_EXAMPLES:
            break
        n_checked += 1
        question = item["question"].strip()
        correct_answers, incorrect_answers = item["correct_answers"], item["incorrect_answers"]
        if not correct_answers or not incorrect_answers:
            continue
        correct_tok = first_content_token_id(tokenizer, correct_answers[0])
        incorrect_tok = first_content_token_id(tokenizer, incorrect_answers[0])
        if correct_tok is None or incorrect_tok is None or correct_tok == incorrect_tok:
            continue
        prompt = build_prompt(tokenizer, question, cfg["chat_template"])
        completion = generate_baseline(prompt, model, tokenizer, device)
        label = label_completion(completion, correct_answers, incorrect_answers)
        if label != 1:
            continue
        candidates.append((prompt, question, correct_tok, incorrect_tok))
        if len(candidates) % 20 == 0:
            print(f"  clean examples found: {len(candidates)} (checked {n_checked}/{len(pool)})", flush=True)

    print(f"Collected {len(candidates)} clean examples after checking {n_checked}/{len(pool)} "
          f"candidates ({model_key})")

    with torch.no_grad():
        emb_std = embed.weight.std().item()
    noise_std = NOISE_SCALE * emb_std
    print(f"Embedding std={emb_std:.4f}, noise std={noise_std:.4f}")

    # Stage 2: precompute per-example clean/corrupted logit_diff + cached clean activations
    examples = []
    for prompt, question, correct_tok, incorrect_tok in candidates:
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        q_start, q_end = find_question_span(tokenizer, prompt, question)
        if q_end > input_ids.shape[1] or q_start >= q_end:
            continue

        clean_cache = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                clean_cache[name] = out_t[0, -1, :].detach().clone()
            return hook

        for i in range(n_layers):
            hooks.append(getattr(layers[i], cfg["mlp_attr"]).register_forward_hook(make_hook(f"mlp_{i}")))
            hooks.append(getattr(layers[i], cfg["attn_attr"]).register_forward_hook(make_hook(f"attn_{i}")))

        with torch.no_grad():
            clean_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.remove()
        clean_logit_diff = (clean_logits[correct_tok] - clean_logits[incorrect_tok]).item()

        with torch.no_grad():
            inputs_embeds = embed(input_ids).clone()
            noise = torch.randn(q_end - q_start, inputs_embeds.shape[-1], device=device) * noise_std
            inputs_embeds[0, q_start:q_end, :] += noise
            corrupted_logits = model(inputs_embeds=inputs_embeds).logits[0, -1, :]
        corrupted_logit_diff = (corrupted_logits[correct_tok] - corrupted_logits[incorrect_tok]).item()

        examples.append({
            "inputs_embeds_corrupted": inputs_embeds,
            "correct_tok": correct_tok, "incorrect_tok": incorrect_tok,
            "clean_logit_diff": clean_logit_diff, "corrupted_logit_diff": corrupted_logit_diff,
            "clean_cache": clean_cache,
        })

    valid = [e for e in examples if e["clean_logit_diff"] > e["corrupted_logit_diff"]]
    print(f"{len(valid)}/{len(examples)} examples show real degradation from corruption "
          f"(clean_logit_diff > corrupted_logit_diff)")

    def restoration_pass(ex, layer_idx, component, source_activation):
        target_module = getattr(layers[layer_idx], cfg["mlp_attr"] if component == "mlp" else cfg["attn_attr"])

        def patch_hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            out_t = out[0] if is_tuple else out
            out_t = out_t.clone()
            out_t[0, -1, :] = source_activation
            return (out_t,) + out[1:] if is_tuple else out_t

        handle = target_module.register_forward_hook(patch_hook)
        with torch.no_grad():
            logits = model(inputs_embeds=ex["inputs_embeds_corrupted"]).logits[0, -1, :]
        handle.remove()
        return (logits[ex["correct_tok"]] - logits[ex["incorrect_tok"]]).item()

    results_own = {f"{c}_{l}": [] for c in ["mlp", "attn"] for l in range(n_layers)}
    results_shuffled = {f"{c}_{l}": [] for c in ["mlp", "attn"] for l in range(n_layers)}

    if len(valid) < 2:
        print(f"WARNING: fewer than 2 valid examples for {model_key} -- cannot run a specificity "
              f"control (needs a distinct 'other' example). Skipping the tracing sweep.")
        return {
            "model": cfg["hf_id"], "n_layers": n_layers, "n_checked": n_checked,
            "n_clean_examples": len(candidates), "n_valid_degraded": len(valid),
            "insufficient_data": True,
        }

    for idx, ex in enumerate(valid):
        denom = ex["clean_logit_diff"] - ex["corrupted_logit_diff"]
        other = valid[(idx + 1) % len(valid)]
        for component in ["mlp", "attn"]:
            for layer_idx in range(n_layers):
                key = f"{component}_{layer_idx}"
                own_act = ex["clean_cache"][f"{component}_{layer_idx}"]
                patched_diff = restoration_pass(ex, layer_idx, component, own_act)
                score = (patched_diff - ex["corrupted_logit_diff"]) / denom
                results_own[key].append(float(np.clip(score, -1, 2)))

                other_act = other["clean_cache"][f"{component}_{layer_idx}"]
                patched_diff_ctrl = restoration_pass(ex, layer_idx, component, other_act)
                score_ctrl = (patched_diff_ctrl - ex["corrupted_logit_diff"]) / denom
                results_shuffled[key].append(float(np.clip(score_ctrl, -1, 2)))
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(valid)}]", flush=True)

    from scipy.stats import wilcoxon

    paired_tests = {}
    raw_p_values = []
    keys_ordered = [f"{c}_{l}" for c in ["mlp", "attn"] for l in range(n_layers)]
    for key in keys_ordered:
        own = np.array(results_own[key])
        shuf = np.array(results_shuffled[key])
        diff = own - shuf
        if np.allclose(diff, 0.0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(own, shuf)
        paired_tests[key] = {
            "own_minus_shuffled_mean": float(np.mean(diff)),
            "wilcoxon_stat": float(stat) if stat == stat else None,
            "wilcoxon_p_uncorrected": float(p),
        }
        raw_p_values.append(p)

    # Holm-Bonferroni across all n_layers*2 (component x layer) paired tests
    order = np.argsort(raw_p_values)
    n_tests = len(raw_p_values)
    holm_significant = [False] * n_tests
    for rank, idx in enumerate(order):
        threshold = 0.05 / (n_tests - rank)
        if raw_p_values[idx] < threshold:
            holm_significant[idx] = True
        else:
            break
    for idx, key in enumerate(keys_ordered):
        paired_tests[key]["holm_bonferroni_significant"] = bool(holm_significant[idx])

    summary = {
        "model": cfg["hf_id"], "n_checked": n_checked, "n_clean_examples": len(candidates),
        "n_valid_degraded": len(valid), "noise_scale": NOISE_SCALE, "noise_std": noise_std,
        "n_layers": n_layers, "n_tests": n_tests,
        "own_activation_restoration": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                        for k, v in results_own.items()},
        "shuffled_activation_control": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                         for k, v in results_shuffled.items()},
        "paired_specificity_tests": paired_tests,
    }

    print(f"\n=== {model_key}: paired specificity tests (own vs shuffled), "
          f"Holm-Bonferroni across {n_tests} tests ===")
    for key in keys_ordered:
        t = paired_tests[key]
        flag = "***" if t["holm_bonferroni_significant"] else ""
        print(f"{key}: own-shuffled={t['own_minus_shuffled_mean']:+.3f}  "
              f"p={t['wilcoxon_p_uncorrected']:.4f} {flag}")

    del model
    return summary


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
            print(f"Skipping {model_key} (already completed, in {OUT_PATH})")
            continue
        all_results[model_key] = run_model(model_key, cfg, qa_pool)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Checkpoint saved: {OUT_PATH}")

    print(f"\nFinal save: {OUT_PATH}")


if __name__ == "__main__":
    main()

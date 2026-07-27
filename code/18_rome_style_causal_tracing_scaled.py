"""
Paper 1 -- elite-review follow-up: scales the ROME-style causal-tracing
sweep (08_rome_style_causal_tracing.py) from N_EXAMPLES=100
(n_valid_degraded=45) toward the full available candidate pool, to raise
statistical power on the one borderline result (Attn L9, p=0.0026,
narrowly missing the joint 24-test Holm threshold of 0.05/24=0.00208).
Per the elite-review recommendation, the JOINT (24-test) correction is
pre-registered here as the primary framing before this rerun, not
selected post-hoc from whichever of joint/per-family looks better --
the per-family framing is still reported alongside it for continuity
with 08's original report, but joint is the one this rerun is designed
to move.

Everything else is identical to 08_rome_style_causal_tracing.py
(addressing the elite-reviewer critique that mean-shift steering is a
weak causal instrument relative to the field's gold-standard method):

Protocol (adapted from ROME to closed-book QA, no single clean "subject
span" assumed -- corrupts the whole question span instead):
  1. CLEAN run: forward the prompt "Q: {question}\nA:" through GPT-2,
     cache every layer's MLP-output and Attn-output activations at the
     LAST token position, and compute logit_diff = logit(first token of
     a correct reference answer) - logit(first token of an incorrect
     reference answer). This is a forced-choice discrimination score,
     not a generation -- avoids conflating causal tracing with sampling
     noise.
  2. CORRUPTED run: add Gaussian noise (3x the empirical std of GPT-2's
     token embeddings, matching ROME's convention) to the question-token
     embeddings only (leaving "Q:"/"A:" scaffolding tokens clean), and
     recompute logit_diff on the corrupted forward pass -- this should
     be more degraded (lower) than the clean logit_diff, confirming the
     corruption actually hurts the discrimination.
  3. RESTORATION sweep: for each layer l in {0..11} and each component
     in {mlp, attn}, redo the corrupted forward pass but patch in the
     CLEAN run's cached component output at that layer, last-token
     position only (leaving all other positions/layers corrupted).
     Record the resulting logit_diff. Restoration score =
     (patched - corrupted) / (clean - corrupted), clipped to [-1, 2].
  4. SPECIFICITY CONTROL: repeat step 3, but patch in a DIFFERENT,
     randomly-chosen example's clean activation instead of this
     example's own clean activation (same layer/component, same
     last-token position). If restoration is roughly as strong from a
     mismatched example's activation, the effect is generic (e.g. just
     injecting large-norm activity), not specific restoration of this
     example's discriminative content.

This produces a full (layer x component) causal-tracing curve, averaged
over N clean examples, plus a shuffled-activation control curve -- a
richer, more standard mech-interp result than a single found-vs-random
comparison at 2 layers x 3 alphas.
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
OUT_PATH = ROOT / "results" / "rome_style_causal_tracing_scaled.json"

RANDOM_STATE = 42
N_EXAMPLES = 300  # raised from 100; will be truncated to whatever the labeled pool actually supports
NOISE_SCALE = 3.0  # x empirical embedding std, matching ROME's convention


def question_from_prompt(prompt: str) -> str:
    q = prompt.split("Q:", 1)[1].split("\nA:", 1)[0]
    return q.strip()


def build_question_answer_map():
    print("Loading TruthfulQA (generation, validation) for answer lookup...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap = {}
    for item in ds:
        qmap[item["question"].strip()] = (item["correct_answers"], item["incorrect_answers"])
    print(f"  {len(qmap)} questions indexed")
    return qmap


def first_content_token_id(tokenizer, answer: str):
    # first token of " {answer}" (leading space matches GPT-2 BPE convention)
    ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def find_question_span(tokenizer, prompt: str):
    """Token index range [start, end) covering the question content only,
    excluding the 'Q: ' prefix and '\\nA:' suffix scaffolding."""
    q = question_from_prompt(prompt)
    prefix = prompt.split(q)[0]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    q_ids = tokenizer.encode(q, add_special_tokens=False)
    start = len(prefix_ids)
    end = start + len(q_ids)
    return start, end


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    with open(VENDORED / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)

    qmap = build_question_answer_map()

    # Select clean (label==1) examples with a usable correct/incorrect answer pair
    rng = random.Random(RANDOM_STATE)
    candidates = []
    for prompt, label in zip(labeled["prompts"], labeled["labels"]):
        if label != 1:
            continue
        q = question_from_prompt(prompt)
        if q not in qmap:
            continue
        correct_answers, incorrect_answers = qmap[q]
        if not correct_answers or not incorrect_answers:
            continue
        correct_tok = first_content_token_id(tokenizer, correct_answers[0])
        incorrect_tok = first_content_token_id(tokenizer, incorrect_answers[0])
        if correct_tok is None or incorrect_tok is None or correct_tok == incorrect_tok:
            continue
        candidates.append((prompt, correct_tok, incorrect_tok))

    rng.shuffle(candidates)
    candidates = candidates[:N_EXAMPLES]
    print(f"Selected {len(candidates)} clean examples with valid correct/incorrect answer pairs")

    wte = model.transformer.wte
    with torch.no_grad():
        emb_std = wte.weight.std().item()
    noise_std = NOISE_SCALE * emb_std
    print(f"Embedding std={emb_std:.4f}, noise std={noise_std:.4f}")

    n_layers = model.config.n_layer

    # Precompute per-example: clean logit_diff, corrupted logit_diff, and
    # cached clean per-layer mlp/attn last-token outputs.
    examples = []
    for prompt, correct_tok, incorrect_tok in candidates:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        q_start, q_end = find_question_span(tokenizer, prompt)

        clean_cache = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                clean_cache[name] = out_t[0, -1, :].detach().clone()
            return hook

        for i in range(n_layers):
            hooks.append(model.transformer.h[i].mlp.register_forward_hook(make_hook(f"mlp_{i}")))
            hooks.append(model.transformer.h[i].attn.register_forward_hook(make_hook(f"attn_{i}")))

        with torch.no_grad():
            clean_logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.remove()
        clean_logit_diff = (clean_logits[correct_tok] - clean_logits[incorrect_tok]).item()

        # corrupted embeddings: add noise to question-span tokens only
        with torch.no_grad():
            inputs_embeds = wte(input_ids).clone()
            noise = torch.randn(q_end - q_start, inputs_embeds.shape[-1], device=device) * noise_std
            inputs_embeds[0, q_start:q_end, :] += noise
            corrupted_logits = model(inputs_embeds=inputs_embeds).logits[0, -1, :]
        corrupted_logit_diff = (corrupted_logits[correct_tok] - corrupted_logits[incorrect_tok]).item()

        examples.append({
            "input_ids": input_ids, "inputs_embeds_corrupted": inputs_embeds,
            "correct_tok": correct_tok, "incorrect_tok": incorrect_tok,
            "clean_logit_diff": clean_logit_diff, "corrupted_logit_diff": corrupted_logit_diff,
            "clean_cache": clean_cache,
        })

    valid = [e for e in examples if e["clean_logit_diff"] > e["corrupted_logit_diff"]]
    print(f"{len(valid)}/{len(examples)} examples show real degradation from corruption "
          f"(clean_logit_diff > corrupted_logit_diff)")

    def restoration_pass(ex, layer, component, source_activation):
        target_module = (model.transformer.h[layer].mlp if component == "mlp"
                          else model.transformer.h[layer].attn)

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

    for idx, ex in enumerate(valid):
        denom = ex["clean_logit_diff"] - ex["corrupted_logit_diff"]
        other = valid[(idx + 1) % len(valid)]  # deterministic "different example" pairing
        for component in ["mlp", "attn"]:
            for layer in range(n_layers):
                key = f"{component}_{layer}"
                own_act = ex["clean_cache"][f"{component}_{layer}"]
                patched_diff = restoration_pass(ex, layer, component, own_act)
                score = (patched_diff - ex["corrupted_logit_diff"]) / denom
                results_own[key].append(float(np.clip(score, -1, 2)))

                other_act = other["clean_cache"][f"{component}_{layer}"]
                patched_diff_ctrl = restoration_pass(ex, layer, component, other_act)
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

    # Holm-Bonferroni across all 24 (component x layer) paired tests
    order = np.argsort(raw_p_values)
    n_tests = len(raw_p_values)
    holm_significant = [False] * n_tests
    for rank, idx in enumerate(order):
        threshold = 0.05 / (n_tests - rank)
        if raw_p_values[idx] < threshold:
            holm_significant[idx] = True
        else:
            break  # step-down: once one fails, all subsequent (larger p) fail too
    for idx, key in enumerate(keys_ordered):
        paired_tests[key]["holm_bonferroni_significant"] = bool(holm_significant[idx])

    summary = {
        "n_candidates": len(candidates), "n_valid_degraded": len(valid),
        "noise_scale": NOISE_SCALE, "noise_std": noise_std, "n_layers": n_layers,
        "own_activation_restoration": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                        for k, v in results_own.items()},
        "shuffled_activation_control": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                         for k, v in results_shuffled.items()},
        "paired_specificity_tests": paired_tests,
        "raw_per_example_own": {k: v for k, v in results_own.items()},
        "raw_per_example_shuffled": {k: v for k, v in results_shuffled.items()},
    }

    print("\n=== Paired specificity tests (own vs shuffled), Holm-Bonferroni across 24 tests ===")
    for key in keys_ordered:
        t = paired_tests[key]
        flag = "***" if t["holm_bonferroni_significant"] else ""
        print(f"{key}: own-shuffled={t['own_minus_shuffled_mean']:+.3f}  p={t['wilcoxon_p_uncorrected']:.4f} {flag}")

    print("\n=== Per-layer restoration (own clean activation) ===")
    for component in ["mlp", "attn"]:
        means = [summary["own_activation_restoration"][f"{component}_{l}"]["mean"] for l in range(n_layers)]
        print(f"{component}: " + " ".join(f"{m:.3f}" for m in means))
    print("\n=== Per-layer restoration (shuffled/mismatched clean activation, control) ===")
    for component in ["mlp", "attn"]:
        means = [summary["shuffled_activation_control"][f"{component}_{l}"]["mean"] for l in range(n_layers)]
        print(f"{component}: " + " ".join(f"{m:.3f}" for m in means))

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

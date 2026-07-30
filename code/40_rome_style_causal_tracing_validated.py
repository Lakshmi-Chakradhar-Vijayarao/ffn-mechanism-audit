"""
Paper 1 -- reruns the ROME-style causal-tracing sweep (code/18) under the
validated judge label, per a fresh audit finding: code/18's candidate
pool is filtered on Jaccard label==1 ("clean" examples), but §4
establishes that only 27/534 GPT-2 samples are actually judge-correct --
so roughly 95% of the "clean" examples driving this paper's self-declared
*stronger* causal test are, on the paper's own validated-label numbers,
actually hallucinations. This is a more structurally serious dependency
on the discredited label than §3.4 has (which reruns under the validated
label; §3.5 never had).

Changes from code/18, everything else kept identical so the comparison
is clean:
  1. Candidate filter: judge_label==1 (from results/gpt2_full_534_judge_labels.json,
     positionally aligned with vendored_mech_int/labeled.pkl -- established
     this session) instead of Jaccard label==1. Given only 27 judge-correct
     GPT-2 samples exist, this candidate pool is small; reported explicitly
     as power-limited rather than silently run on an inadequate sample.
  2. N_NOISE_DRAWS=10 independent corruption draws per example (was 1),
     averaging both the corrupted-logit-diff denominator and every
     restoration score across draws -- removes the original's selection
     on a single noisy realization (the old `valid = [... clean > corrupted]`
     filter, which conditioned the retained denominators on that specific
     draw, is dropped entirely; every candidate is retained and its
     10-draw-averaged degradation is used as-is, whatever sign it has).
  3. Mismatched-donor control: 10 randomly sampled donors per example,
     averaged (was a single deterministic ring-neighbor `(idx+1) % len`).
  4. Both clipped ([-1,2], matching code/18) and unclipped restoration
     scores are saved; the clipped version's boundary-hit rate is reported
     so a reader can see how much of the clipped distribution is sitting
     at the clip boundaries.
  5. Holm-Bonferroni applied twice: across this test's own 24
     (component x layer) comparisons (primary, matching code/18's own
     framing), and across a 120-test study-wide family (24 x 5, matching
     the audit's suggested accounting for GPT-2 + the two other tested
     architectures' equivalent sweeps, even though only GPT-2 is rerun
     here).

NOT changed in this round (explicit, disclosed limitation): the
forced-choice score is still first-token log(correct) - log(incorrect)
(code/18's `first_content_token_id`), not a length-normalized
full-sequence log-probability. Implementing the latter would require
patching activations mid-sequence while scoring a multi-token teacher-forced
continuation -- a materially larger redesign of the causal-tracing hook
machinery than was tractable in this round; deferred and disclosed rather
than attempted partially.
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import wilcoxon
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
OUT_PATH = ROOT / "results" / "rome_style_causal_tracing_validated.json"

RANDOM_STATE = 42
NOISE_SCALE = 3.0
N_NOISE_DRAWS = 10
N_DONOR_DRAWS = 10


def question_from_prompt(prompt: str) -> str:
    return prompt.split("Q:", 1)[1].split("\nA:", 1)[0].strip()


def build_question_answer_map():
    ds = load_dataset("truthful_qa", "generation", split="validation")
    return {item["question"].strip(): (item["correct_answers"], item["incorrect_answers"]) for item in ds}


def first_content_token_id(tokenizer, answer: str):
    ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    return ids[0] if ids else None


def find_question_span(tokenizer, prompt: str):
    q = question_from_prompt(prompt)
    prefix = prompt.split(q)[0]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    q_ids = tokenizer.encode(q, add_special_tokens=False)
    start = len(prefix_ids)
    return start, start + len(q_ids)


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    with open(VENDORED / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    judge_data = json.load(open(JUDGE_LABELS_PATH))
    judge_labels = judge_data["judge_labels"]
    assert len(judge_labels) == len(labeled["prompts"]), (len(judge_labels), len(labeled["prompts"]))

    qmap = build_question_answer_map()

    rng = random.Random(RANDOM_STATE)
    candidates = []
    for prompt, jlabel in zip(labeled["prompts"], judge_labels):
        if jlabel != 1:
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
    print(f"Candidate pool under VALIDATED judge label: {len(candidates)} "
          f"(out of {judge_data['n_correct']} total judge-correct GPT-2 samples) -- "
          f"power-limited by design, disclosed explicitly.", flush=True)

    wte = model.transformer.wte
    with torch.no_grad():
        emb_std = wte.weight.std().item()
    noise_std = NOISE_SCALE * emb_std
    n_layers = model.config.n_layer
    print(f"Embedding std={emb_std:.4f}, noise std={noise_std:.4f}", flush=True)

    def capture_clean_cache(input_ids):
        cache = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                cache[name] = out_t[0, -1, :].detach().clone()
            return hook

        for i in range(n_layers):
            hooks.append(model.transformer.h[i].mlp.register_forward_hook(make_hook(f"mlp_{i}")))
            hooks.append(model.transformer.h[i].attn.register_forward_hook(make_hook(f"attn_{i}")))
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        for h in hooks:
            h.remove()
        return logits, cache

    def restoration_pass(inputs_embeds_corrupted, correct_tok, incorrect_tok, layer, component, source_activation):
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
            logits = model(inputs_embeds=inputs_embeds_corrupted).logits[0, -1, :]
        handle.remove()
        return (logits[correct_tok] - logits[incorrect_tok]).item()

    # Precompute clean activations/logit_diff (deterministic, one pass per example)
    examples = []
    for prompt, correct_tok, incorrect_tok in candidates:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        q_start, q_end = find_question_span(tokenizer, prompt)
        clean_logits, clean_cache = capture_clean_cache(input_ids)
        clean_logit_diff = (clean_logits[correct_tok] - clean_logits[incorrect_tok]).item()
        examples.append({
            "input_ids": input_ids, "q_start": q_start, "q_end": q_end,
            "correct_tok": correct_tok, "incorrect_tok": incorrect_tok,
            "clean_logit_diff": clean_logit_diff, "clean_cache": clean_cache,
        })

    n = len(examples)
    print(f"Running {N_NOISE_DRAWS}-draw-averaged restoration sweep on {n} examples...", flush=True)

    keys_ordered = [f"{c}_{l}" for c in ["mlp", "attn"] for l in range(n_layers)]
    results_own_clipped = {k: [] for k in keys_ordered}
    results_own_unclipped = {k: [] for k in keys_ordered}
    results_shuffled_clipped = {k: [] for k in keys_ordered}
    results_shuffled_unclipped = {k: [] for k in keys_ordered}
    n_degraded_draws = 0
    n_total_draws = 0

    for idx, ex in enumerate(examples):
        input_ids = ex["input_ids"]
        q_start, q_end = ex["q_start"], ex["q_end"]
        correct_tok, incorrect_tok = ex["correct_tok"], ex["incorrect_tok"]
        clean_diff = ex["clean_logit_diff"]

        donor_pool = [e for j, e in enumerate(examples) if j != idx]
        donor_examples = [donor_pool[rng.randrange(len(donor_pool))] for _ in range(N_DONOR_DRAWS)]

        per_key_own_scores = {k: [] for k in keys_ordered}
        per_key_shuf_scores = {k: [] for k in keys_ordered}

        for draw in range(N_NOISE_DRAWS):
            with torch.no_grad():
                inputs_embeds = wte(input_ids).clone()
                noise = torch.randn(q_end - q_start, inputs_embeds.shape[-1], device=device) * noise_std
                inputs_embeds[0, q_start:q_end, :] += noise
                corrupted_logits = model(inputs_embeds=inputs_embeds).logits[0, -1, :]
            corrupted_diff = (corrupted_logits[correct_tok] - corrupted_logits[incorrect_tok]).item()
            denom = clean_diff - corrupted_diff
            n_total_draws += 1
            if denom > 0:
                n_degraded_draws += 1
            if abs(denom) < 1e-6:
                denom = 1e-6 if denom >= 0 else -1e-6

            for component in ["mlp", "attn"]:
                for layer in range(n_layers):
                    key = f"{component}_{layer}"
                    own_act = ex["clean_cache"][f"{component}_{layer}"]
                    patched_diff = restoration_pass(inputs_embeds, correct_tok, incorrect_tok, layer, component, own_act)
                    per_key_own_scores[key].append((patched_diff - corrupted_diff) / denom)

                    donor = donor_examples[draw % len(donor_examples)]
                    donor_act = donor["clean_cache"][f"{component}_{layer}"]
                    patched_diff_ctrl = restoration_pass(inputs_embeds, correct_tok, incorrect_tok, layer, component, donor_act)
                    per_key_shuf_scores[key].append((patched_diff_ctrl - corrupted_diff) / denom)

        for key in keys_ordered:
            own_mean_unclipped = float(np.mean(per_key_own_scores[key]))
            shuf_mean_unclipped = float(np.mean(per_key_shuf_scores[key]))
            results_own_unclipped[key].append(own_mean_unclipped)
            results_shuffled_unclipped[key].append(shuf_mean_unclipped)
            results_own_clipped[key].append(float(np.clip(own_mean_unclipped, -1, 2)))
            results_shuffled_clipped[key].append(float(np.clip(shuf_mean_unclipped, -1, 2)))

        if (idx + 1) % 5 == 0 or (idx + 1) == n:
            print(f"  [{idx+1}/{n}]", flush=True)

    print(f"\nDegradation rate across all {n_total_draws} individual corruption draws: "
          f"{n_degraded_draws}/{n_total_draws} = {n_degraded_draws/n_total_draws:.3f} "
          f"(no longer used as a selection filter -- all examples retained regardless)", flush=True)

    paired_tests = {}
    raw_p_values = []
    for key in keys_ordered:
        own = np.array(results_own_clipped[key])
        shuf = np.array(results_shuffled_clipped[key])
        diff = own - shuf
        if np.allclose(diff, 0.0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(own, shuf)
        n_clip_low_own = int(np.sum(np.array(results_own_unclipped[key]) < -1))
        n_clip_high_own = int(np.sum(np.array(results_own_unclipped[key]) > 2))
        paired_tests[key] = {
            "own_minus_shuffled_mean_clipped": float(np.mean(diff)),
            "own_minus_shuffled_mean_unclipped": float(np.mean(np.array(results_own_unclipped[key]) - np.array(results_shuffled_unclipped[key]))),
            "wilcoxon_stat": float(stat) if stat == stat else None,
            "wilcoxon_p_uncorrected": float(p),
            "n_boundary_hits_own_low": n_clip_low_own, "n_boundary_hits_own_high": n_clip_high_own,
        }
        raw_p_values.append(p)

    def holm_correct(pvals, n_family):
        order = np.argsort(pvals)
        sig = [False] * len(pvals)
        for rank, idx in enumerate(order):
            threshold = 0.05 / (n_family - rank)
            if pvals[idx] < threshold:
                sig[idx] = True
            else:
                break
        return sig

    holm_24 = holm_correct(raw_p_values, 24)
    holm_120 = holm_correct(raw_p_values, 120)
    for i, key in enumerate(keys_ordered):
        paired_tests[key]["holm_significant_24_test_family"] = bool(holm_24[i])
        paired_tests[key]["holm_significant_120_test_study_wide_family"] = bool(holm_120[i])

    summary = {
        "n_candidates_judge_correct_pool": judge_data["n_correct"],
        "n_candidates_used": n, "power_limited": n < 40,
        "noise_scale": NOISE_SCALE, "noise_std": noise_std, "n_layers": n_layers,
        "n_noise_draws": N_NOISE_DRAWS, "n_donor_draws": N_DONOR_DRAWS,
        "degradation_rate_across_draws": n_degraded_draws / n_total_draws,
        "own_activation_restoration_clipped": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in results_own_clipped.items()},
        "shuffled_activation_control_clipped": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in results_shuffled_clipped.items()},
        "paired_specificity_tests": paired_tests,
        "raw_per_example_own_unclipped": {k: v for k, v in results_own_unclipped.items()},
        "raw_per_example_shuffled_unclipped": {k: v for k, v in results_shuffled_unclipped.items()},
    }

    print("\n=== Paired specificity tests (own vs shuffled, clipped), Holm across 24 vs 120 tests ===")
    for key in keys_ordered:
        t = paired_tests[key]
        flag24 = "***24***" if t["holm_significant_24_test_family"] else ""
        flag120 = "***120***" if t["holm_significant_120_test_study_wide_family"] else ""
        print(f"{key}: own-shuffled={t['own_minus_shuffled_mean_clipped']:+.3f} "
              f"(unclipped={t['own_minus_shuffled_mean_unclipped']:+.3f}) p={t['wilcoxon_p_uncorrected']:.4f} {flag24} {flag120}")

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

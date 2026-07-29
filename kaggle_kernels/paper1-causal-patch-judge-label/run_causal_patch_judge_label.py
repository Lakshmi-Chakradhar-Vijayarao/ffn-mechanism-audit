"""
Paper 1 -- the decisive follow-up to the LLM-judge label-validity audit.

The passive AUROC probe (paper1-llm-judge-relabel kernel) showed that
relabeling with a validated LLM judge instead of the Jaccard word-overlap
heuristic changes the FFN-vs-Attention picture substantially (higher
AUROCs, FFN's majority restored on Qwen0.5B-chat). But the paper's actual
load-bearing claim is the CAUSAL patching null (Section 3.4), not the
passive AUROC. This kernel redoes that causal test end to end using the
judge label instead of Jaccard, for GPT-2 (the architecture with by far
the best power for this test):

1. Judge all 534 of GPT-2's already-generated, already-Jaccard-labeled
   completions (results/vendored_mech_int/labeled.pkl, uploaded here as
   a Kaggle dataset) with the same LLM judge used throughout this project
   (Qwen2.5-3B-Instruct).
2. Split into train/test using an explicit per-class allocation (not a
   uniform fraction), since the judge label is expected to be severely
   imbalanced (as it was for Pythia/Qwen0.5B): put a fixed, small number
   of each class into train (enough for a stable mean-difference
   direction), and reserve everything else -- especially the
   judge-hallucinated majority -- for test, maximizing power.
3. Compute the FFN found-direction at L8/L9 from the JUDGE-labeled train
   split (identical methodology to 10_ffn_causal_patch_scaled.py).
4. Run the identical causal-patching loop (baseline + FFN-found/
   FFN-random/Attn-found at L8/L9 x alpha{20,40}) on every
   judge-hallucinated test prompt.
5. Score EVERY generated completion (baseline and all patched variants)
   with BOTH the judge and the original Jaccard heuristic, so the
   McNemar flip-to-correct tests can be computed and directly compared
   under both labeling schemes in one run.
"""
import base64
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

OUT_DIR = Path("/kaggle/working")

# __LABELED_DATA_B64__ is replaced with the real gzip+base64-encoded
# contents of results/vendored_mech_int/labeled.pkl (prompts/labels/
# completions) by build_kernel.py before this script is pushed to Kaggle.
LABELED_DATA_B64 = "<omitted -- generated from results/vendored_mech_int/labeled.pkl at push time; see build script in scratch history>"


def load_labeled_data():
    compressed = base64.b64decode(LABELED_DATA_B64)
    return json.loads(gzip.decompress(compressed))

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LAYER = 8
SECOND_LAYER = 9
ALPHAS = [20.0, 40.0]
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12
TRAIN_N_PER_CLASS = 40  # matches the paper's own "minimum for a stable per-class mean" rationale
RANDOM_STATE = 42

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
    if "CORRECT" in verdict and "HALLUCINAT" not in verdict:
        return 1
    if "HALLUCINAT" in verdict:
        return 0
    return -1


def question_from_prompt(prompt: str) -> str:
    return prompt.split("Q:", 1)[1].split("\nA:", 1)[0].strip()


def _word_overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def jaccard_label(completion, correct_answers, incorrect_answers):
    best_correct = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_incorrect = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_correct > LABEL_THRESHOLD or best_incorrect > LABEL_THRESHOLD:
        return 1 if best_correct >= best_incorrect else 0
    return -1


def extract_ffn_last_token(prompt, model, tokenizer, device, layer_idx):
    captured = {}

    def hook(module, inp, out):
        captured["ffn"] = out[0, -1, :].detach().cpu().float().numpy().copy()

    h = model.transformer.h[layer_idx].mlp.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        h.remove()
    return captured["ffn"]


def compute_ffn_direction(train_prompts, train_labels, model, tokenizer, device, layer_idx):
    correct_vecs, hall_vecs = [], []
    for p, y in zip(train_prompts, train_labels):
        vec = extract_ffn_last_token(p, model, tokenizer, device, layer_idx)
        (correct_vecs if y == 1 else hall_vecs).append(vec)
    direction = np.mean(correct_vecs, axis=0) - np.mean(hall_vecs, axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("FFN direction near zero.")
    return direction / norm, len(correct_vecs), len(hall_vecs)


def random_orthogonal_direction(direction, seed=42):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(direction.shape)
    v -= np.dot(v, direction) * direction
    v /= np.linalg.norm(v)
    return v


def patched_generate(prompt, model, tokenizer, device, layer_idx, direction, alpha, sublayer):
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        if sublayer == "mlp":
            return out + alpha * direction_t
        modified = out[0] + alpha * direction_t
        return (modified,) + out[1:]

    module = (model.transformer.h[layer_idx].mlp if sublayer == "mlp"
              else model.transformer.h[layer_idx].attn)
    handle = module.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        handle.remove()


def unpatched_generate(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def mcnemar_summary(rows, key, label_field_found, label_field_rand, label_field_attn):
    ffn_found_flip = [r[f"{key}_{label_field_found}"] == 1 for r in rows]
    ffn_rand_flip = [r[f"{key}_{label_field_rand}"] == 1 for r in rows]
    attn_found_flip = [r[f"{key}_{label_field_attn}"] == 1 for r in rows]
    b = sum(1 for f, rr in zip(ffn_found_flip, ffn_rand_flip) if f and not rr)
    c = sum(1 for f, rr in zip(ffn_found_flip, ffn_rand_flip) if not f and rr)
    mcnemar_p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    b2 = sum(1 for f, a in zip(ffn_found_flip, attn_found_flip) if f and not a)
    c2 = sum(1 for f, a in zip(ffn_found_flip, attn_found_flip) if not f and a)
    mcnemar_p_attn = binomtest(b2, b2 + c2, 0.5).pvalue if (b2 + c2) > 0 else 1.0
    return {
        "n": len(rows),
        "ffn_found_flip_rate": round(float(np.mean(ffn_found_flip)), 4),
        "ffn_random_flip_rate": round(float(np.mean(ffn_rand_flip)), 4),
        "attn_found_flip_rate": round(float(np.mean(attn_found_flip)), 4),
        "mcnemar_found_vs_random_b": b, "mcnemar_found_vs_random_c": c,
        "mcnemar_found_vs_random_p": round(float(mcnemar_p), 4),
        "mcnemar_found_vs_attn_b": b2, "mcnemar_found_vs_attn_c": c2,
        "mcnemar_found_vs_attn_p": round(float(mcnemar_p_attn), 4),
    }


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    labeled = load_labeled_data()
    prompts, jaccard_labels_orig, completions = labeled["prompts"], labeled["labels"], labeled["completions"]
    print(f"Loaded {len(prompts)} GPT-2 labeled examples "
          f"(Jaccard: {sum(jaccard_labels_orig)} correct / {len(jaccard_labels_orig)-sum(jaccard_labels_orig)} hallucinated)", flush=True)

    print(f"Loading TruthfulQA for question->answer maps...", flush=True)
    ds = load_dataset("truthful_qa", "generation", split="validation")
    qmap_judge = {item["question"].strip(): item["best_answer"] for item in ds}
    qmap_jaccard = {item["question"].strip(): (item["correct_answers"], item["incorrect_answers"]) for item in ds}

    print(f"Loading judge model {JUDGE_MODEL}...", flush=True)
    judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    judge_dtype = torch.float16 if device == "cuda" else torch.float32
    judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, torch_dtype=judge_dtype).to(device)
    judge_model.eval()

    print(f"\nJudging all {len(prompts)} original GPT-2 completions...", flush=True)
    judge_labels = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        q = question_from_prompt(prompt)
        ref = qmap_judge.get(q)
        if ref is None:
            judge_labels.append(-1)
            continue
        judge_labels.append(query_judge(judge_model, judge_tok, device, q, ref, completion))
        if (i + 1) % 100 == 0:
            print(f"  judged {i+1}/{len(prompts)}", flush=True)

    judge_labels = np.array(judge_labels)
    valid_mask = judge_labels != -1
    n_correct = int((judge_labels[valid_mask] == 1).sum())
    n_hall = int((judge_labels[valid_mask] == 0).sum())
    print(f"\nJudge labeling done: n_valid={valid_mask.sum()} correct={n_correct} hallucinated={n_hall} "
          f"(hallucination_rate={n_hall/(n_correct+n_hall):.3f})", flush=True)

    agreement = float((judge_labels[valid_mask] == np.array(jaccard_labels_orig)[valid_mask]).mean())
    print(f"Agreement with original Jaccard label: {agreement:.4f}", flush=True)

    # ---- Per-class train/test split, adaptive to the judge label's class balance ----
    valid_idx = np.where(valid_mask)[0]
    correct_idx = [i for i in valid_idx if judge_labels[i] == 1]
    hall_idx = [i for i in valid_idx if judge_labels[i] == 0]
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(correct_idx)
    rng.shuffle(hall_idx)

    n_train_correct = min(TRAIN_N_PER_CLASS, int(0.7 * len(correct_idx)) if len(correct_idx) < TRAIN_N_PER_CLASS * 2 else TRAIN_N_PER_CLASS)
    n_train_hall = min(TRAIN_N_PER_CLASS, int(0.7 * len(hall_idx)) if len(hall_idx) < TRAIN_N_PER_CLASS * 2 else TRAIN_N_PER_CLASS)
    n_train_correct = max(n_train_correct, min(5, len(correct_idx)))
    n_train_hall = max(n_train_hall, min(5, len(hall_idx)))

    train_correct_idx = correct_idx[:n_train_correct]
    train_hall_idx = hall_idx[:n_train_hall]
    train_idx = train_correct_idx + train_hall_idx
    hallucinated_test_idx = hall_idx[n_train_hall:]  # everything else in the hallucinated class -> test

    print(f"\nTrain: {len(train_idx)} ({len(train_correct_idx)} correct, {len(train_hall_idx)} hallucinated)")
    print(f"Test (judge-hallucinated, all remaining): {len(hallucinated_test_idx)}", flush=True)

    train_prompts = [prompts[i] for i in train_idx]
    train_labels_judge = [int(judge_labels[i]) for i in train_idx]

    print(f"\nLoading GPT-2...", flush=True)
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2.eval()

    print(f"Computing FFN found-direction at L{TARGET_LAYER} (judge-labeled train, n={len(train_idx)})...", flush=True)
    direction_l8, nc8, nh8 = compute_ffn_direction(train_prompts, train_labels_judge, gpt2, gpt2_tok, device, TARGET_LAYER)
    random_l8 = random_orthogonal_direction(direction_l8, seed=RANDOM_STATE)
    print(f"Computing FFN found-direction at L{SECOND_LAYER}...", flush=True)
    direction_l9, nc9, nh9 = compute_ffn_direction(train_prompts, train_labels_judge, gpt2, gpt2_tok, device, SECOND_LAYER)
    random_l9 = random_orthogonal_direction(direction_l9, seed=RANDOM_STATE)
    directions = {TARGET_LAYER: (direction_l8, random_l8), SECOND_LAYER: (direction_l9, random_l9)}

    print(f"\nRunning causal-patching loop on {len(hallucinated_test_idx)} judge-hallucinated test prompts...", flush=True)
    results = []
    n = len(hallucinated_test_idx)
    for progress_i, i in enumerate(hallucinated_test_idx):
        prompt = prompts[i]
        question = question_from_prompt(prompt)
        if question not in qmap_judge or question not in qmap_jaccard:
            continue
        ref_judge = qmap_judge[question]
        correct_answers, incorrect_answers = qmap_jaccard[question]

        row = {"idx": int(i), "prompt": prompt}
        baseline_completion = unpatched_generate(prompt, gpt2, gpt2_tok, device)
        row["baseline_judge_label"] = query_judge(judge_model, judge_tok, device, question, ref_judge, baseline_completion)
        row["baseline_jaccard_label"] = jaccard_label(baseline_completion, correct_answers, incorrect_answers)

        for layer_idx in (TARGET_LAYER, SECOND_LAYER):
            found_dir, rand_dir = directions[layer_idx]
            for alpha in ALPHAS:
                ffn_found_c = patched_generate(prompt, gpt2, gpt2_tok, device, layer_idx, found_dir, alpha, "mlp")
                ffn_rand_c = patched_generate(prompt, gpt2, gpt2_tok, device, layer_idx, rand_dir, alpha, "mlp")
                attn_found_c = patched_generate(prompt, gpt2, gpt2_tok, device, layer_idx, found_dir, alpha, "attn")

                key = f"L{layer_idx}_a{int(alpha)}"
                for cname, ctext in (("ffn_found", ffn_found_c), ("ffn_rand", ffn_rand_c), ("attn_found", attn_found_c)):
                    row[f"{key}_{cname}_judge_label"] = query_judge(judge_model, judge_tok, device, question, ref_judge, ctext)
                    row[f"{key}_{cname}_jaccard_label"] = jaccard_label(ctext, correct_answers, incorrect_answers)

        results.append(row)
        if (progress_i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (progress_i + 1)) * (n - progress_i - 1) / 60
            print(f"  [{progress_i + 1}/{n}] elapsed={elapsed:.0f}s ETA={eta:.0f}min", flush=True)

    summary_judge, summary_jaccard = {}, {}
    for layer_idx in (TARGET_LAYER, SECOND_LAYER):
        for alpha in ALPHAS:
            key = f"L{layer_idx}_a{int(alpha)}"
            summary_judge[key] = mcnemar_summary(results, key, "ffn_found_judge_label", "ffn_rand_judge_label", "attn_found_judge_label")
            summary_jaccard[key] = mcnemar_summary(results, key, "ffn_found_jaccard_label", "ffn_rand_jaccard_label", "attn_found_jaccard_label")

    print("\n=== Flip-to-correct summary UNDER JUDGE LABEL ===")
    for key, s in summary_judge.items():
        print(f"  {key}: FFN-found={s['ffn_found_flip_rate']:.3f} FFN-random={s['ffn_random_flip_rate']:.3f} "
              f"Attn-found={s['attn_found_flip_rate']:.3f} "
              f"McNemar(found vs random) p={s['mcnemar_found_vs_random_p']:.4f} "
              f"McNemar(found vs attn) p={s['mcnemar_found_vs_attn_p']:.4f} (n={s['n']})")

    print("\n=== Flip-to-correct summary UNDER JACCARD LABEL (same prompts/patches, for direct comparison) ===")
    for key, s in summary_jaccard.items():
        print(f"  {key}: FFN-found={s['ffn_found_flip_rate']:.3f} FFN-random={s['ffn_random_flip_rate']:.3f} "
              f"Attn-found={s['attn_found_flip_rate']:.3f} "
              f"McNemar(found vs random) p={s['mcnemar_found_vs_random_p']:.4f} "
              f"McNemar(found vs attn) p={s['mcnemar_found_vs_attn_p']:.4f} (n={s['n']})")

    out = {
        "config": {"target_layer": TARGET_LAYER, "second_layer": SECOND_LAYER, "alphas": ALPHAS,
                   "max_new_tokens": MAX_NEW_TOKENS, "train_n_correct": len(train_correct_idx),
                   "train_n_hall": len(train_hall_idx)},
        "judge_relabel_stats": {"n_valid": int(valid_mask.sum()), "n_correct": n_correct, "n_hall": n_hall,
                                 "agreement_with_jaccard": agreement},
        "n_hallucinated_test": len(results),
        "summary_under_judge_label": summary_judge,
        "summary_under_jaccard_label": summary_jaccard,
        "per_sample": results,
    }
    out_path = OUT_DIR / "causal_patch_judge_label_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

"""
Paper 1 -- regenerate completions (Pythia-410M, Qwen2.5-0.5B-Instruct bare,
Qwen2.5-0.5B-Instruct chat-templated) so the LLM-judge label-validity audit
(code/16_llm_judge_label_noise.py, currently GPT-2-only) can be extended to
all three architectures the paper draws conclusions across.

02_cross_arch_component_probe.py already generates these completions but
never saves the completion text itself (only aggregated AUROC results and
raw activation arrays). This script reuses that exact generation logic
(same prompts, same greedy decoding, same Jaccard labeling) and saves
prompts+completions+labels to disk, so a downstream script can run the
LLM judge on a stratified sample without needing to re-run generation.

No GPU-only step is required here: this is generation only (no hook-based
activation extraction), which is far cheaper than the original script's
full run and is feasible on CPU/MPS locally, if slow.
"""
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_THRESHOLD = 0.12
MAX_NEW_TOKENS = 40

MODEL_REGISTRY = {
    "pythia": {"hf_id": "EleutherAI/pythia-410m", "chat_template": False},
    "qwen05": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "chat_template": False},
    "qwen05chat": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "chat_template": True},
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


def main(model_key):
    cfg = MODEL_REGISTRY[model_key]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{model_key}] Loading {cfg['hf_id']} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32)
    model.eval()
    model.to(device)

    dataset = load_dataset("truthful_qa", "generation", split="validation")
    use_chat_template = cfg["chat_template"]

    records = []
    for i, item in enumerate(dataset):
        question = item["question"]
        if use_chat_template:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        records.append({
            "question": question,
            "best_answer": item["best_answer"],
            "prompt": prompt,
            "completion": completion,
            "jaccard_label": label,
        })
        if (i + 1) % 50 == 0:
            kept = sum(1 for r in records if r["jaccard_label"] != -1)
            print(f"  [{model_key}] {i + 1}/{len(dataset)}  (kept so far: {kept})", flush=True)

    kept_records = [r for r in records if r["jaccard_label"] != -1]
    n_correct = sum(1 for r in kept_records if r["jaccard_label"] == 1)
    print(f"[{model_key}] DONE. kept={len(kept_records)}/{len(records)}  "
          f"correct={n_correct}  hallucinated={len(kept_records) - n_correct}")

    out_path = OUT_DIR / f"completions_for_judge_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(kept_records, f, indent=2)
    print(f"[{model_key}] Saved: {out_path}")


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "pythia"
    main(model_key)

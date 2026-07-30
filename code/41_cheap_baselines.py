"""
Paper 1 -- cheap baselines the audit flagged as planned but never run:
(1) an undecomposed last-layer (final resid_post) probe, using the exact
    same mean-pool + 5-fold CV logistic-regression pipeline as the
    FFN/Attn component probes (code/02), for a fair comparison; and
(2) generation-confidence baselines (mean/min teacher-forced log-prob,
    mean/min per-token max-softmax) over the actual completion tokens.
Both use the validated judge label on GPT-2's full 534-sample set
(results/gpt2_full_534_judge_labels.json), matching the rest of this
paper's validated-label reanalyses.

The cached activations.pkl only stores the PROMPT's forward pass (its
hidden states/logits), not the completion's -- consistent with every
other probe in this paper reading a pre-generation signal. The
generation-confidence baselines need a fresh, cheap (GPT-2-small, no
sampling, teacher-forced) forward pass over prompt+completion to score
the actual completion tokens; this is local, CPU-feasible, no Kaggle
needed.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MECHINT = Path(os.path.expanduser(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int"))) / "data" / "processed"
OUT_PATH = ROOT / "results" / "cheap_baselines.json"


def load_data():
    with open(MECHINT / "activations.pkl", "rb") as f:
        acts = pickle.load(f)
    with open(MECHINT / "labeled.pkl", "rb") as f:
        lab = pickle.load(f)
    with open(ROOT / "results" / "gpt2_full_534_judge_labels.json") as f:
        judge = json.load(f)
    y = np.array(judge["judge_labels"])
    assert len(acts) == len(lab["prompts"]) == len(y) == 534
    return acts, lab, y


def probe_auroc(X, y, n_splits=5):
    actual_splits = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    probe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    from sklearn.metrics import make_scorer
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    return float(cv["test_auroc"].mean()), float(cv["test_auroc"].std())


def last_layer_baseline(acts, y):
    X = np.stack([a["hidden_states"][-1].mean(axis=0) for a in acts])  # [N, 768], mean-pooled final resid_post
    return probe_auroc(X, y)


def generation_confidence_baselines(lab, y):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    mean_logprob, min_logprob, mean_max_softmax, min_max_softmax = [], [], [], []
    with torch.no_grad():
        for i, (prompt, completion) in enumerate(zip(lab["prompts"], lab["completions"])):
            prompt_ids = tok(prompt, return_tensors="pt")["input_ids"]
            completion_ids = tok(completion, return_tensors="pt", truncation=True, max_length=64)["input_ids"]
            if completion_ids.shape[1] == 0:
                mean_logprob.append(0.0); min_logprob.append(0.0)
                mean_max_softmax.append(1.0); min_max_softmax.append(1.0)
                continue
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            out = model(full_ids)
            logits = out.logits[0]  # [seq_len, vocab]
            n_prompt = prompt_ids.shape[1]
            n_comp = completion_ids.shape[1]
            # position t's logits predict token at t+1; completion tokens start at n_prompt
            pred_logits = logits[n_prompt - 1: n_prompt - 1 + n_comp]  # [n_comp, vocab]
            log_probs = log_softmax(pred_logits.numpy(), axis=-1)
            probs = softmax(pred_logits.numpy(), axis=-1)
            actual_ids = completion_ids[0].numpy()
            token_logprobs = log_probs[np.arange(n_comp), actual_ids]
            token_max_softmax = probs.max(axis=-1)

            mean_logprob.append(float(token_logprobs.mean()))
            min_logprob.append(float(token_logprobs.min()))
            mean_max_softmax.append(float(token_max_softmax.mean()))
            min_max_softmax.append(float(token_max_softmax.min()))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/534", flush=True)

    feats = {
        "mean_logprob": np.array(mean_logprob), "min_logprob": np.array(min_logprob),
        "mean_max_softmax": np.array(mean_max_softmax), "min_max_softmax": np.array(min_max_softmax),
    }
    results = {}
    for name, f in feats.items():
        auc_mean, auc_std = probe_auroc(f.reshape(-1, 1), y)
        results[name] = {"auroc_mean": auc_mean, "auroc_std": auc_std}
    X_all = np.stack(list(feats.values()), axis=1)
    auc_mean, auc_std = probe_auroc(X_all, y)
    results["all_4_combined"] = {"auroc_mean": auc_mean, "auroc_std": auc_std}
    return results


def main():
    acts, lab, y = load_data()
    print(f"Loaded {len(y)} samples, hall_rate={1 - y.mean():.3f} (label=1 is 'correct')")

    ll_mean, ll_std = last_layer_baseline(acts, y)
    print(f"Last-layer (undecomposed resid_post) probe: AUROC={ll_mean:.4f} +/- {ll_std:.4f}")

    print("Computing generation-confidence baselines (fresh forward pass, completions)...")
    gen_results = generation_confidence_baselines(lab, y)
    for name, r in gen_results.items():
        print(f"  {name}: AUROC={r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}")

    out = {
        "last_layer_undecomposed_probe": {"auroc_mean": ll_mean, "auroc_std": ll_std},
        "generation_confidence_baselines": gen_results,
        "n_samples": int(len(y)),
        "note": "All probes/baselines use the validated LLM-judge label (gpt2_full_534_judge_labels.json), n=534.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

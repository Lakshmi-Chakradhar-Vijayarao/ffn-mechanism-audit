"""
Paper 1 -- dosage-mismatch diagnostic for the flagship causal-patching test
(code/01_ffn_causal_patch.py, code/10, code/14, and the judge-label rerun in
kaggle_kernels/paper1-causal-patch-judge-label/).

QUESTION
--------
patched_generate() adds `alpha * direction` (direction is unit-norm) to
EITHER the MLP sublayer output OR the Attention sublayer output, using the
SAME alpha for both, at layers 8 and 9 of GPT-2. If MLP and Attention
outputs have different typical norms at these layers, the same alpha is a
different *relative* perturbation for each -- a genuine confound for the
paper's "FFN intervention causes X, Attention intervention doesn't"
component-specificity claim: an apparent asymmetry could just be "the
FFN patch is relatively bigger," not "the FFN locus matters."

This script measures the actual per-token output norm of the MLP and
Attention sublayers at layers 8 and 9, on real TruthfulQA prompts (the
same data source as the causal-patching test), and reports the ratio.
No training, no generation -- one forward pass per prompt, activations
read directly. Cheap, no GPU required beyond CPU-level forward passes on
GPT-2-small.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "ffn_attn_dosage_diagnostic.json"
LAYERS = [8, 9]
ALPHAS = [20.0, 40.0]
N_PROMPTS = 100

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_prompts(n):
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation")["validation"]
        prompts = [ds[i]["question"] for i in range(min(n, len(ds)))]
    except Exception as e:
        print(f"Falling back to synthetic prompts (dataset load failed: {e})")
        prompts = [f"What is the capital of country number {i}?" for i in range(n)]
    return prompts


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    prompts = load_prompts(N_PROMPTS)
    print(f"Measuring output norms on {len(prompts)} prompts, layers {LAYERS}", flush=True)

    norms = {f"L{l}_{comp}": [] for l in LAYERS for comp in ["mlp", "attn"]}

    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(device)
        captured = {}

        def make_hook(key):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[key] = o[0, -1, :].detach()
            return hook

        handles = []
        for l in LAYERS:
            handles.append(model.transformer.h[l].mlp.register_forward_hook(make_hook(f"L{l}_mlp")))
            handles.append(model.transformer.h[l].attn.register_forward_hook(make_hook(f"L{l}_attn")))

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for l in LAYERS:
            norms[f"L{l}_mlp"].append(float(torch.norm(captured[f"L{l}_mlp"]).item()))
            norms[f"L{l}_attn"].append(float(torch.norm(captured[f"L{l}_attn"]).item()))

    out = {"n_prompts": len(prompts), "layers": LAYERS, "alphas": ALPHAS, "results": {}}
    print("\n--- Output-norm comparison (last-token, per layer) ---")
    for l in LAYERS:
        mlp_arr = np.array(norms[f"L{l}_mlp"])
        attn_arr = np.array(norms[f"L{l}_attn"])
        ratio = mlp_arr.mean() / attn_arr.mean()
        print(f"L{l}: MLP output norm mean={mlp_arr.mean():.3f} (std={mlp_arr.std():.3f}), "
              f"Attn output norm mean={attn_arr.mean():.3f} (std={attn_arr.std():.3f}), "
              f"ratio MLP/Attn={ratio:.3f}")
        for alpha in ALPHAS:
            rel_mlp = alpha / mlp_arr.mean()
            rel_attn = alpha / attn_arr.mean()
            print(f"  alpha={alpha}: relative perturbation MLP={rel_mlp:.3f} ({rel_mlp*100:.1f}% of typical norm), "
                  f"Attn={rel_attn:.3f} ({rel_attn*100:.1f}% of typical norm), "
                  f"relative-dosage ratio (Attn/MLP)={rel_attn/rel_mlp:.3f}")
        out["results"][f"L{l}"] = {
            "mlp_norm_mean": float(mlp_arr.mean()), "mlp_norm_std": float(mlp_arr.std()),
            "attn_norm_mean": float(attn_arr.mean()), "attn_norm_std": float(attn_arr.std()),
            "ratio_mlp_over_attn": float(ratio),
            "per_alpha": {
                str(alpha): {
                    "relative_perturbation_mlp": float(alpha / mlp_arr.mean()),
                    "relative_perturbation_attn": float(alpha / attn_arr.mean()),
                } for alpha in ALPHAS
            },
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

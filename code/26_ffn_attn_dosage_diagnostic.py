"""
Paper 1 -- dosage-mismatch diagnostic for the flagship causal-patching test
(code/01_ffn_causal_patch.py and successors).

CORRECTED VERSION (v2): a fresh review caught two errors in the original
version of this diagnostic:
1. It measured on BARE TruthfulQA questions (dataset["question"]), not
   the "Q: {question}\nA:" formatted prompts patched_generate() actually
   uses -- a different, shorter input than the real experiment.
2. It normalized alpha against each SUBLAYER'S OWN output norm (the MLP
   module's or Attention module's own forward-output norm). This is the
   wrong denominator: patched_generate()'s hook REPLACES that sublayer's
   own output with (out + alpha*direction), and this combined value is
   then added into the RESIDUAL STREAM by the transformer block's own
   forward code. The quantity the intervention actually has to compete
   with, at the point it enters the computation, is the residual
   stream's own norm at that layer -- not the sublayer's own output norm,
   which the intervention never interacts with directly. Because both
   the FFN-site and Attention-site hooks add alpha*direction into
   essentially the same accumulated residual stream (the difference in
   accumulated norm from one attention-sublayer addition, at layer 8-9
   after 8-9 layers of accumulation, is small), the corrected, honest
   comparison is APPROXIMATELY THE SAME relative dosage for both arms --
   not the 2-3x asymmetry the original (wrong-denominator) version
   reported.

This script measures both the (corrected) residual-stream norm and, for
transparency, retains the original (incorrect) per-sublayer-output-norm
measurement side by side, on the CORRECT "Q: ... A:" prompts.
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
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation")["validation"]
        prompts = [f"Q: {ds[i]['question']}\nA:" for i in range(min(n, len(ds)))]
    except Exception as e:
        print(f"Falling back to synthetic prompts (dataset load failed: {e})")
        prompts = [f"Q: What is the capital of country number {i}?\nA:" for i in range(n)]
    return prompts


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    prompts = load_prompts(N_PROMPTS)
    print(f"Measuring output norms on {len(prompts)} prompts (Q:...A: format), layers {LAYERS}", flush=True)

    norms = {f"L{l}_{comp}": [] for l in LAYERS for comp in ["mlp", "attn", "residual"]}

    for p in prompts:
        inputs = tok(p, return_tensors="pt", truncation=True, max_length=128).to(device)
        captured = {}

        def make_sublayer_hook(key):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[key] = o[0, -1, :].detach()
            return hook

        handles = []
        for l in LAYERS:
            handles.append(model.transformer.h[l].mlp.register_forward_hook(make_sublayer_hook(f"L{l}_mlp")))
            handles.append(model.transformer.h[l].attn.register_forward_hook(make_sublayer_hook(f"L{l}_attn")))

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for h in handles:
            h.remove()

        # hidden_states[l+1] = the residual stream AFTER block l's full
        # attn+mlp update -- the accumulated stream this intervention
        # actually has to compete with at the point it's injected.
        for l in LAYERS:
            residual_vec = out.hidden_states[l + 1][0, -1, :].detach()
            norms[f"L{l}_mlp"].append(float(torch.norm(captured[f"L{l}_mlp"]).item()))
            norms[f"L{l}_attn"].append(float(torch.norm(captured[f"L{l}_attn"]).item()))
            norms[f"L{l}_residual"].append(float(torch.norm(residual_vec).item()))

    out_json = {"n_prompts": len(prompts), "layers": LAYERS, "alphas": ALPHAS, "results": {}}
    print("\n--- Output-norm comparison (last-token, per layer, Q:...A: prompts) ---")
    for l in LAYERS:
        mlp_arr = np.array(norms[f"L{l}_mlp"])
        attn_arr = np.array(norms[f"L{l}_attn"])
        resid_arr = np.array(norms[f"L{l}_residual"])
        ratio_wrong = mlp_arr.mean() / attn_arr.mean()
        print(f"L{l}: [WRONG denominator, sublayer's own output norm] "
              f"MLP={mlp_arr.mean():.3f}, Attn={attn_arr.mean():.3f}, ratio={ratio_wrong:.3f}")
        print(f"L{l}: [CORRECT denominator, residual-stream norm] mean={resid_arr.mean():.3f} (std={resid_arr.std():.3f})")
        entry = {
            "mlp_sublayer_output_norm_mean": float(mlp_arr.mean()),
            "attn_sublayer_output_norm_mean": float(attn_arr.mean()),
            "wrong_denominator_ratio_mlp_over_attn": float(ratio_wrong),
            "residual_stream_norm_mean": float(resid_arr.mean()),
            "residual_stream_norm_std": float(resid_arr.std()),
            "per_alpha": {},
        }
        for alpha in ALPHAS:
            rel_correct = alpha / resid_arr.mean()
            rel_wrong_mlp = alpha / mlp_arr.mean()
            rel_wrong_attn = alpha / attn_arr.mean()
            print(f"  alpha={alpha}: [CORRECT] relative perturbation to residual stream = {rel_correct*100:.1f}% "
                  f"(identical for both arms, since both inject into the same accumulated stream)")
            print(f"  alpha={alpha}: [WRONG] relative-to-own-output: MLP={rel_wrong_mlp*100:.1f}%, Attn={rel_wrong_attn*100:.1f}%")
            entry["per_alpha"][str(alpha)] = {
                "relative_perturbation_to_residual_stream": float(rel_correct),
                "relative_perturbation_wrong_denominator_mlp": float(rel_wrong_mlp),
                "relative_perturbation_wrong_denominator_attn": float(rel_wrong_attn),
            }
        out_json["results"][f"L{l}"] = entry

    with open(OUT_PATH, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- extend the validated-judge-label revalidation (already done for
the causal-patching test, code/24 for Pythia/Qwen0.5B's component probe)
to GPT-2's own Sec 3.1 (layer localization), Sec 3.2 (component
decomposition -- GPT-2 specifically was never re-checked, only the other
three architectures were), and the Sec 3.7 difficulty-matched control.

Reuses mech-int's own probing code as a library (no modification to that
sibling project) against the SAME cached activations.pkl, swapping in the
full 534-sample judge label array built via
kaggle_kernels/paper1-gpt2-full-judge-relabel/ (code/28's local attempt
hung on MPS generation and was abandoned in favor of the Kaggle GPU run,
consistent with how the other three architectures' judge relabeling was
done) in place of the original Jaccard labels, and reruns every method
under both labels for direct comparison. Positional alignment between
activations.pkl and the label arrays was independently verified (decoding
input_ids back to prompt text) before this script was written.

No re-extraction of activations, no GPU needed here -- every method below
operates on already-cached hidden_states/attn_outputs/ffn_outputs, so this
is a CPU-only rerun.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

MECH_INT = Path("/Users/chakrivijayarao/Desktop/mech-int")
sys.path.insert(0, str(MECH_INT / "src"))

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "gpt2_full_validated_relabel_rerun.json"

from probing.layer_probe import probe_all_layers, probe_layer_sparse, probe_token_positions_all_layers
from probing.component_probe import probe_all_layers_components
from analysis.logit_attribution import compute_dla_all, summarise_dla
from model.load_model import load_gpt2


def run_all_probes(activations, labels, label_name):
    print(f"\n{'='*70}\nRunning all Sec 3.1/3.2 probes under {label_name} labels\n{'='*70}", flush=True)
    out = {}

    dense = probe_all_layers(activations, labels)
    out["dense_probe"] = dense
    dense_peak = max(dense, key=lambda r: r["mean_auroc"])
    print(f"[{label_name}] Dense probe peak: L{dense_peak['layer']} AUROC={dense_peak['mean_auroc']:.4f}")

    sparse_peak_layer = dense_peak["layer"]
    sparse = probe_layer_sparse(activations, labels, sparse_peak_layer)
    out["sparse_probe_at_dense_peak"] = sparse
    print(f"[{label_name}] Sparse L1 probe at L{sparse_peak_layer}: "
          f"{sparse['n_nonzero']}/{sparse['hidden_dim']} active dims "
          f"({sparse['sparsity']*100:.0f}% sparse), CV AUROC={sparse['auroc_cv']:.4f}, "
          f"in-sample={sparse['auroc_train']:.4f}")

    token_pos = probe_token_positions_all_layers(activations, labels)
    out["token_position_probe"] = token_pos
    best_last_token = max(token_pos, key=lambda r: r[-1])
    print(f"[{label_name}] Token-position probe, best last-token layer: "
          f"L{best_last_token['layer']} AUROC={best_last_token[-1]:.4f}")

    comp = probe_all_layers_components(activations, labels)
    out["component_probe"] = comp
    ffn_peak = max(comp["ffn"], key=lambda r: r["mean_auroc"])
    attn_peak = max(comp["attn"], key=lambda r: r["mean_auroc"])
    ffn_wins = sum(1 for f, a in zip(comp["ffn"], comp["attn"]) if f["mean_auroc"] >= a["mean_auroc"])
    print(f"[{label_name}] Component probe: FFN peak L{ffn_peak['layer']} AUROC={ffn_peak['mean_auroc']:.4f}, "
          f"Attn peak L{attn_peak['layer']} AUROC={attn_peak['mean_auroc']:.4f}, "
          f"FFN wins {ffn_wins}/{len(comp['ffn'])} layers")

    return out


def run_dla(activations, labels_jaccard, labels_judge):
    print(f"\n{'='*70}\nRunning DLA (label-independent computation, label-dependent summary)\n{'='*70}", flush=True)
    model, tokenizer, device = load_gpt2()
    dla_results = compute_dla_all(activations, model, device)

    summary_jaccard = summarise_dla(dla_results, labels_jaccard.tolist())
    summary_judge = summarise_dla(dla_results, labels_judge.tolist())

    def pack(s):
        return {
            "peak_ffn_diff_layer": int(s["peak_ffn_diff_layer"]),
            "peak_attn_diff_layer": int(s["peak_attn_diff_layer"]),
            "correct_mean_ffn_dla": s["correct_mean_ffn_dla"].tolist(),
            "hallucinated_mean_ffn_dla": s["hallucinated_mean_attn"].tolist() if False else s["hallucinated_mean_ffn"].tolist(),
            "ffn_dla_diff": s["ffn_dla_diff"].tolist(),
        }

    out = {"jaccard": pack(summary_jaccard), "judge": pack(summary_judge)}
    print(f"[jaccard] DLA peak FFN-diff layer: L{out['jaccard']['peak_ffn_diff_layer']}")
    print(f"[judge]   DLA peak FFN-diff layer: L{out['judge']['peak_ffn_diff_layer']}")
    return out


def main():
    print("Loading activations.pkl (mech-int, 3GB)...", flush=True)
    with open(MECH_INT / "data" / "processed" / "activations.pkl", "rb") as f:
        activations = pickle.load(f)
    print(f"Loaded {len(activations)} samples", flush=True)

    with open(MECH_INT / "data" / "processed" / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    labels_jaccard = np.array(labeled["labels"])

    judge_path = ROOT / "results" / "gpt2_full_534_judge_labels.json"
    with open(judge_path) as f:
        judge_data = json.load(f)
    labels_judge_raw = np.array(judge_data["judge_labels"])
    valid_mask = labels_judge_raw != -1
    print(f"Judge labels: n_valid={valid_mask.sum()}/{len(labels_judge_raw)}, "
          f"kappa_vs_jaccard={judge_data['cohen_kappa']:.4f}", flush=True)

    activations_valid = [a for a, v in zip(activations, valid_mask) if v]
    labels_jaccard_valid = labels_jaccard[valid_mask]
    labels_judge_valid = labels_judge_raw[valid_mask]

    out = {"n_valid": int(valid_mask.sum()), "cohen_kappa": judge_data["cohen_kappa"]}
    out["under_jaccard"] = run_all_probes(activations_valid, labels_jaccard_valid.tolist(), "JACCARD (original)")
    out["under_judge"] = run_all_probes(activations_valid, labels_judge_valid.tolist(), "JUDGE (validated)")
    out["dla"] = run_dla(activations_valid, labels_jaccard_valid, labels_judge_valid)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- final-audit fix. §3.1's "7 converging methods" GPT-2 numbers
(dense probe 0.5827, sparse probe CV AUROC 0.589, token-position 0.6036,
component probe FFN 0.6053 / Attn 0.6165, DLA 5.08/4.85 at L8 and the
L10/L11 DLA-correction figures) previously had no backing artifact
anywhere in this repo -- they were computed in a separate, unshipped
sibling project (mech-int) and only quoted as numbers in the draft.

This script does not recompute anything from raw activations (the
underlying 2.9GB activations.pkl is not vendored here, disclosed below);
it loads the small (<10KB each) per-method result summaries vendored
into results/vendored_mech_int/ (copied verbatim from mech-int's own
results/logs/*.npy) and reprints exactly the numbers §3.1 and §3.2 cite,
so a reader can verify them against a real file in this repo without
needing the sibling project at all.

NOT vendored (too large for this repo, ~2.9GB): mech-int's
data/processed/activations.pkl, the raw per-sample hidden-state
activations these summaries were originally computed from. Anyone
wanting to recompute these summaries from scratch (rather than verify
the already-computed values) needs the mech-int project itself.
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "vendored_mech_int"


def main():
    dense = np.load(DATA_DIR / "layer_probe_results.npy", allow_pickle=True)
    dense_best = max(dense, key=lambda r: r["mean_auroc"])
    print(f"Dense probe peak: L{dense_best['layer']} "
          f"AUROC={dense_best['mean_auroc']:.4f}  (paper: L9, 0.5827)")

    sparse = np.load(DATA_DIR / "sparse_probe_results.npy", allow_pickle=True).item()
    print(f"Sparse L1 probe peak: L{sparse['layer']} "
          f"{sparse['n_nonzero']}/{sparse['hidden_dim']} active dims "
          f"({sparse['sparsity']*100:.0f}% sparse) "
          f"CV AUROC={sparse['auroc_cv']:.4f}, in-sample={sparse['auroc_train']:.4f}  "
          f"(paper: L9, 100/768, 87% sparse, CV 0.589, in-sample 0.874)")

    tok = np.load(DATA_DIR / "token_position_results.npy", allow_pickle=True)
    tok_best = max(tok, key=lambda r: r[-1])
    print(f"Token-position (last-token) peak: L{tok_best['layer']} "
          f"AUROC={tok_best[-1]:.4f}  (paper: L8, 0.6036)")

    comp = np.load(DATA_DIR / "component_results.npy", allow_pickle=True).item()
    ffn_best = max(comp["ffn"], key=lambda r: r["mean_auroc"])
    attn_best = max(comp["attn"], key=lambda r: r["mean_auroc"])
    print(f"Component probe FFN peak: L{ffn_best['layer']} AUROC={ffn_best['mean_auroc']:.4f}  "
          f"(paper: L8, 0.6053)")
    print(f"Component probe Attn peak: L{attn_best['layer']} AUROC={attn_best['mean_auroc']:.4f}  "
          f"(paper: L3, 0.6165)")

    dla = np.load(DATA_DIR / "dla_results.npy", allow_pickle=True).item()
    l8_hall = dla["hallucinated_mean_ffn"][8]
    l8_correct = dla["correct_mean_ffn_dla"][8]
    print(f"FFN DLA at L8: hallucinated={l8_hall:.3f}, correct={l8_correct:.3f}  "
          f"(paper: 5.08, 4.85)")
    print(f"FFN DLA diff, L10={dla['ffn_dla_diff'][10]:+.2f}, L11={dla['ffn_dla_diff'][11]:+.2f}  "
          f"(paper's DLA correction: L10 +0.90, L11 +0.71, both larger than L9)")

    print("\nAll values reproduced from results/vendored_mech_int/*.npy, "
          "no external repo required.")


if __name__ == "__main__":
    main()

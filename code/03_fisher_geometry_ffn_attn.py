"""
Paper 1 — Fisher-geometric grounding of the FFN-vs-Attention component probe.

WHY THIS EXISTS
---------------
GEOM-PROOF (a separate, unpublished project in this arc) derived a closed-
form relationship between a Fisher discriminant ratio J and linear-probe
AUROC (AUROC ~ Phi(sqrt(J)/2) -- itself classical signal-detection theory,
see Simpson & Fitter 1973, not a new result) and found, applying it across
model families, that WHICH divergence measure best predicts a representation's
actual separability is architecture-dependent -- Fisher wins on GPT-2-Medium,
Sliced-Wasserstein/Bures wins on Qwen 2.5 3B, with the ranking fully
reversing between them. That crossover was computed on whole hidden states.

Paper 1 decomposes the residual stream into FFN and Attention sublayer
outputs and shows FFN dominates as a hallucination-detection signal. This
script asks the natural follow-up question neither project asked alone:
does the FFN-vs-Attention AUROC gap correspond to a genuine gap in Fisher
separability between the two components, and does the divergence-family
crossover GEOM-PROOF found (architecture-dependent, not universal) also
show up when comparing FFN geometry against Attention geometry within a
single architecture? This directly reuses GEOM-PROOF's fisher_ratio /
auroc_bound implementation (src/fisher.py, Ledoit-Wolf-shrinkage Fisher
ratio + Gaussian AUROC bound) rather than reimplementing it.

For GPT-2, this runs immediately against already-extracted activations
(mech-int/data/processed/activations.pkl) -- no new model inference needed.
For Pythia/Qwen2.5-0.5B, 02_cross_arch_component_probe.py needs to be
extended to also persist raw per-sample component vectors (not just
per-layer AUROC summaries) before this script can run on them; that
extension is a separate, explicit follow-up, not done implicitly here.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
# Fixed post-final-audit: both were hardcoded to personal absolute paths
# to two genuine external sibling projects. GEOM_PROOF_ROOT has no
# vendorable substitute (imports live code, not data) -- env var
# override added. MECH_INT_ROOT's only use in this file
# (results/logs/component_results.npy, checked further below) is
# already vendored into this repo; the external path remains as a
# fallback for anything not vendored (e.g. the 2.9GB activations.pkl
# this file's docstring describes but does not itself load).
GEOM_PROOF_ROOT = Path(os.environ.get("GEOM_PROOF_ROOT", "~/Desktop/geom-proof")).expanduser()
sys.path.insert(0, str(GEOM_PROOF_ROOT))
from src.fisher import fisher_ratio, auroc_bound  # noqa: E402

MECH_INT_ROOT = Path(os.environ.get("MECH_INT_ROOT", "~/Desktop/mech-int")).expanduser()
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gpt2_component_data():
    labeled_path = VENDORED / "labeled.pkl"
    if not labeled_path.exists():
        labeled_path = MECH_INT_ROOT / "data/processed/labeled.pkl"
    with open(labeled_path, "rb") as f:
        labeled = pickle.load(f)
    labels = np.array(labeled["labels"])

    # activations.pkl (2.9GB) is not vendored into this repo -- genuine
    # external dependency on the mech-int sibling project, disclosed in
    # this file's module docstring.
    print("Loading GPT-2 activations.pkl (3GB, this takes a moment)...")
    with open(MECH_INT_ROOT / "data/processed/activations.pkl", "rb") as f:
        activations = pickle.load(f)
    print(f"Loaded {len(activations)} samples.")

    num_layers = activations[0]["ffn_outputs"].shape[0]
    ffn_mean_pooled = np.stack([
        np.stack([act["ffn_outputs"][l].mean(axis=0) for l in range(num_layers)])
        for act in activations
    ])  # [N, L, D]
    attn_mean_pooled = np.stack([
        np.stack([act["attn_outputs"][l].mean(axis=0) for l in range(num_layers)])
        for act in activations
    ])  # [N, L, D]
    return ffn_mean_pooled, attn_mean_pooled, labels, num_layers


def fisher_geometry_curve(X, y, method="pca", n_components=100):
    """Per-layer Fisher J + AUROC bound, reusing GEOM-PROOF's exact implementation."""
    num_layers = X.shape[1]
    J_per_layer, bound_per_layer = [], []
    for l in range(num_layers):
        J = fisher_ratio(X[:, l, :], y, method=method, n_components=n_components)
        J_per_layer.append(J)
        bound_per_layer.append(auroc_bound(J))
    return J_per_layer, bound_per_layer


def main():
    ffn, attn, labels, num_layers = load_gpt2_component_data()

    print(f"\nComputing Fisher J per layer for FFN outputs ({num_layers} layers)...")
    ffn_J, ffn_bound = fisher_geometry_curve(ffn, labels)
    print(f"Computing Fisher J per layer for Attention outputs ({num_layers} layers)...")
    attn_J, attn_bound = fisher_geometry_curve(attn, labels)

    print(f"\n{'Layer':<8}{'FFN J':>10}{'FFN bound':>12}{'Attn J':>10}{'Attn bound':>12}{'Geom winner':>14}")
    print("-" * 66)
    ffn_geom_wins = 0
    for l in range(num_layers):
        winner = "FFN" if ffn_J[l] >= attn_J[l] else "Attn"
        ffn_geom_wins += int(winner == "FFN")
        print(f"L{l:<7}{ffn_J[l]:>10.4f}{ffn_bound[l]:>12.4f}{attn_J[l]:>10.4f}{attn_bound[l]:>12.4f}{winner:>14}")

    peak_ffn_layer = int(np.argmax(ffn_J))
    peak_attn_layer = int(np.argmax(attn_J))
    print(f"\nFFN geometric peak: L{peak_ffn_layer} (J={ffn_J[peak_ffn_layer]:.4f}, "
          f"bound={ffn_bound[peak_ffn_layer]:.4f})")
    print(f"Attn geometric peak: L{peak_attn_layer} (J={attn_J[peak_attn_layer]:.4f}, "
          f"bound={attn_bound[peak_attn_layer]:.4f})")
    print(f"FFN geometrically dominates in {ffn_geom_wins}/{num_layers} layers")

    # Cross-check against the already-known empirical component-probe AUROC
    # (component_probe.py's results, if present) -- does the Fisher-bound
    # curve track the actual cross-validated AUROC curve?
    comp_probe_path = VENDORED / "component_results.npy"
    if not comp_probe_path.exists():
        comp_probe_path = MECH_INT_ROOT / "results/logs/component_results.npy"
    correlation_note = None
    if comp_probe_path.exists():
        comp_results = np.load(comp_probe_path, allow_pickle=True).item()
        ffn_auroc_actual = [r["mean_auroc"] for r in comp_results["ffn"]]
        attn_auroc_actual = [r["mean_auroc"] for r in comp_results["attn"]]
        ffn_corr = float(np.corrcoef(ffn_bound, ffn_auroc_actual)[0, 1])
        attn_corr = float(np.corrcoef(attn_bound, attn_auroc_actual)[0, 1])
        correlation_note = {"ffn_bound_vs_actual_auroc_pearson": round(ffn_corr, 4),
                             "attn_bound_vs_actual_auroc_pearson": round(attn_corr, 4)}
        print(f"\nFisher-bound vs. actual CV-AUROC correlation: "
              f"FFN r={ffn_corr:.3f}  Attn r={attn_corr:.3f}")

    out = {
        "model": "gpt2",
        "num_layers": num_layers,
        "ffn_fisher_J": ffn_J,
        "ffn_auroc_bound": ffn_bound,
        "attn_fisher_J": attn_J,
        "attn_auroc_bound": attn_bound,
        "peak_ffn_layer": peak_ffn_layer,
        "peak_attn_layer": peak_attn_layer,
        "ffn_geometric_wins": ffn_geom_wins,
        "correlation_with_empirical_auroc": correlation_note,
        "method_note": "Fisher ratio via GEOM-PROOF src/fisher.py (Ledoit-Wolf shrinkage, "
                       "PCA-100 pre-reduction); AUROC bound = Phi(sqrt(J)/2), classical "
                       "signal-detection identity (Simpson & Fitter 1973), reused here as "
                       "an analytical lens on the FFN-vs-Attn component decomposition, not "
                       "presented as new theory.",
    }
    out_path = OUT_DIR / "fisher_geometry_ffn_attn_gpt2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- fixes a statistical gap a fresh review found: the paper's
central quantity (FFN AUROC minus Attention AUROC) is never directly
tested anywhere. Every comparison in Sec 3.2/3.3 is a visual comparison
of two SEPARATELY-estimated CV means against their own fold SDs, even
though FFN and Attention probes are fit on the SAME samples and SAME
folds -- they are paired, and the correct instrument is a paired
bootstrap / test on the per-sample difference, not two independent
intervals.

This script:
  1. Extracts GPT-2's per-layer, mean-pooled FFN/Attn features directly
     from the vendored mech-int activations (attn_outputs, ffn_outputs
     already computed and cached -- no new model inference), matching
     Pythia/Qwen's already-cached results/cross_arch_raw_features_*.npz
     format exactly.
  2. For every layer on every architecture, fits the SAME 5-fold-CV
     logistic-regression probe (identical to code/02's probe_component_at_layer)
     on both FFN and Attn features using the SAME folds, and computes a
     BCa bootstrap 95% CI on the paired AUROC gap.
  3. Nested-CV selection-adjusted peak: selects the peak layer using
     inner folds only, evaluates on the outer fold -- reported alongside
     the naive argmax, to check the paper's peak-AUROC comparisons for
     winner's-curse-style selection bias.
  4. A mixed-effects-style pooled estimate across architectures/layers
     (implemented via a simple random-intercept GEE-equivalent: weighted
     average of per-architecture mean deltas with between-architecture
     variance added to the CI, since statsmodels MixedLM's LR-index
     random effect isn't a natural fit for a small, irregular per-arch
     layer count -- documented explicitly rather than mis-applying a
     tool).
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "paired_component_delta_auroc.json"
RNG = np.random.default_rng(2026)
N_BOOT = 2000


def extract_gpt2_features():
    """Mean-pool attn_outputs/ffn_outputs over sequence length per layer,
    matching code/02_cross_arch_component_probe.py's extract_components()
    pooling convention exactly (mean over token positions)."""
    mech_int_root = Path("~/Desktop/mech-int").expanduser()
    acts_path = mech_int_root / "data" / "processed" / "activations.pkl"
    if not acts_path.exists():
        return None
    with open(acts_path, "rb") as f:
        records = pickle.load(f)
    with open(ROOT / "results" / "vendored_mech_int" / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    labels = np.array(labeled["labels"])
    assert len(records) == len(labels), f"{len(records)} vs {len(labels)}"

    n = len(records)
    n_layers = records[0]["ffn_outputs"].shape[0]
    hidden = records[0]["ffn_outputs"].shape[-1]
    ffn = np.zeros((n, n_layers, hidden), dtype=np.float32)
    attn = np.zeros((n, n_layers, hidden), dtype=np.float32)
    for i, r in enumerate(records):
        ffn[i] = r["ffn_outputs"].mean(axis=1)   # mean over seq_len
        attn[i] = r["attn_outputs"].mean(axis=1)
    return {"ffn": ffn, "attn": attn, "labels": labels}


def probe_layer_oof(X_layer, y, seed=0, n_splits=5):
    """5-fold CV logistic regression, out-of-fold predicted probabilities,
    identical protocol to code/02's probe_component_at_layer."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # The scaler MUST live inside the pipeline so it is refit per fold. An
    # earlier version called scaler.fit_transform on the full X_layer before
    # cross_val_predict, which leaks each test fold's mean/variance into its
    # own training fold. code/02, code/47, code/49 and code/50 all use the
    # pipeline form; this script was the one exception.
    probe = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))])
    probs = cross_val_predict(probe, X_layer, y, cv=skf, method="predict_proba")[:, 1]
    return probs


def bca_ci_on_delta(probs_ffn, probs_attn, y):
    """BCa bootstrap CI on AUROC(ffn) - AUROC(attn), resampling SAMPLES
    (preserves the pairing, since both probs arrays are indexed identically)."""
    y = np.asarray(y)

    def delta_stat(idx):
        idx = idx.astype(int)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return np.nan
        return roc_auc_score(yy, probs_ffn[idx]) - roc_auc_score(yy, probs_attn[idx])

    idx_arr = np.arange(len(y))
    try:
        res = bootstrap((idx_arr,), delta_stat, n_resamples=N_BOOT, method="BCa",
                         random_state=RNG, confidence_level=0.95)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as e:
        return None, None


def nested_cv_peak(X, y, n_outer=5, n_inner=5, seed=0):
    """Select peak layer on inner folds, evaluate on the held-out outer
    fold. Returns mean nested-peak AUROC (comparable to the naive argmax)."""
    n_layers = X.shape[1]
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    outer_aucs = []
    for tr_idx, te_idx in outer.split(X[:, 0, :], y):
        y_tr = y[tr_idx]
        inner_layer_aucs = np.zeros(n_layers)
        for layer in range(n_layers):
            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            probe = Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000))])
            try:
                probs = cross_val_predict(probe, X[tr_idx, layer, :], y_tr, cv=inner, method="predict_proba")[:, 1]
                inner_layer_aucs[layer] = roc_auc_score(y_tr, probs)
            except ValueError:
                inner_layer_aucs[layer] = 0.5
        selected_layer = int(np.argmax(inner_layer_aucs))

        probe = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000))])
        probe.fit(X[tr_idx, selected_layer, :], y_tr)
        auc = roc_auc_score(y[te_idx], probe.predict_proba(X[te_idx, selected_layer, :])[:, 1])
        outer_aucs.append(auc)
    return float(np.mean(outer_aucs)), float(np.std(outer_aucs))


def analyze_architecture(name, X_ffn, X_attn, y):
    n_layers = X_ffn.shape[1]
    print(f"\n=== {name} ({n_layers} layers, n={len(y)}) ===", flush=True)
    layer_results = []
    for layer in range(n_layers):
        probs_ffn = probe_layer_oof(X_ffn[:, layer, :], y)
        probs_attn = probe_layer_oof(X_attn[:, layer, :], y)
        auc_ffn = roc_auc_score(y, probs_ffn)
        auc_attn = roc_auc_score(y, probs_attn)
        delta = auc_ffn - auc_attn
        ci_low, ci_high = bca_ci_on_delta(probs_ffn, probs_attn, y)
        layer_results.append({
            "layer": layer, "auroc_ffn": float(auc_ffn), "auroc_attn": float(auc_attn),
            "delta": float(delta), "bca_ci_95": [ci_low, ci_high],
        })

    peak_ffn_layer = int(np.argmax([r["auroc_ffn"] for r in layer_results]))
    peak_attn_layer = int(np.argmax([r["auroc_attn"] for r in layer_results]))
    print(f"  naive peak FFN: L{peak_ffn_layer} AUROC={layer_results[peak_ffn_layer]['auroc_ffn']:.4f}")
    print(f"  naive peak Attn: L{peak_attn_layer} AUROC={layer_results[peak_attn_layer]['auroc_attn']:.4f}")

    nested_ffn_mean, nested_ffn_sd = nested_cv_peak(X_ffn, y)
    nested_attn_mean, nested_attn_sd = nested_cv_peak(X_attn, y)
    print(f"  nested-CV selection-adjusted FFN peak: {nested_ffn_mean:.4f}+/-{nested_ffn_sd:.4f}")
    print(f"  nested-CV selection-adjusted Attn peak: {nested_attn_mean:.4f}+/-{nested_attn_sd:.4f}")

    deltas = [r["delta"] for r in layer_results]
    return {
        "n_layers": n_layers, "n_samples": len(y),
        "layer_results": layer_results,
        "naive_peak_ffn": {"layer": peak_ffn_layer, "auroc": layer_results[peak_ffn_layer]["auroc_ffn"]},
        "naive_peak_attn": {"layer": peak_attn_layer, "auroc": layer_results[peak_attn_layer]["auroc_attn"]},
        "nested_cv_peak_ffn": {"mean": nested_ffn_mean, "std": nested_ffn_sd},
        "nested_cv_peak_attn": {"mean": nested_attn_mean, "std": nested_attn_sd},
        "mean_delta_across_layers": float(np.mean(deltas)),
        "std_delta_across_layers": float(np.std(deltas)),
    }


def main():
    out = {}

    print("Extracting GPT-2 features from vendored mech-int activations...", flush=True)
    gpt2_data = extract_gpt2_features()
    if gpt2_data is not None:
        out["gpt2"] = analyze_architecture("GPT-2", gpt2_data["ffn"], gpt2_data["attn"], gpt2_data["labels"])
    else:
        print("  mech-int activations not found, skipping GPT-2", flush=True)

    print("\nLoading cached Pythia features...", flush=True)
    d = np.load(ROOT / "results" / "cross_arch_raw_features_pythia.npz")
    out["pythia"] = analyze_architecture("Pythia-410M", d["ffn"], d["attn"], d["labels"])

    print("\nLoading cached Qwen0.5B features...", flush=True)
    d = np.load(ROOT / "results" / "cross_arch_raw_features_qwen05.npz")
    out["qwen05"] = analyze_architecture("Qwen2.5-0.5B (bare)", d["ffn"], d["attn"], d["labels"])

    # Pooled estimate across architectures: weighted mean of per-architecture
    # mean deltas (weighted by n_layers), with between-architecture variance
    # added explicitly rather than assuming a MixedLM random-intercept model
    # fits this small, irregular-layer-count setting well.
    arch_means = [out[a]["mean_delta_across_layers"] for a in out]
    arch_weights = [out[a]["n_layers"] for a in out]
    pooled_mean = float(np.average(arch_means, weights=arch_weights))
    between_arch_var = float(np.var(arch_means, ddof=1)) if len(arch_means) > 1 else 0.0
    out["pooled_across_architectures"] = {
        "weighted_mean_delta": pooled_mean,
        "between_architecture_variance": between_arch_var,
        "per_architecture_means": {a: out[a]["mean_delta_across_layers"] for a in out if a != "pooled_across_architectures"},
        "note": "Weighted mean of per-architecture mean deltas (weighted by n_layers), "
                "with between-architecture variance reported explicitly rather than fit "
                "via MixedLM, given the small (3) and irregular-layer-count architecture set.",
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"\nPooled weighted mean delta (FFN-Attn) across architectures: {pooled_mean:+.4f} "
          f"(between-arch var={between_arch_var:.6f})")


if __name__ == "__main__":
    main()

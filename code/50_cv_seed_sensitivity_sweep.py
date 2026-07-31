"""
Paper 1 -- how much of this paper's FFN-vs-Attention evidence is a
cross-validation fold-seed artifact?

An earlier draft attributed the gap between two AUROCs this repo reports
for the SAME features (GPT-2 FFN L8: 0.6053 in Sec 3.2, 0.643 in
code/37's paired-delta analysis) to an aggregation-convention
difference (mean-of-folds vs. pooled out-of-fold predictions). That
explanation is wrong. code/37 uses StratifiedKFold(random_state=0);
code/02_cross_arch_component_probe.py:205 uses random_state=42. This
script decomposes the gap and characterizes the resulting sensitivity:

  1. Exact decomposition of the 0.6053 vs 0.643 gap into a seed
     component and an aggregation component.
  2. A 50-seed sweep of the CV fold assignment (code/02's protocol,
     unchanged except for StratifiedKFold's random_state) at GPT-2
     L8/L9 for both components, reporting mean, SD, and range -- the
     scale against which every FFN-vs-Attention margin in this paper
     (0.003-0.011) must be read.
  3. The same sweep applied to the FFN-minus-Attention DIFFERENCE at
     each layer, which is the paper's actual estimand and is paired
     (same folds for both components), so it is less seed-sensitive
     than either component alone -- reported so the seed-sensitivity
     result is not overstated.
  4. Peak-layer (argmax) stability across seeds for all three
     architectures, which is what the "naming collision" between
     code/02's and code/37's reported peak layers for Qwen0.5B actually
     reflects.

No new model inference: GPT-2 uses the cached mean-pooled features from
code/49 (derived from the vendored mech-int activations); Pythia and
Qwen use results/cross_arch_raw_features_*.npz.
"""
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "cv_seed_sensitivity_sweep.json"
N_SEEDS = 50
SEEDS = list(range(N_SEEDS))


def make_probe():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])


def mean_of_folds(X, y, seed, n_splits=5):
    actual = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=seed)
    cv = cross_validate(make_probe(), X, y, cv=skf,
                        scoring={"auroc": make_scorer(roc_auc_score, response_method="predict_proba")})
    return float(cv["test_auroc"].mean())


def pooled_oof(X, y, seed, n_splits=5):
    actual = max(2, min(n_splits, int(y.sum()), int((y == 0).sum())))
    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=seed)
    proba = cross_val_predict(make_probe(), X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def load_arch(name):
    if name == "gpt2":
        z = np.load(ROOT / "results" / "gpt2_meanpool.npz")
        return z["ffn"], z["attn"], z["labels"]
    z = np.load(ROOT / "results" / f"cross_arch_raw_features_{name}.npz")
    keys = set(z.files)
    ffn_key = "ffn" if "ffn" in keys else [k for k in keys if "ffn" in k][0]
    attn_key = "attn" if "attn" in keys else [k for k in keys if "attn" in k][0]
    lab_key = "labels" if "labels" in keys else [k for k in keys if "label" in k][0]
    return z[ffn_key], z[attn_key], z[lab_key]


def main():
    out = {"n_seeds": N_SEEDS, "seeds": SEEDS}

    # ---- 1. exact decomposition of the 0.6053 vs 0.643 gap (GPT-2 FFN L8)
    ffn, attn, y = load_arch("gpt2")
    X = ffn[:, 8, :]
    a = mean_of_folds(X, y, 42)   # code/02's number
    b = mean_of_folds(X, y, 0)    # seed changed only
    c = pooled_oof(X, y, 0)       # code/37's number
    d = pooled_oof(X, y, 42)
    out["gap_decomposition_gpt2_ffn_L8"] = {
        "code02_mean_of_folds_seed42": a,
        "mean_of_folds_seed0": b,
        "code37_pooled_oof_seed0": c,
        "pooled_oof_seed42": d,
        "total_gap_code37_minus_code02": c - a,
        "seed_component_same_aggregation": b - a,
        "aggregation_component_same_seed0": c - b,
        "aggregation_component_same_seed42": d - a,
    }
    print("GPT-2 FFN L8 gap decomposition:", flush=True)
    print(f"  code/02 mean-of-folds seed42 = {a:.4f}")
    print(f"  mean-of-folds seed0          = {b:.4f}   (seed component: {b-a:+.4f})")
    print(f"  code/37 pooled-OOF seed0     = {c:.4f}   (aggregation component: {c-b:+.4f})")
    print(f"  total gap                    = {c-a:+.4f}", flush=True)

    # ---- 2/3. seed sweep at GPT-2 L8/L9, per component and on the paired delta
    arch_layers = {"gpt2": [8, 9], "pythia": [11, 4], "qwen05": [8, 17]}
    out["seed_sweep"] = {}
    for arch, layers in arch_layers.items():
        try:
            F, A, yy = load_arch(arch)
        except Exception as e:  # noqa: BLE001
            print(f"skip {arch}: {e}", flush=True)
            continue
        entry = {}
        for li in layers:
            per = {}
            for comp, arr in (("ffn", F), ("attn", A)):
                vals = np.array(Parallel(n_jobs=6)(
                    delayed(mean_of_folds)(arr[:, li, :], yy, s) for s in SEEDS))
                per[comp] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                             "min": float(vals.min()), "max": float(vals.max()),
                             "range": float(vals.max() - vals.min()),
                             "at_seed42": float(vals[SEEDS.index(42)]) if 42 in SEEDS else None}
                per[f"{comp}_values"] = [round(float(v), 5) for v in vals]
            dv = np.array(per["ffn_values"]) - np.array(per["attn_values"])
            per["delta_ffn_minus_attn"] = {
                "mean": float(dv.mean()), "sd": float(dv.std(ddof=1)),
                "min": float(dv.min()), "max": float(dv.max()),
                "frac_seeds_ffn_ahead": float((dv > 0).mean()),
            }
            entry[f"L{li}"] = per
            print(f"{arch} L{li}: FFN {per['ffn']['mean']:.4f}+/-{per['ffn']['sd']:.4f} "
                  f"[{per['ffn']['min']:.4f},{per['ffn']['max']:.4f}] | "
                  f"Attn {per['attn']['mean']:.4f}+/-{per['attn']['sd']:.4f} | "
                  f"delta {per['delta_ffn_minus_attn']['mean']:+.4f}"
                  f"+/-{per['delta_ffn_minus_attn']['sd']:.4f}, "
                  f"FFN ahead in {per['delta_ffn_minus_attn']['frac_seeds_ffn_ahead']*100:.0f}% of seeds",
                  flush=True)
        out["seed_sweep"][arch] = entry

    # ---- 4. peak-layer (argmax) stability across seeds
    out["peak_layer_stability"] = {}
    for arch in ("gpt2", "pythia", "qwen05"):
        try:
            F, A, yy = load_arch(arch)
        except Exception:  # noqa: BLE001
            continue
        n_layers = F.shape[1]
        sub = SEEDS[:12]  # 12 seeds x n_layers x 2 components
        peaks = {"ffn": [], "attn": []}
        for comp, arr in (("ffn", F), ("attn", A)):
            flat = Parallel(n_jobs=6, verbose=5)(
                delayed(mean_of_folds)(arr[:, li, :], yy, s) for s in sub for li in range(n_layers))
            for i in range(len(sub)):
                peaks[comp].append(int(np.argmax(flat[i * n_layers:(i + 1) * n_layers])))
        out["peak_layer_stability"][arch] = {
            "n_seeds": len(sub), "n_layers": int(n_layers),
            "ffn_peak_layers": peaks["ffn"], "attn_peak_layers": peaks["attn"],
            "ffn_n_distinct_peaks": len(set(peaks["ffn"])),
            "attn_n_distinct_peaks": len(set(peaks["attn"])),
            "ffn_modal_peak": int(max(set(peaks["ffn"]), key=peaks["ffn"].count)),
            "attn_modal_peak": int(max(set(peaks["attn"]), key=peaks["attn"].count)),
            "ffn_modal_peak_frac": float(peaks["ffn"].count(max(set(peaks["ffn"]), key=peaks["ffn"].count)) / len(sub)),
            "attn_modal_peak_frac": float(peaks["attn"].count(max(set(peaks["attn"]), key=peaks["attn"].count)) / len(sub)),
        }
        st = out["peak_layer_stability"][arch]
        print(f"{arch} peak-layer stability over {len(sub)} seeds: FFN {st['ffn_n_distinct_peaks']} distinct "
              f"(modal L{st['ffn_modal_peak']} in {st['ffn_modal_peak_frac']*100:.0f}%), "
              f"Attn {st['attn_n_distinct_peaks']} distinct "
              f"(modal L{st['attn_modal_peak']} in {st['attn_modal_peak_frac']*100:.0f}%)", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

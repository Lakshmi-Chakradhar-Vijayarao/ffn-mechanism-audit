"""
Paper 1 -- does the passive FFN/Attention component probe's above-chance
AUROC survive removing every prompt whose baseline completion is a
degenerate repetition loop?

Motivation. code/04_degeneration_check.py and
code/14_causal_patch_scaled_degeneration_filter.py established that
51.9%-53.1% of nominally "hallucinated" GPT-2 completions on this pool
are repetition loops rather than confabulations. That is a
construct-validity problem for the causal-patching flip-rate metric --
but it is also a live alternative explanation for the PASSIVE probe: if
the probe is really detecting "this prompt will send GPT-2 into a loop"
rather than "this prompt will produce a false claim," its AUROC should
collapse once degenerate items are removed from both classes.

This script applies the identical is_repetitive() criterion (repeated
4/5/6/8-word phrase occurring 3+ times) to all 534 cached baseline
completions, then reruns code/02_cross_arch_component_probe.py's exact
probe protocol (5-fold stratified CV, standardized logistic regression,
C=1.0) on the non-degenerate subset only, at every layer, for both
components, under BOTH labels (Jaccard and the validated judge label).

Because code/50_cv_seed_sensitivity_sweep.py shows a single CV seed
moves GPT-2's AUROC by more than several effects argued about in this
paper, every number here is reported both at code/02's own seed (42) and
as a mean +/- SD over 6 additional seeds.

Features: per-layer mean-pooled FFN/Attn sublayer outputs from the
vendored mech-int activations, identical to
code/37_paired_component_delta_auroc.py::extract_gpt2_features()
(cached to results/gpt2_meanpool.npz on first run).
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "results" / "vendored_mech_int"
CACHE = ROOT / "results" / "gpt2_meanpool.npz"
JUDGE_LABELS_PATH = ROOT / "results" / "gpt2_full_534_judge_labels.json"
OUT_PATH = ROOT / "results" / "nondegenerate_subset_probe.json"

SEEDS = list(range(6))
CODE02_SEED = 42


def is_repetitive(text, min_repeat=3):
    """Byte-identical to code/04 and code/14's criterion."""
    words = text.split()
    if len(words) < 12:
        return False
    for chunk_len in (4, 5, 6, 8):
        seen = {}
        for i in range(len(words) - chunk_len + 1):
            chunk = " ".join(words[i:i + chunk_len])
            seen[chunk] = seen.get(chunk, 0) + 1
        if max(seen.values(), default=0) >= min_repeat:
            return True
    return False


def load_features():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["ffn"], z["attn"], z["labels"]
    acts_path = Path("~/Desktop/mech-int/data/processed/activations.pkl").expanduser()
    if not acts_path.exists():
        raise SystemExit(f"Needs unshipped mech-int activations at {acts_path} (see Data availability).")
    with open(acts_path, "rb") as f:
        records = pickle.load(f)
    with open(VENDORED / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    labels = np.array(labeled["labels"])
    n = len(records)
    n_layers, hidden = records[0]["ffn_outputs"].shape[0], records[0]["ffn_outputs"].shape[-1]
    ffn = np.zeros((n, n_layers, hidden), dtype=np.float32)
    attn = np.zeros((n, n_layers, hidden), dtype=np.float32)
    for i, r in enumerate(records):
        ffn[i] = r["ffn_outputs"].mean(axis=1)
        attn[i] = r["attn_outputs"].mean(axis=1)
    np.savez_compressed(CACHE, ffn=ffn, attn=attn, labels=labels)
    return ffn, attn, labels


def probe(X, y, seed):
    """code/02::probe_component_at_layer, seed exposed."""
    actual = max(2, min(5, int(y.sum()), int((y == 0).sum())))
    p = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=seed)
    cv = cross_validate(p, X, y, cv=skf,
                        scoring={"auroc": make_scorer(roc_auc_score, response_method="predict_proba")})
    return float(cv["test_auroc"].mean())


def sweep(X, y):
    at42 = probe(X, y, CODE02_SEED)
    across = np.array([probe(X, y, s) for s in SEEDS])
    return {"auroc_seed42": at42,
            "auroc_seedsweep_mean": float(across.mean()),
            "auroc_seedsweep_sd": float(across.std(ddof=1)),
            "auroc_seedsweep_min": float(across.min()),
            "auroc_seedsweep_max": float(across.max())}


def main():
    with open(VENDORED / "labeled.pkl", "rb") as f:
        labeled = pickle.load(f)
    completions = labeled["completions"]
    jaccard = np.array(labeled["labels"])
    judge = np.array(json.load(open(JUDGE_LABELS_PATH))["judge_labels"])
    assert len(judge) == len(completions) == 534

    degenerate = np.array([is_repetitive(c) for c in completions])
    keep = ~degenerate
    print(f"Degenerate (repetition loop) baseline completions: "
          f"{int(degenerate.sum())}/{len(degenerate)} = {degenerate.mean()*100:.1f}%", flush=True)
    for name, lab in (("jaccard", jaccard), ("judge", judge)):
        for cls, cname in ((1, "correct"), (0, "hallucinated")):
            m = lab == cls
            print(f"  {name} label, {cname} (n={int(m.sum())}): "
                  f"{int(degenerate[m].sum())} degenerate ({degenerate[m].mean()*100:.1f}%)", flush=True)

    ffn, attn, labels_chk = load_features()
    assert np.array_equal(labels_chk, jaccard), "feature cache label mismatch"
    n_layers = ffn.shape[1]

    results = {
        "n_total": int(len(completions)),
        "n_degenerate": int(degenerate.sum()),
        "degenerate_pct": round(float(degenerate.mean() * 100), 2),
        "n_nondegenerate": int(keep.sum()),
        "degeneracy_by_class": {
            lname: {
                cname: {"n": int((lab == cls).sum()),
                        "n_degenerate": int(degenerate[lab == cls].sum()),
                        "pct_degenerate": round(float(degenerate[lab == cls].mean() * 100), 2)}
                for cls, cname in ((1, "correct"), (0, "hallucinated"))
            } for lname, lab in (("jaccard", jaccard), ("judge", judge))
        },
        "seeds_swept": SEEDS,
        "labels": {},
    }

    for lname, lab in (("jaccard", jaccard), ("judge", judge)):
        sub_y = lab[keep]
        print(f"\n=== {lname} label: full n={len(lab)} ({int(lab.sum())} correct) | "
              f"non-degenerate n={int(keep.sum())} ({int(sub_y.sum())} correct) ===", flush=True)
        per_layer = {}
        for li in range(n_layers):
            entry = {}
            for comp, arr in (("ffn", ffn), ("attn", attn)):
                entry[f"{comp}_full"] = sweep(arr[:, li, :], lab)
                entry[f"{comp}_nondegenerate"] = sweep(arr[keep][:, li, :], sub_y)
            per_layer[f"L{li}"] = entry
            print(f"  L{li}: FFN full={entry['ffn_full']['auroc_seed42']:.4f} "
                  f"-> nondeg={entry['ffn_nondegenerate']['auroc_seed42']:.4f} | "
                  f"Attn full={entry['attn_full']['auroc_seed42']:.4f} "
                  f"-> nondeg={entry['attn_nondegenerate']['auroc_seed42']:.4f}", flush=True)
        # peak summaries (seed-sweep mean, the seed-robust version)
        summary = {}
        for comp in ("ffn", "attn"):
            for scope in ("full", "nondegenerate"):
                vals = np.array([per_layer[f"L{li}"][f"{comp}_{scope}"]["auroc_seedsweep_mean"]
                                 for li in range(n_layers)])
                v42 = np.array([per_layer[f"L{li}"][f"{comp}_{scope}"]["auroc_seed42"]
                                for li in range(n_layers)])
                summary[f"{comp}_{scope}"] = {
                    "peak_layer_seedsweep": int(vals.argmax()), "peak_auroc_seedsweep": float(vals.max()),
                    "peak_layer_seed42": int(v42.argmax()), "peak_auroc_seed42": float(v42.max()),
                    "mean_over_layers_seedsweep": float(vals.mean()),
                }
        summary["ffn_layer_majority_full"] = int(sum(
            per_layer[f"L{li}"]["ffn_full"]["auroc_seedsweep_mean"] >
            per_layer[f"L{li}"]["attn_full"]["auroc_seedsweep_mean"] for li in range(n_layers)))
        summary["ffn_layer_majority_nondegenerate"] = int(sum(
            per_layer[f"L{li}"]["ffn_nondegenerate"]["auroc_seedsweep_mean"] >
            per_layer[f"L{li}"]["attn_nondegenerate"]["auroc_seedsweep_mean"] for li in range(n_layers)))
        summary["n_layers"] = int(n_layers)
        results["labels"][lname] = {"per_layer": per_layer, "summary": summary,
                                    "n_full": int(len(lab)), "n_correct_full": int(lab.sum()),
                                    "n_nondegenerate": int(keep.sum()),
                                    "n_correct_nondegenerate": int(sub_y.sum())}
        print(f"  SUMMARY {lname}: FFN peak full L{summary['ffn_full']['peak_layer_seedsweep']}="
              f"{summary['ffn_full']['peak_auroc_seedsweep']:.4f} -> nondeg L"
              f"{summary['ffn_nondegenerate']['peak_layer_seedsweep']}="
              f"{summary['ffn_nondegenerate']['peak_auroc_seedsweep']:.4f}; "
              f"Attn peak full L{summary['attn_full']['peak_layer_seedsweep']}="
              f"{summary['attn_full']['peak_auroc_seedsweep']:.4f} -> nondeg L"
              f"{summary['attn_nondegenerate']['peak_layer_seedsweep']}="
              f"{summary['attn_nondegenerate']['peak_auroc_seedsweep']:.4f}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Paper 1 -- closes a gap a fresh review found: §4 claims "a trivial
length/lexical baseline does not explain the judge label's structure
(chance-level AUROC)", but no such computation existed anywhere in this
repo -- the surface-feature baseline (code/05) was only ever run against
the Jaccard label, never against the validated judge label. This matters
because §3.4 separately establishes that 51.9-53.1% of "hallucinated"
completions are degenerate repetition loops, which are trivially
detectable from surface features (entropy, logit variance) -- so without
this baseline, the validated-label AUROC rise reported in §3.2/3.3 could
be measuring degeneracy-detection rather than a stronger factuality
signal, and the paper had no evidence against that alternative.

This script reruns the identical 6-feature surface classifier
(05_surface_baseline_classifier.py's train_and_evaluate, unmodified)
against the validated judge label instead of Jaccard, using the same
already-cached features.npy (no new extraction).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_module_path = ROOT / "code" / "05_surface_baseline_classifier.py"
_spec = importlib.util.spec_from_file_location("surface_baseline_classifier", _module_path)
_surface_baseline_classifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_surface_baseline_classifier)
train_and_evaluate = _surface_baseline_classifier.train_and_evaluate

DATA_DIR = ROOT / "results" / "surface_baseline"


def main():
    X = np.load(DATA_DIR / "features.npy")
    y_jaccard = np.load(DATA_DIR / "labels.npy")

    with open(ROOT / "results" / "gpt2_full_534_judge_labels.json") as f:
        judge_data = json.load(f)
    y_judge_raw = np.array(judge_data["judge_labels"])
    valid_mask = y_judge_raw != -1

    assert X.shape[0] == len(y_judge_raw), f"length mismatch: {X.shape[0]} vs {len(y_judge_raw)}"

    X_valid = X[valid_mask]
    y_judge = y_judge_raw[valid_mask]
    print(f"n_valid={valid_mask.sum()}, correct={y_judge.sum()}, hallucinated={(y_judge==0).sum()}")

    results, lr, mlp, X_train, X_test, y_train, y_test = train_and_evaluate(
        X_valid, y_judge, test_size=0.2, save_dir=str(DATA_DIR / "models_judge_label")
    )

    print("\n--- Surface baseline vs. VALIDATED JUDGE LABEL ---")
    for name, metrics in results.items():
        cv_mean = metrics.get("cv_auroc_mean")
        cv_std = metrics.get("cv_auroc_std")
        if isinstance(cv_mean, (int, float)) and isinstance(cv_std, (int, float)):
            print(f"  {name:<12} CV AUROC = {cv_mean:.4f} +/- {cv_std:.4f} (5-fold)")
        else:
            print(f"  {name:<12} CV AUROC = {cv_mean} +/- {cv_std} (5-fold)")

    clean = {"n_valid": int(valid_mask.sum()), "n_correct": int(y_judge.sum()),
             "n_hallucinated": int((y_judge == 0).sum()), "results": {}}
    for name, m in results.items():
        clean["results"][name] = {
            "cv_auroc_mean": m.get("cv_auroc_mean"),
            "cv_auroc_std": m.get("cv_auroc_std"),
            "auroc": m.get("auroc"),
        }
    out_path = ROOT / "results" / "surface_baseline_vs_judge_label.json"
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

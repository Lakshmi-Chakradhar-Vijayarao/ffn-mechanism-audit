"""
Phase 3 (Step 3): Surface Feature Classifier — the intentional null result.

Scientific question
-------------------
Can GPT-2's own output statistics predict whether its answer will be hallucinated,
WITHOUT looking inside the model?

This experiment is designed to FAIL in an informative way. If output features
alone achieve AUROC > 0.80, there is no need to look at internal activations.
The weak result here (AUROC ~0.576) is the justification for every subsequent
internal analysis — it proves that external observation is insufficient.

What the surface features capture
----------------------------------
Six scalar features are computed per prompt from the model's logit distribution:

  mean_entropy    : H = -Σ p log p averaged over all generated tokens.
                    Measures average uncertainty in token predictions.
  max_entropy     : Maximum entropy at any single token position.
                    Captures the "hardest" prediction in the sequence.
  logit_variance  : Variance of the top logit value across sequence positions.
                    High variance = model is inconsistently confident.
  confidence_gap  : Mean (p_top1 - p_top2) across positions.
                    Low gap = the model had close alternatives — hesitant prediction.
  attention_entropy : Mean attention weight entropy across all heads and layers.
                    Low entropy = focused attention; high entropy = diffuse attention.
  activation_norm : L2 norm of the final layer's hidden state at the last position.
                    Proxy for the "energy" of the model's final representation.

Why these features are insufficient
------------------------------------
GPT-2 hallucinates with exactly the same fluency and confidence as when it is
correct. The surface statistics — entropy, confidence gap — reflect lexical
fluency, not factual accuracy. A confident wrong answer looks identical to a
confident right answer in logit space.

This is consistent with the literature: HaloScope and ReDeEP both observe that
scalar uncertainty features underperform full activation probes on factual QA tasks.

Models trained
--------------
  Logistic Regression   : Linear model, 5-fold CV. CV AUROC ~0.531.
  MLP (2-layer, ReLU)   : Non-linear model, 5-fold CV. CV AUROC ~0.576.

The MLP's marginal improvement (0.531 → 0.576) shows non-linear interactions
between features carry some additional signal, but neither reaches operational
reliability. Best accuracy is ~57.9% — barely above the 50.5% majority class baseline.

Outputs
-------
  results/logs/predictor_results.txt      — CV AUROC / accuracy / F1 for all models
  results/plots/roc_curve.png             — ROC curves for all models
  results/plots/confusion_matrix.png      — confusion matrix at 0.5 threshold
  results/plots/confidence_vs_accuracy.png — calibration plot

Usage
-----
    python code/05_run_surface_baseline.py

Fixed post-review (final audit pass): this driver previously imported a
nonexistent `src.predictor.classifier` / `src.evaluation.metrics` package
layout and read from a nonexistent `data/processed/` directory -- a stale
leftover from before this project was flattened into numbered `code/`
scripts. `train_and_evaluate` actually lives in the sibling script
`05_surface_baseline_classifier.py` in this same directory; the real
feature/label arrays live in `results/surface_baseline/`. Both are fixed
below. Plotting helpers (`plot_roc_curve`, etc.) never existed in this
project either -- removed rather than stubbed out, since the numeric log
output is what the paper actually cites.
"""

import importlib.util
import numpy as np
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "05_surface_baseline_classifier.py"
_spec = importlib.util.spec_from_file_location("surface_baseline_classifier", _module_path)
_surface_baseline_classifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_surface_baseline_classifier)
train_and_evaluate = _surface_baseline_classifier.train_and_evaluate

DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "surface_baseline"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "surface_baseline"
LOGS_DIR = RESULTS_DIR
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 4: Hallucination Predictor ===\n")

    X = np.load(DATA_DIR / "features.npy")
    y = np.load(DATA_DIR / "labels.npy")
    print(f"Loaded features: {X.shape},  labels: {y.shape}")
    print(f"Class balance — Correct: {y.sum()}, Hallucinated: {(y==0).sum()}\n")

    save_dir = Path(__file__).resolve().parent.parent / "results" / "surface_baseline" / "models"
    results, lr, mlp, X_train, X_test, y_train, y_test = train_and_evaluate(
        X, y, test_size=0.2, save_dir=str(save_dir)
    )

    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    # --- Cross-validated AUROC (fold mean/std; NOT a valid SEM-based CI) ---
    # Fixed post-review: this previously printed a "95% CI" computed as
    # cv_mean +/- 1.96*cv_std over only 5 CV folds -- exactly the invalid
    # fold-std-as-SEM error the paper's own §3.6 identifies and replaces
    # with a label-permutation test. No CI is reported here; cv_auroc_std
    # is fold-to-fold variability only, not a standard error.
    print("\n--- Cross-Validated AUROC (fold mean +/- fold std, not a CI) ---")
    for name in ["logistic", "mlp"]:
        cv_mean = results[name].get("cv_auroc_mean", "N/A")
        cv_std  = results[name].get("cv_auroc_std",  "N/A")
        print(f"  {name:<12}  CV AUROC = {cv_mean:.4f} +/- {cv_std:.4f} (5-fold)")

    best_model_name = max(
        ["logistic", "mlp"],
        key=lambda k: results[k].get("cv_auroc_mean", results[k]["auroc"])
    )
    print(f"\nBest model: {best_model_name} (CV AUROC={results[best_model_name].get('cv_auroc_mean', results[best_model_name]['auroc'])})")
    print("\nClassification Report (held-out test set):")
    print(results[best_model_name]["report"])

    # --- Save results log (matches the already-committed results/surface_baseline/predictor_results.txt) ---
    log_path = LOGS_DIR / "predictor_results.txt"
    with open(log_path, "w") as f:
        for name, metrics in results.items():
            f.write(f"{name}: {metrics}\n")
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()

"""
Hallucination predictor: logistic regression and small MLP trained on feature vectors.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from pathlib import Path


def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
        )),
    ])


def evaluate(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return dict of AUROC, F1, accuracy."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "auroc":    round(roc_auc_score(y_test, y_prob), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "report":   classification_report(y_test, y_pred, target_names=["hallucinated", "correct"]),
    }


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    save_dir: str = None,
) -> dict:
    """
    Split data, train both models, evaluate against all baselines.

    Returns dict with results for: logistic, mlp, random_baseline, prob_only_baseline
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    results = {}

    # --- Model A: Logistic Regression ---
    lr = build_logistic_pipeline()
    lr.fit(X_train, y_train)
    results["logistic"] = evaluate(lr, X_test, y_test)

    # --- Model B: MLP ---
    mlp = build_mlp_pipeline()
    mlp.fit(X_train, y_train)
    results["mlp"] = evaluate(mlp, X_test, y_test)

    # --- Baseline 1: Random predictor ---
    rng = np.random.default_rng(42)
    y_random_prob = rng.random(len(y_test))
    results["random_baseline"] = {
        "auroc":    round(roc_auc_score(y_test, y_random_prob), 4),
        "f1":       None,
        "accuracy": round((y_random_prob > 0.5).mean() == y_test).mean() if False else None,
    }

    # --- Baseline 2: Token probability only (feature index 3 = confidence_gap) ---
    confidence_gap = X_test[:, 3]
    results["prob_only_baseline"] = {
        "auroc": round(roc_auc_score(y_test, confidence_gap), 4),
        "f1":    None,
        "accuracy": None,
    }

    # --- Baseline 3: No intervention (majority class) ---
    majority_prob = np.full(len(y_test), y_train.mean())
    results["majority_baseline"] = {
        "auroc": round(roc_auc_score(y_test, majority_prob), 4),
        "f1":    None,
        "accuracy": round(max(y_test.mean(), 1 - y_test.mean()), 4),
    }

    # --- Cross-validated AUROC (with confidence intervals) ---
    # These are more reliable than the single held-out split above.
    n_splits = min(5, min(int(y.sum()), int((y == 0).sum())))
    n_splits = max(2, n_splits)

    cv_scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_lr  = cross_validate(build_logistic_pipeline(), X, y, cv=skf, scoring=cv_scoring)
    cv_mlp = cross_validate(build_mlp_pipeline(),      X, y, cv=skf, scoring=cv_scoring)

    results["logistic"]["cv_auroc_mean"] = round(float(cv_lr["test_auroc"].mean()),  4)
    results["logistic"]["cv_auroc_std"]  = round(float(cv_lr["test_auroc"].std()),   4)
    results["mlp"]["cv_auroc_mean"]      = round(float(cv_mlp["test_auroc"].mean()), 4)
    results["mlp"]["cv_auroc_std"]       = round(float(cv_mlp["test_auroc"].std()),  4)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, save_dir / "logistic_regression.pkl")
        joblib.dump(mlp, save_dir / "mlp.pkl")
        print(f"Models saved to {save_dir}")

    return results, lr, mlp, X_train, X_test, y_train, y_test

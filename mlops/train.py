"""Training: StandardScaler + LogisticRegression with byte-identical metrics.

Two runs with the same seed produce byte-identical metrics — enforced by
``tests/test_train.py`` and by the milestone gate.
"""

from __future__ import annotations

import json
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops.config import MAX_ITER, SEED
from mlops.data import load_breast_cancer, split_data, validate_frame

__all__ = ["run_training", "metrics_to_bytes"]


def build_model(seed: int = SEED) -> Pipeline:
    """StandardScaler + LogisticRegression(max_iter=1000, random_state=seed)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=MAX_ITER, random_state=seed),
            ),
        ]
    )


def metrics_to_bytes(metrics: dict[str, Any]) -> bytes:
    """Canonical, deterministic JSON encoding of a metrics dict."""
    return json.dumps(
        metrics, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def run_training(seed: int = SEED) -> dict[str, Any]:
    """Train the model and report accuracy + ROC-AUC on the validation split.

    Fully deterministic for a given seed: same data, same split, same
    solver settings, hence byte-identical metrics.
    """
    X, y, feature_names = load_breast_cancer()
    validate_frame(X, y)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        X, y, seed=seed
    )

    model = build_model(seed=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    metrics: dict[str, Any] = {
        "seed": int(seed),
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "feature_names": list(feature_names),
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "seed": int(seed),
    }

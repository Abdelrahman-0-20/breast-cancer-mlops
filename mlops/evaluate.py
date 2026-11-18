"""Evaluation gate (milestone M2): holdout test metrics vs config thresholds.

Charter contract: exit **0 iff** test ROC-AUC >= ``MIN_TEST_ROC_AUC`` and
accuracy >= ``MIN_TEST_ACCURACY`` — both thresholds live in
``mlops/config.py`` and are read at call time, never inlined here. A
non-zero exit prints exactly which threshold failed.

The gate itself never writes artifacts: with no registered model it
trains the canonical seeded model in memory, otherwise it gates the
latest registered model (``--artifacts-dir``, or the default registry
when one exists).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from mlops import config
from mlops.data import load_breast_cancer, split_data, validate_frame
from mlops.registry import latest as _registry_latest
from mlops.registry import load_model as _registry_load_model
from mlops.registry import load_registry as _registry_load_registry
from mlops.train import run_training

__all__ = [
    "EXIT_PASSED",
    "EXIT_FAILED",
    "EXIT_USAGE",
    "compute_test_metrics",
    "gate_failures",
    "evaluate_model",
    "main",
]

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def compute_test_metrics(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Accuracy + ROC-AUC of ``model`` on the holdout test split."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def gate_failures(metrics: Mapping[str, float]) -> list[str]:
    """One line per threshold not met; an empty list means the gate passed.

    Thresholds are read from ``mlops/config.py`` at call time, so the
    gate follows the configuration exactly (never an inline literal).
    """
    failures: list[str] = []
    roc_auc = float(metrics["roc_auc"])
    if roc_auc < config.MIN_TEST_ROC_AUC:
        failures.append(
            f"test ROC-AUC {roc_auc:.4f} < {config.MIN_TEST_ROC_AUC:.4f} "
            f"(MIN_TEST_ROC_AUC)"
        )
    accuracy = float(metrics["accuracy"])
    if accuracy < config.MIN_TEST_ACCURACY:
        failures.append(
            f"test accuracy {accuracy:.4f} < {config.MIN_TEST_ACCURACY:.4f} "
            f"(MIN_TEST_ACCURACY)"
        )
    return failures


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """Holdout metrics plus the gate decision for ``model``."""
    metrics = compute_test_metrics(model, X_test, y_test)
    failures = gate_failures(metrics)
    return {
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
    }


def _holdout_split(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The seeded holdout test split of the canonical dataset."""
    X, y, _ = load_breast_cancer()
    validate_frame(X, y)
    _, _, X_test, _, _, y_test = split_data(X, y, seed=seed)
    return X_test, y_test


def _resolve_model(
    seed: int, artifacts_dir: Path | None
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return ``(model, X_test, y_test)`` for the gate.

    An explicit ``artifacts_dir`` must hold a registered model. Without
    one, the default registry is used when it has an entry; when no model
    has been registered yet the canonical seeded training run provides
    one in memory (evaluation never writes files).
    """
    if artifacts_dir is not None:
        entry = _registry_latest(artifacts_dir)
        model = _registry_load_model(int(entry["version"]), artifacts_dir)
        X_test, y_test = _holdout_split(seed)
        return model, X_test, y_test

    if _registry_load_registry():
        entry = _registry_latest()
        model = _registry_load_model(int(entry["version"]))
        X_test, y_test = _holdout_split(seed)
        return model, X_test, y_test

    run = run_training(seed=seed)
    return run["model"], run["X_test"], run["y_test"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry: 0 iff every config threshold is met, else non-zero."""
    parser = argparse.ArgumentParser(
        prog="python3 -m mlops.evaluate",
        description=(
            "Holdout evaluation gate: exit 0 iff test ROC-AUC >= "
            f"{config.MIN_TEST_ROC_AUC} and accuracy >= "
            f"{config.MIN_TEST_ACCURACY} (thresholds from mlops/config.py)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="seed of the holdout split (default: %(default)s)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "gate the latest model registered in this directory "
            "(default: the default registry when present, else a fresh "
            "canonical training run)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        model, X_test, y_test = _resolve_model(args.seed, args.artifacts_dir)
    except FileNotFoundError as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return EXIT_USAGE

    result = evaluate_model(model, X_test, y_test)
    metrics = result["metrics"]
    print(
        f"holdout test ROC-AUC: {metrics['roc_auc']:.4f} "
        f"(MIN_TEST_ROC_AUC >= {config.MIN_TEST_ROC_AUC:.4f})"
    )
    print(
        f"holdout test accuracy: {metrics['accuracy']:.4f} "
        f"(MIN_TEST_ACCURACY >= {config.MIN_TEST_ACCURACY:.4f})"
    )
    for failure in result["failures"]:
        print(f"FAILED: {failure}")
    if result["passed"]:
        print("evaluation gate PASSED")
        return EXIT_PASSED
    print("evaluation gate FAILED")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())

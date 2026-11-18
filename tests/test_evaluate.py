"""M2 evaluation-gate tests: config thresholds at their exact boundaries.

Charter contract (README): the gate exits **0 iff** test ROC-AUC >=
``MIN_TEST_ROC_AUC`` and accuracy >= ``MIN_TEST_ACCURACY`` — thresholds
live in ``mlops/config.py`` and are read at call time — and a non-zero
exit prints exactly which threshold failed.

Boundary tests use the threshold values themselves (exact equality must
pass) and ``math.nextafter`` one ulp below (must fail), so no rounding
can blur the boundary. Module-level tests invoke ``python3 -m
mlops.evaluate`` exactly as documented.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops import config
from mlops.data import load_breast_cancer, split_data
from mlops.evaluate import (
    EXIT_FAILED,
    EXIT_PASSED,
    EXIT_USAGE,
    compute_test_metrics,
    evaluate_model,
    gate_failures,
    main,
)
from mlops.registry import register
from mlops.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return env


def _run_evaluate_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mlops.evaluate", *args],
        cwd=cwd,
        env=_module_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def run() -> dict:
    """One canonical training run shared by this module's tests."""
    return run_training()


def _register_dummy(artifacts_dir: Path) -> Path:
    """Register a constant-0 model that fails every threshold."""
    X, y, _ = load_breast_cancer()
    X_ref, _, _, y_ref, _, _ = split_data(X, y, seed=config.SEED)
    model = DummyClassifier(strategy="constant", constant=0).fit(X_ref, y_ref)
    register(
        model,
        {"accuracy": 0.0, "roc_auc": 0.0},
        ["x"] * config.EXPECTED_N_FEATURES,
        X_ref,
        y_ref,
        artifacts_dir=artifacts_dir,
    )
    return artifacts_dir


# ------------------------------------------------------------ config contract


def test_thresholds_match_charter_values():
    assert config.MIN_TEST_ROC_AUC == 0.95
    assert config.MIN_TEST_ACCURACY == 0.90


def test_gate_reads_roc_auc_threshold_from_config(monkeypatch):
    """The gate must follow config at call time, never an inline literal."""
    monkeypatch.setattr(config, "MIN_TEST_ROC_AUC", 0.99)
    failures = gate_failures({"roc_auc": 0.95, "accuracy": 0.95})
    assert len(failures) == 1
    assert "MIN_TEST_ROC_AUC" in failures[0]


def test_gate_reads_accuracy_threshold_from_config(monkeypatch):
    monkeypatch.setattr(config, "MIN_TEST_ACCURACY", 0.99)
    failures = gate_failures({"roc_auc": 0.99, "accuracy": 0.95})
    assert len(failures) == 1
    assert "MIN_TEST_ACCURACY" in failures[0]


# ------------------------------------------------------- boundary behaviour


def test_gate_passes_at_exact_boundaries():
    """ROC-AUC == 0.95 and accuracy == 0.90 must pass (>= is inclusive)."""
    failures = gate_failures(
        {"roc_auc": config.MIN_TEST_ROC_AUC, "accuracy": config.MIN_TEST_ACCURACY}
    )
    assert failures == []


def test_gate_passes_above_boundaries():
    assert gate_failures({"roc_auc": 0.99, "accuracy": 0.95}) == []


def test_gate_fails_one_ulp_below_roc_auc_boundary():
    value = math.nextafter(config.MIN_TEST_ROC_AUC, 0.0)
    failures = gate_failures({"roc_auc": value, "accuracy": 0.99})
    assert len(failures) == 1
    assert "ROC-AUC" in failures[0]
    assert "MIN_TEST_ROC_AUC" in failures[0]


def test_gate_fails_one_ulp_below_accuracy_boundary():
    value = math.nextafter(config.MIN_TEST_ACCURACY, 0.0)
    failures = gate_failures({"roc_auc": 0.99, "accuracy": value})
    assert len(failures) == 1
    assert "accuracy" in failures[0]
    assert "MIN_TEST_ACCURACY" in failures[0]


def test_gate_reports_both_thresholds_when_both_fail():
    failures = gate_failures({"roc_auc": 0.5, "accuracy": 0.37})
    assert len(failures) == 2
    assert any("MIN_TEST_ROC_AUC" in line for line in failures)
    assert any("MIN_TEST_ACCURACY" in line for line in failures)


# ---------------------------------------------------------- metric plumbing


def test_compute_test_metrics_matches_manual_reference():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 0, 0, 1])
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    metrics = compute_test_metrics(model, X, y)
    assert metrics["accuracy"] == pytest.approx(0.75)  # 3 of 4 correct
    assert metrics["roc_auc"] == pytest.approx(0.5)  # constant scores


class _ScriptedModel:
    """A deterministic stand-in: fixed labels, fixed P(class=1) per row."""

    def __init__(self, labels: np.ndarray, p_one):
        self._labels = np.asarray(labels)
        self._p_one = np.broadcast_to(np.asarray(p_one, dtype=float), self._labels.shape)

    def predict(self, X):
        return self._labels.copy()

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._p_one, self._p_one])


def test_evaluate_model_pass_flag_consistent_with_failures():
    y = np.array([0, 1, 0, 1])
    X = np.zeros((4, 1))
    # perfectly ranked probabilities -> ROC-AUC 1.0, exact labels -> 1.0
    passing = evaluate_model(_ScriptedModel(y, y.astype(float)), X, y)
    assert passing["metrics"]["accuracy"] == 1.0
    assert passing["metrics"]["roc_auc"] == 1.0
    assert passing["failures"] == []
    assert passing["passed"] is True

    # constant scores -> AUC 0.5, always-0 labels -> accuracy 0.5
    failing = evaluate_model(
        _ScriptedModel(np.zeros(4, dtype=int), 0.3), X, y
    )
    assert len(failing["failures"]) == 2
    assert failing["passed"] is False


# ------------------------------------------------------------- main() exits


def test_main_exits_zero_without_any_registry(tmp_path, monkeypatch):
    """No registered model -> canonical seeded training -> gate passes."""
    monkeypatch.chdir(tmp_path)
    assert main([]) == EXIT_PASSED == 0


def test_main_exits_one_and_prints_failing_thresholds(
    tmp_path, monkeypatch, capsys
):
    artifacts = _register_dummy(tmp_path / "badart")
    monkeypatch.chdir(tmp_path)
    code = main(["--artifacts-dir", str(artifacts)])
    assert code == EXIT_FAILED == 1
    out = capsys.readouterr().out
    assert "MIN_TEST_ROC_AUC" in out
    assert "MIN_TEST_ACCURACY" in out


def test_main_exits_zero_for_registered_passing_model(run, tmp_path, monkeypatch):
    artifacts = tmp_path / "goodart"
    register(
        run["model"],
        run["metrics"],
        run["feature_names"],
        run["X_train"],
        run["y_train"],
        artifacts_dir=artifacts,
    )
    monkeypatch.chdir(tmp_path)
    assert main(["--artifacts-dir", str(artifacts)]) == EXIT_PASSED == 0


def test_main_missing_artifacts_dir_exits_usage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--artifacts-dir", str(tmp_path / "does-not-exist")]) == (
        EXIT_USAGE
    ) == 2


# ------------------------------------------------- module invokability (CLI)


def test_module_invokable_exits_zero(tmp_path):
    proc = _run_evaluate_module([], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "evaluation gate PASSED" in proc.stdout


def test_module_invokable_failing_model_names_thresholds(tmp_path):
    artifacts = _register_dummy(tmp_path / "badart")
    proc = _run_evaluate_module(["--artifacts-dir", str(artifacts)], cwd=tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "MIN_TEST_ROC_AUC" in proc.stdout
    assert "MIN_TEST_ACCURACY" in proc.stdout

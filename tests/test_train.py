"""M1 training tests: model contract + byte-identical same-seed metrics.

The charter requires ``StandardScaler`` + ``LogisticRegression(max_iter=1000,
random_state=seed)`` reporting accuracy + ROC-AUC on the validation split,
and two runs with the same seed producing byte-identical metrics.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops.config import EXPECTED_N_FEATURES, MAX_ITER, SEED
from mlops.data import load_breast_cancer
from mlops.train import build_model, metrics_to_bytes, run_training

EXPECTED_SPLIT_SIZES = (398, 85, 86)


@pytest.fixture(scope="module")
def run() -> dict:
    return run_training(SEED)


@pytest.fixture(scope="module")
def rerun() -> dict:
    """A second, independent training run with the same seed."""
    return run_training(SEED)


@pytest.fixture(scope="module")
def other_seed_run() -> dict:
    return run_training(SEED + 1)


# ------------------------------------------------------------- run_training


def test_result_contains_required_keys(run):
    required = {
        "model",
        "metrics",
        "feature_names",
        "X_train",
        "y_train",
        "X_valid",
        "y_valid",
        "X_test",
        "y_test",
        "seed",
    }
    assert required <= set(run)


def test_metrics_has_required_fields(run):
    assert {"seed", "accuracy", "roc_auc", "n_train", "n_valid", "n_test"} <= set(
        run["metrics"]
    )


def test_metrics_values_are_unit_interval_floats(run):
    for key in ("accuracy", "roc_auc"):
        value = run["metrics"][key]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_metrics_split_sizes(run):
    assert run["metrics"]["n_train"] == EXPECTED_SPLIT_SIZES[0]
    assert run["metrics"]["n_valid"] == EXPECTED_SPLIT_SIZES[1]
    assert run["metrics"]["n_test"] == EXPECTED_SPLIT_SIZES[2]


def test_metrics_records_seed(run):
    assert run["metrics"]["seed"] == SEED
    assert run["seed"] == SEED


def test_returned_arrays_match_contract(run):
    assert run["X_train"].shape == (EXPECTED_SPLIT_SIZES[0], EXPECTED_N_FEATURES)
    assert run["X_valid"].shape == (EXPECTED_SPLIT_SIZES[1], EXPECTED_N_FEATURES)
    assert run["X_test"].shape == (EXPECTED_SPLIT_SIZES[2], EXPECTED_N_FEATURES)
    assert run["y_train"].shape == (EXPECTED_SPLIT_SIZES[0],)
    assert run["y_valid"].shape == (EXPECTED_SPLIT_SIZES[1],)
    assert run["y_test"].shape == (EXPECTED_SPLIT_SIZES[2],)


def test_feature_names_match_dataset(run):
    _, _, feature_names = load_breast_cancer()
    assert run["feature_names"] == feature_names


def test_reported_accuracy_matches_recomputation(run):
    model = run["model"]
    y_pred = model.predict(run["X_valid"])
    recomputed = accuracy_score(run["y_valid"], y_pred)
    assert recomputed == run["metrics"]["accuracy"]


def test_reported_roc_auc_matches_recomputation(run):
    model = run["model"]
    y_proba = model.predict_proba(run["X_valid"])[:, 1]
    recomputed = roc_auc_score(run["y_valid"], y_proba)
    assert recomputed == run["metrics"]["roc_auc"]


def test_predictions_are_binary(run):
    y_pred = run["model"].predict(run["X_valid"])
    assert set(np.unique(y_pred).tolist()) <= {0, 1}


# --------------------------------------------------------------- determinism


def test_same_seed_metrics_are_byte_identical(run, rerun):
    """Charter requirement: two runs with the same seed -> identical bytes."""
    assert metrics_to_bytes(run["metrics"]) == metrics_to_bytes(rerun["metrics"])


def test_same_seed_metrics_dicts_are_equal(run, rerun):
    assert run["metrics"] == rerun["metrics"]


def test_same_seed_uses_identical_split(run, rerun):
    for key in ("X_train", "X_valid", "X_test", "y_train", "y_valid", "y_test"):
        assert np.array_equal(run[key], rerun[key]), key


def test_different_seed_uses_different_split(run, other_seed_run):
    assert not np.array_equal(run["X_train"], other_seed_run["X_train"])
    assert other_seed_run["seed"] == SEED + 1


# ------------------------------------------------------------ metrics_to_bytes


def test_metrics_to_bytes_round_trips(run):
    raw = metrics_to_bytes(run["metrics"])
    assert isinstance(raw, bytes)
    assert json.loads(raw.decode("utf-8")) == run["metrics"]


def test_metrics_to_bytes_is_compact_and_sorted(run):
    text = metrics_to_bytes(run["metrics"]).decode("utf-8")
    assert ", " not in text and ": " not in text
    assert list(json.loads(text)) == sorted(run["metrics"])


def test_metrics_to_bytes_ignores_key_order():
    first = {"accuracy": 0.5, "roc_auc": 0.9, "seed": 42}
    second = {"seed": 42, "roc_auc": 0.9, "accuracy": 0.5}
    assert metrics_to_bytes(first) == metrics_to_bytes(second)


def test_metrics_to_bytes_depends_on_values():
    base = {"accuracy": 0.5, "seed": 42}
    other = {"accuracy": 0.500001, "seed": 42}
    assert metrics_to_bytes(base) != metrics_to_bytes(other)


# ------------------------------------------------------------- model contract


def test_build_model_is_a_pipeline():
    model = build_model()
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["scaler", "clf"]


def test_scaler_is_standard_scaler():
    scaler = build_model().named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)


def test_clf_is_logistic_regression_with_contract_params():
    clf = build_model().named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.max_iter == MAX_ITER == 1000
    assert clf.random_state == SEED == 42


def test_clf_random_state_follows_requested_seed():
    clf = build_model(seed=123).named_steps["clf"]
    assert clf.random_state == 123


def test_model_is_fitted_after_training(run):
    model = run["model"]
    assert hasattr(model.named_steps["clf"], "coef_")
    assert model.named_steps["clf"].n_features_in_ == EXPECTED_N_FEATURES

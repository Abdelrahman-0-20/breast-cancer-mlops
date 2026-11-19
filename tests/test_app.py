"""Tests for the Streamlit app helpers (milestone M4)."""

from __future__ import annotations

import numpy as np

import app
from mlops.data import load_breast_cancer


def test_class_distribution_counts():
    y = np.array([0, 0, 1, 1, 1])
    table = app.class_distribution(y)
    assert list(table["class"]) == ["benign", "malignant"]
    assert list(table["count"]) == [2, 3]
    assert int(table["count"].sum()) == 5


def test_feature_frame_shape_and_labels():
    X, y, names = load_breast_cancer()
    frame = app.feature_frame(X, y, names)
    assert frame.shape == (569, 31)
    assert list(frame.columns[-1]) == ["class"] if False else frame.columns[-1] == "class"
    assert set(frame["class"]) == {"benign", "malignant"}
    assert len(frame.columns) == len(names) + 1


def test_correlation_table_is_square_and_symmetric():
    X, y, names = load_breast_cancer()
    frame = app.feature_frame(X, y, names)
    corr = app.correlation_table(frame, names)
    assert corr.shape == (30, 30)
    assert float(corr.loc[names[0], names[0]]) == 1.0


def test_confusion_table_labels():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    table = app.confusion_table(y_true, y_pred)
    assert table.shape == (2, 2)
    assert int(table.to_numpy().sum()) == 4
    assert "actual benign" in table.index


def test_verdict_text_is_ascii_and_mentions_class():
    for label in (0, 1):
        text = app.verdict_text(label, 0.91)
        assert text.isascii(), "verdict must stay ascii (no emojis)"
    assert "malignant" in app.verdict_text(1, 0.93)
    assert "benign" in app.verdict_text(0, 0.93)


def test_predict_one_agrees_with_model():
    from mlops.train import run_training

    run = run_training(seed=42)
    row = run["X_test"][0]
    label, probability = app.predict_one(run["model"], row)
    assert label in (0, 1)
    assert 0.0 <= probability <= 1.0
    expected = int(run["model"].predict(row.reshape(1, -1))[0])
    assert label == expected


def test_evaluate_current_passes_gate():
    model, _, _ = app.model_bundle()
    result = app.evaluate_current(model)
    assert result["passed"] is True
    assert result["metrics"]["roc_auc"] >= 0.95

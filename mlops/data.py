"""Data layer: load the breast-cancer dataset, split it, validate it.

All arrays are plain numpy arrays (dependencies are limited to
scikit-learn, joblib, scipy and pytest — no pandas).
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.datasets import load_breast_cancer as _sklearn_load_breast_cancer
from sklearn.model_selection import train_test_split

from mlops.config import (
    EXPECTED_N_FEATURES,
    SEED,
    TEST_FRACTION,
    VALID_FRACTION,
)

__all__ = [
    "load_breast_cancer",
    "split_data",
    "validate_frame",
    "sha256_of_training_data",
]

_TARGET_CLASSES = (0, 1)


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load sklearn's built-in breast-cancer dataset (zero network).

    Returns ``(X, y, feature_names)`` where ``X`` is a float64 matrix of
    569 samples x 30 features, ``y`` is an int array of labels in {0, 1}
    and ``feature_names`` is the list of 30 feature names.
    """
    dataset = _sklearn_load_breast_cancer()
    X = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.int64)
    feature_names = [str(name) for name in dataset.feature_names]
    return X, y, feature_names


def validate_frame(X: np.ndarray, y: np.ndarray) -> None:
    """Contract check: raise ``ValueError`` unless the frame is valid.

    Checks: 2-D feature matrix, expected feature count (30), no NaNs,
    matching lengths, target values strictly in {0, 1} and both classes
    present. Returns ``None`` on success.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"expected {EXPECTED_N_FEATURES} features, got {X.shape[1]}"
        )
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")

    classes = np.unique(y)
    if not np.isin(classes, np.asarray(_TARGET_CLASSES)).all():
        raise ValueError(f"target must be in {_TARGET_CLASSES}, got {classes}")
    if len(classes) != len(_TARGET_CLASSES):
        raise ValueError(f"target must contain both classes, got {classes}")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, ...]:
    """Stratified 70/15/15 train/valid/test split with the given seed.

    Returns ``(X_train, X_valid, X_test, y_train, y_valid, y_test)``.
    With the full 569-sample dataset and seed 42 this yields
    398 train / 85 valid / 86 test samples.
    """
    test_size = VALID_FRACTION + TEST_FRACTION  # 0.30 held out
    X_train, X_held_out, y_train, y_held_out = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    # carve the 15% validation slice out of the 30% held-out slice
    valid_share = VALID_FRACTION / test_size  # 0.5 of the held-out slice
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_held_out,
        y_held_out,
        test_size=1.0 - valid_share,
        random_state=seed,
        stratify=y_held_out,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def sha256_of_training_data(X_train: np.ndarray, y_train: np.ndarray) -> str:
    """Deterministic sha256 hex digest over the training features + labels."""
    X = np.ascontiguousarray(X_train, dtype=np.float64)
    y = np.ascontiguousarray(y_train, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(X.shape).encode("utf-8"))
    digest.update(X.tobytes())
    digest.update(str(y.shape).encode("utf-8"))
    digest.update(y.tobytes())
    return digest.hexdigest()

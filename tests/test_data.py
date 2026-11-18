"""M1 contract tests for the data layer (mlops/data.py).

Covers: dataset loading, the ``validate_frame`` contract, the seeded
stratified 70/15/15 split and the training-data sha256 helper.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from mlops.config import (
    EXPECTED_N_FEATURES,
    SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALID_FRACTION,
)
from mlops.data import (
    load_breast_cancer,
    sha256_of_training_data,
    split_data,
    validate_frame,
)

N_SAMPLES = 569
EXPECTED_SPLIT_SIZES = (398, 85, 86)  # 70/15/15 of 569 with seed 42


@pytest.fixture(scope="module")
def frame() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return load_breast_cancer()


@pytest.fixture(scope="module")
def split(frame) -> tuple[np.ndarray, ...]:
    X, y, _ = frame
    return split_data(X, y)


# ---------------------------------------------------------------- loading


def test_load_shape(frame):
    X, y, _ = frame
    assert X.shape == (N_SAMPLES, EXPECTED_N_FEATURES)
    assert y.shape == (N_SAMPLES,)


def test_load_feature_names(frame):
    _, _, feature_names = frame
    assert len(feature_names) == EXPECTED_N_FEATURES
    assert all(isinstance(name, str) and name for name in feature_names)


def test_load_target_is_binary(frame):
    _, y, _ = frame
    assert set(np.unique(y).tolist()) == {0, 1}


def test_load_dtypes(frame):
    X, y, _ = frame
    assert X.dtype == np.float64
    assert np.issubdtype(y.dtype, np.integer)


def test_load_is_deterministic():
    X1, y1, names1 = load_breast_cancer()
    X2, y2, names2 = load_breast_cancer()
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert names1 == names2


# ----------------------------------------------------------- validate_frame


def test_validate_frame_accepts_real_data(frame):
    X, y, _ = frame
    assert validate_frame(X, y) is None


def test_validate_frame_rejects_wrong_feature_count(frame):
    X, y, _ = frame
    with pytest.raises(ValueError, match="features"):
        validate_frame(X[:, :-1], y)


def test_validate_frame_rejects_nan(frame):
    X, y, _ = frame
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_frame(X_bad, y)


def test_validate_frame_rejects_inf(frame):
    X, y, _ = frame
    X_bad = X.copy()
    X_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_frame(X_bad, y)


def test_validate_frame_rejects_length_mismatch(frame):
    X, y, _ = frame
    with pytest.raises(ValueError, match="length mismatch"):
        validate_frame(X, y[:-1])


def test_validate_frame_rejects_out_of_range_target(frame):
    X, y, _ = frame
    y_bad = y.copy()
    y_bad[0] = 2
    with pytest.raises(ValueError, match="target must be in"):
        validate_frame(X, y_bad)


def test_validate_frame_rejects_single_class_target(frame):
    X, y, _ = frame
    with pytest.raises(ValueError, match="both classes"):
        validate_frame(X, np.zeros_like(y))


def test_validate_frame_rejects_non_2d_features(frame):
    X, y, _ = frame
    with pytest.raises(ValueError, match="2-D"):
        validate_frame(X[0], y)


# ------------------------------------------------------------------- split


def test_split_sizes(split):
    X_train, X_valid, X_test, y_train, y_valid, y_test = split
    assert (len(X_train), len(X_valid), len(X_test)) == EXPECTED_SPLIT_SIZES
    assert (len(y_train), len(y_valid), len(y_test)) == EXPECTED_SPLIT_SIZES


def test_split_returns_six_arrays(split):
    assert len(split) == 6
    assert all(isinstance(part, np.ndarray) for part in split)


def test_split_dtypes(split):
    X_train, X_valid, X_test, y_train, y_valid, y_test = split
    for X_part in (X_train, X_valid, X_test):
        assert X_part.dtype == np.float64
    for y_part in (y_train, y_valid, y_test):
        assert np.issubdtype(y_part.dtype, np.integer)


def test_split_fractions_match_config(split):
    X_train, X_valid, X_test, _, _, _ = split
    total = len(X_train) + len(X_valid) + len(X_test)
    assert total == N_SAMPLES
    assert abs(len(X_train) / total - TRAIN_FRACTION) < 0.01
    assert abs(len(X_valid) / total - VALID_FRACTION) < 0.01
    assert abs(len(X_test) / total - TEST_FRACTION) < 0.01
    assert TRAIN_FRACTION + VALID_FRACTION + TEST_FRACTION == 1.0


def test_split_is_stratified(split):
    _, _, _, y_train, y_valid, y_test = split
    _, y_full, _ = load_breast_cancer()
    full_rate = float(y_full.mean())
    for part in (y_train, y_valid, y_test):
        assert set(np.unique(part).tolist()) == {0, 1}
        assert abs(float(part.mean()) - full_rate) < 0.02


def test_split_is_deterministic_for_seed_42(frame):
    X, y, _ = frame
    first = split_data(X, y, seed=42)
    second = split_data(X, y, seed=42)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_split_default_seed_is_configured_seed(frame):
    X, y, _ = frame
    default_split = split_data(X, y)
    explicit_split = split_data(X, y, seed=SEED)
    for a, b in zip(default_split, explicit_split):
        assert np.array_equal(a, b)


def test_split_differs_for_different_seed(frame):
    X, y, _ = frame
    train_a = split_data(X, y, seed=SEED)[0]
    train_b = split_data(X, y, seed=0)[0]
    assert not np.array_equal(train_a, train_b)


def test_split_partitions_are_disjoint_and_complete(split):
    X_train, X_valid, X_test, _, _, _ = split
    X_full, _, _ = load_breast_cancer()
    train_rows = {row.tobytes() for row in X_train}
    valid_rows = {row.tobytes() for row in X_valid}
    test_rows = {row.tobytes() for row in X_test}
    assert not (train_rows & valid_rows)
    assert not (train_rows & test_rows)
    assert not (valid_rows & test_rows)
    all_rows = train_rows | valid_rows | test_rows
    assert all_rows == {row.tobytes() for row in X_full}
    assert len(all_rows) == N_SAMPLES


# ------------------------------------------------------------------ sha256


def test_sha256_is_64_hex_chars(split):
    X_train, _, _, y_train, _, _ = split
    digest = sha256_of_training_data(X_train, y_train)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_sha256_is_deterministic(split):
    X_train, _, _, y_train, _, _ = split
    first = sha256_of_training_data(X_train, y_train)
    second = sha256_of_training_data(X_train, y_train)
    assert first == second


def test_sha256_changes_when_a_label_flips(split):
    X_train, _, _, y_train, _, _ = split
    y_flipped = y_train.copy()
    y_flipped[0] = 1 - y_flipped[0]
    assert sha256_of_training_data(X_train, y_train) != sha256_of_training_data(
        X_train, y_flipped
    )


def test_sha256_changes_when_a_feature_changes(split):
    X_train, _, _, y_train, _, _ = split
    X_changed = X_train.copy()
    X_changed[0, 0] += 1.0
    assert sha256_of_training_data(X_train, y_train) != sha256_of_training_data(
        X_changed, y_train
    )


def test_sha256_label_dtype_insensitive(split):
    X_train, _, _, y_train, _, _ = split
    assert sha256_of_training_data(X_train, y_train) == sha256_of_training_data(
        X_train, y_train.astype(np.int32)
    )

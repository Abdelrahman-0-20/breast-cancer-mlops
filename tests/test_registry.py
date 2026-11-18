"""M1 registry tests: register/load round-trip, latest(), entry fields.

The charter requires ``artifacts/model-v{n}.joblib`` + ``artifacts/registry.json``
recording version, trained_at, metrics, feature names and a sha256 of the
training data, with ``latest()`` always resolving the newest version.
Every test runs against an isolated ``tmp_path`` so the repository's own
``artifacts/`` directory is never touched.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mlops.config import SEED
from mlops.data import sha256_of_training_data
from mlops.registry import (
    DEFAULT_ARTIFACTS_DIR,
    REGISTRY_FILENAME,
    latest,
    load_model,
    load_registry,
    model_path_for,
    register,
)
from mlops.train import run_training

REQUIRED_ENTRY_FIELDS = {
    "version",
    "trained_at",
    "metrics",
    "feature_names",
    "training_data_sha256",
    "artifact",
}


@pytest.fixture(scope="module")
def run() -> dict:
    return run_training(SEED)


@pytest.fixture()
def artifacts_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _register(run: dict, artifacts_dir: Path) -> dict:
    return register(
        run["model"],
        run["metrics"],
        run["feature_names"],
        run["X_train"],
        run["y_train"],
        artifacts_dir=artifacts_dir,
    )


def _registry_file(artifacts_dir: Path) -> Path:
    return artifacts_dir / REGISTRY_FILENAME


# ------------------------------------------------------------- entry contract


def test_register_returns_entry_with_required_fields(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert REQUIRED_ENTRY_FIELDS <= set(entry)


def test_register_writes_artifact_file(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert entry["artifact"] == "model-v1.joblib"
    assert (artifacts_dir / "model-v1.joblib").is_file()


def test_register_writes_registry_json(run, artifacts_dir):
    _register(run, artifacts_dir)
    assert _registry_file(artifacts_dir).is_file()


def test_registry_json_has_entries_list(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    payload = json.loads(_registry_file(artifacts_dir).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["entries"] == [entry]


def test_load_registry_missing_dir_returns_empty(artifacts_dir):
    assert load_registry(artifacts_dir) == []


def test_version_starts_at_one(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert entry["version"] == 1


def test_versions_increment_on_repeated_registration(run, artifacts_dir):
    first = _register(run, artifacts_dir)
    second = _register(run, artifacts_dir)
    assert (first["version"], second["version"]) == (1, 2)
    assert (artifacts_dir / "model-v1.joblib").is_file()
    assert (artifacts_dir / "model-v2.joblib").is_file()


def test_trained_at_is_iso8601_with_timezone(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    trained_at = datetime.fromisoformat(entry["trained_at"])
    assert trained_at.tzinfo is not None


def test_entry_metrics_preserved_exactly(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert entry["metrics"] == run["metrics"]


def test_entry_feature_names_preserved_exactly(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert entry["feature_names"] == run["feature_names"]


def test_entry_training_data_sha256(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    digest = entry["training_data_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == sha256_of_training_data(run["X_train"], run["y_train"])


# -------------------------------------------------------------------- latest


def test_latest_resolves_first_registration(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    assert latest(artifacts_dir)["version"] == entry["version"] == 1


def test_latest_resolves_newest_version(run, artifacts_dir):
    _register(run, artifacts_dir)
    _register(run, artifacts_dir)
    assert latest(artifacts_dir)["version"] == 2


def test_latest_ignores_json_entry_order(run, artifacts_dir):
    _register(run, artifacts_dir)
    _register(run, artifacts_dir)
    path = _registry_file(artifacts_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].reverse()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert latest(artifacts_dir)["version"] == 2


def test_latest_raises_when_registry_is_empty(artifacts_dir):
    with pytest.raises(FileNotFoundError):
        latest(artifacts_dir)


def test_latest_metrics_equal_registered_metrics(run, artifacts_dir):
    _register(run, artifacts_dir)
    assert latest(artifacts_dir)["metrics"] == run["metrics"]


# -------------------------------------------------------------- model roundtrip


def test_load_model_roundtrip_predicts_identically(run, artifacts_dir):
    entry = _register(run, artifacts_dir)
    original = run["model"].predict(run["X_valid"])
    loaded = load_model(entry["version"], artifacts_dir)
    assert np.array_equal(loaded.predict(run["X_valid"]), original)
    assert hasattr(loaded, "predict_proba")


def test_load_model_missing_version_raises(artifacts_dir):
    with pytest.raises(FileNotFoundError):
        load_model(99, artifacts_dir)


def test_model_path_for_names_the_artifact(artifacts_dir):
    assert model_path_for(3, artifacts_dir) == artifacts_dir / "model-v3.joblib"


def test_default_paths_follow_charter():
    assert DEFAULT_ARTIFACTS_DIR == Path("artifacts")
    assert REGISTRY_FILENAME == "registry.json"

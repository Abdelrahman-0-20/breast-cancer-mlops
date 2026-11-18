"""Model registry: versioned artifacts + registry.json bookkeeping.

Each ``register()`` call writes ``artifacts/model-v{n}.joblib`` and
records an entry in ``artifacts/registry.json`` with the version,
training timestamp, metrics, feature names and a sha256 of the training
data. ``latest()`` always resolves the newest version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from mlops.data import sha256_of_training_data

__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "REGISTRY_FILENAME",
    "register",
    "latest",
    "load_model",
    "load_registry",
    "model_path_for",
]

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
REGISTRY_FILENAME = "registry.json"


def model_path_for(version: int, artifacts_dir: Path | str | None = None) -> Path:
    """Path of the joblib artifact for ``version``."""
    directory = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    return directory / f"model-v{version}.joblib"


def _registry_path(artifacts_dir: Path | str | None) -> Path:
    directory = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    return directory / REGISTRY_FILENAME


def load_registry(artifacts_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Return the list of registry entries (empty when none recorded yet)."""
    path = _registry_path(artifacts_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["entries"] if isinstance(payload, dict) else payload
    return list(entries)


def register(
    model: Any,
    metrics: dict[str, Any],
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Persist ``model`` as the next version and record its metadata.

    Returns the registry entry that was written.
    """
    directory = (
        Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    )
    directory.mkdir(parents=True, exist_ok=True)

    entries = load_registry(directory)
    version = max((int(entry["version"]) for entry in entries), default=0) + 1

    entry: dict[str, Any] = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": dict(metrics),
        "feature_names": list(feature_names),
        "training_data_sha256": sha256_of_training_data(X_train, y_train),
        "artifact": f"model-v{version}.joblib",
    }

    joblib.dump(model, directory / entry["artifact"])
    entries.append(entry)
    _registry_path(directory).write_text(
        json.dumps({"entries": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return entry


def latest(artifacts_dir: Path | str | None = None) -> dict[str, Any]:
    """Return the registry entry with the highest version."""
    entries = load_registry(artifacts_dir)
    if not entries:
        directory = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
        raise FileNotFoundError(f"no registry entries in {directory}")
    return max(entries, key=lambda entry: int(entry["version"]))


def load_model(version: int, artifacts_dir: Path | str | None = None) -> Any:
    """Load and return the model artifact registered as ``version``."""
    path = model_path_for(version, artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(f"no artifact for version {version}: {path}")
    return joblib.load(path)

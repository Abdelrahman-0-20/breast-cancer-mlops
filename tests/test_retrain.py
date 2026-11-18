"""M2 auto-retrain tests: exit-code API + reference+incoming retraining.

Charter contract (README): when drifted -> retrain on reference+incoming
and register the **next** registry version; **exit 0 = retrained, exit 3
= no drift** — these distinct codes are an API. The drift verdict itself
comes from ``mlops/drift.py`` (report written either way).

Unit tests call ``main()`` in-process against isolated ``tmp_path``
registries; two tests invoke ``python3 -m mlops.retrain`` exactly as
documented to keep the module invokability honest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlops import config
from mlops.data import load_breast_cancer, sha256_of_training_data, split_data
from mlops.registry import REGISTRY_FILENAME, register
from mlops.retrain import EXIT_NO_DRIFT, EXIT_RETRAINED, main
from mlops.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return env


def _run_retrain_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mlops.retrain", *args],
        cwd=cwd,
        env=_module_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return load_breast_cancer()


@pytest.fixture(scope="module")
def reference_split(data):
    X, y, feature_names = data
    X_ref, _, _, y_ref, _, _ = split_data(X, y, seed=config.SEED)
    return X_ref, y_ref, feature_names


@pytest.fixture()
def clean_batch(tmp_path, reference_split) -> Path:
    X_ref, y_ref, _ = reference_split
    path = tmp_path / "clean.npz"
    np.savez(path, X=X_ref, y=y_ref)
    return path


@pytest.fixture()
def perturbed_batch(tmp_path, reference_split) -> Path:
    X_ref, y_ref, _ = reference_split
    path = tmp_path / "perturbed.npz"
    np.savez(path, X=X_ref * 5.0 + 10.0, y=y_ref)
    return path


def _args(batch: Path, report: Path, artifacts: Path) -> list[str]:
    return [
        "--batch", str(batch),
        "--report", str(report),
        "--artifacts-dir", str(artifacts),
    ]


# ------------------------------------------------------ documented exit codes


def test_exit_codes_are_the_documented_api():
    assert EXIT_RETRAINED == 0
    assert EXIT_NO_DRIFT == 3
    assert EXIT_RETRAINED != EXIT_NO_DRIFT


def test_clean_batch_exits_3_writes_report_and_skips_registration(
    tmp_path, clean_batch
):
    report = tmp_path / "report.json"
    artifacts = tmp_path / "artifacts"
    assert main(_args(clean_batch, report, artifacts)) == EXIT_NO_DRIFT == 3
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["drifted"] is False
    # nothing to do -> the registry directory must not even be created
    assert not artifacts.exists()


def test_perturbed_batch_exits_0_and_registers_version_1(
    tmp_path, perturbed_batch
):
    report = tmp_path / "report.json"
    artifacts = tmp_path / "artifacts"
    assert main(_args(perturbed_batch, report, artifacts)) == EXIT_RETRAINED == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["drifted"] is True
    registry = json.loads(
        (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    assert [entry["version"] for entry in registry["entries"]] == [1]
    assert (artifacts / "model-v1.joblib").is_file()


def test_retrain_registers_the_next_version_after_existing_ones(
    tmp_path, reference_split, perturbed_batch
):
    X_ref, y_ref, feature_names = reference_split
    run = run_training()
    artifacts = tmp_path / "artifacts"
    register(
        run["model"], run["metrics"], run["feature_names"],
        run["X_train"], run["y_train"], artifacts_dir=artifacts,
    )  # version 1 (the pre-drift model)
    assert main(_args(perturbed_batch, tmp_path / "r.json", artifacts)) == 0
    registry = json.loads(
        (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    assert [entry["version"] for entry in registry["entries"]] == [1, 2]
    assert (artifacts / "model-v2.joblib").is_file()
    # sanity: the first entry was left untouched by the retrain
    assert registry["entries"][0]["feature_names"] == feature_names


def test_retrain_trains_on_reference_plus_incoming(
    tmp_path, reference_split
):
    X_ref, y_ref, _ = reference_split
    artifacts = tmp_path / "artifacts"
    # perturb only enough to trigger drift, keeping labels valid
    X_pert = np.concatenate([X_ref, X_ref * 5.0 + 10.0], axis=0)
    y_pert = np.concatenate([y_ref, y_ref])
    batch = tmp_path / "mix.npz"
    np.savez(batch, X=X_pert, y=y_pert)
    assert main(_args(batch, tmp_path / "r.json", artifacts)) == 0

    entry = json.loads(
        (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8")
    )["entries"][-1]
    # training-data digest proves the fit saw reference + incoming rows
    expected_digest = sha256_of_training_data(
        np.concatenate([X_ref, X_pert], axis=0),
        np.concatenate([y_ref, y_pert], axis=0),
    )
    assert entry["training_data_sha256"] == expected_digest
    assert entry["metrics"]["n_train"] == len(X_ref) + len(X_pert)
    assert entry["metrics"]["n_reference"] == len(X_ref)
    assert entry["metrics"]["n_incoming"] == len(X_pert)
    assert 0.0 <= entry["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= entry["metrics"]["roc_auc"] <= 1.0


# ------------------------------------------------- module invokability (CLI)


def test_module_invokable_exit_3_when_no_drift(tmp_path, clean_batch):
    artifacts = tmp_path / "art"
    proc = _run_retrain_module(
        _args(clean_batch, tmp_path / "r.json", artifacts), cwd=tmp_path
    )
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert (tmp_path / "r.json").is_file()
    assert not artifacts.exists()


def test_module_invokable_exit_0_when_retrained(tmp_path, perturbed_batch):
    artifacts = tmp_path / "art"
    proc = _run_retrain_module(
        _args(perturbed_batch, tmp_path / "r.json", artifacts), cwd=tmp_path
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    registry = json.loads(
        (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    assert [entry["version"] for entry in registry["entries"]] == [1]

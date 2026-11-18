"""M2 integration: train -> evaluate -> drift on a perturbed batch ->
retrain, invoking every module exactly as documented (``python3 -m ...``).

Each step asserts the charter's exit-code API end to end:

* ``mlops.evaluate`` exits 0 for the registered trained model, and 1
  (naming both thresholds) for a model that fails them;
* ``mlops.drift`` exits 0 and writes a report whose ``drifted`` flag is
  False for the untouched reference batch and True for the perturbed one;
* ``mlops.retrain`` exits 3 on the clean batch (nothing to do) and 0 on
  the drifted batch, registering the **next** registry version.

Everything runs inside ``tmp_path`` (no network, no repo pollution).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops import config
from mlops.data import load_breast_cancer, split_data
from mlops.registry import REGISTRY_FILENAME, register
from mlops.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture()
def workdir(tmp_path) -> Path:
    """Sandbox with clean + deliberately perturbed incoming batches."""
    X_ref, y_ref = _reference_split()
    np.savez(tmp_path / "clean.npz", X=X_ref, y=y_ref)
    np.savez(tmp_path / "perturbed.npz", X=X_ref * 5.0 + 10.0, y=y_ref)
    return tmp_path


def _reference_split():
    X, y, _ = load_breast_cancer()
    X_ref, _, _, y_ref, _, _ = split_data(X, y, seed=config.SEED)
    return X_ref, y_ref


def _train_and_register(artifacts: Path) -> dict:
    """M1 train step: fit the canonical model and register version 1."""
    run = run_training(config.SEED)
    return register(
        run["model"],
        run["metrics"],
        run["feature_names"],
        run["X_train"],
        run["y_train"],
        artifacts_dir=artifacts,
    )


def _registry_versions(artifacts: Path) -> list[int]:
    payload = json.loads(
        (artifacts / REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    return [entry["version"] for entry in payload["entries"]]


def test_full_cycle_train_evaluate_drift_retrain(workdir):
    artifacts = workdir / "artifacts"

    # -- step 1: train (M1 layer) -> registry version 1
    entry = _train_and_register(artifacts)
    assert entry["version"] == 1

    # -- step 2: evaluation gate passes for the registered model
    ev = _run_module(
        "mlops.evaluate", ["--artifacts-dir", str(artifacts)], cwd=workdir
    )
    assert ev.returncode == 0, ev.stdout + ev.stderr
    assert "evaluation gate PASSED" in ev.stdout

    # -- step 3a: drift on the untouched reference batch -> no drift
    dr_clean = _run_module(
        "mlops.drift",
        ["--batch", "clean.npz", "--report", "drift_clean.json"],
        cwd=workdir,
    )
    assert dr_clean.returncode == 0, dr_clean.stdout + dr_clean.stderr
    clean_report = json.loads(
        (workdir / "drift_clean.json").read_text(encoding="utf-8")
    )
    assert clean_report["drifted"] is False
    assert clean_report["mean_psi"] == pytest.approx(0.0, abs=1e-12)
    assert len(clean_report["psi"]) == config.EXPECTED_N_FEATURES

    # -- step 3b: drift on the deliberately perturbed batch -> drift
    dr_pert = _run_module(
        "mlops.drift",
        ["--batch", "perturbed.npz", "--report", "drift_pert.json"],
        cwd=workdir,
    )
    assert dr_pert.returncode == 0, dr_pert.stdout + dr_pert.stderr
    pert_report = json.loads(
        (workdir / "drift_pert.json").read_text(encoding="utf-8")
    )
    assert pert_report["drifted"] is True
    assert pert_report["mean_psi"] > config.PSI_MEAN_THRESHOLD
    assert pert_report["max_psi"] > config.PSI_MAX_THRESHOLD

    # -- step 4a: retrain on the clean batch -> exit 3, registry untouched
    rt_clean = _run_module(
        "mlops.retrain",
        [
            "--batch", "clean.npz",
            "--report", "retrain_clean.json",
            "--artifacts-dir", str(artifacts),
        ],
        cwd=workdir,
    )
    assert rt_clean.returncode == 3, rt_clean.stdout + rt_clean.stderr
    assert _registry_versions(artifacts) == [1]

    # -- step 4b: retrain on the drifted batch -> exit 0, next version
    rt_pert = _run_module(
        "mlops.retrain",
        [
            "--batch", "perturbed.npz",
            "--report", "retrain_pert.json",
            "--artifacts-dir", str(artifacts),
        ],
        cwd=workdir,
    )
    assert rt_pert.returncode == 0, rt_pert.stdout + rt_pert.stderr
    assert _registry_versions(artifacts) == [1, 2]
    assert (artifacts / "model-v2.joblib").is_file()

    # the retrain's own drift report agrees with the drift module's
    retrain_report = json.loads(
        (workdir / "retrain_pert.json").read_text(encoding="utf-8")
    )
    assert retrain_report["drifted"] is pert_report["drifted"] is True
    assert retrain_report["psi"] == pytest.approx(pert_report["psi"])


def test_evaluate_module_fails_loudly_for_a_failing_model(workdir):
    """Documented non-zero exit prints exactly which threshold failed."""
    X_ref, y_ref = _reference_split()
    bad = workdir / "badart"
    model = DummyClassifier(strategy="constant", constant=0).fit(X_ref, y_ref)
    register(
        model,
        {"accuracy": 0.0, "roc_auc": 0.0},
        ["x"] * config.EXPECTED_N_FEATURES,
        X_ref,
        y_ref,
        artifacts_dir=bad,
    )
    proc = _run_module(
        "mlops.evaluate", ["--artifacts-dir", str(bad)], cwd=workdir
    )
    assert proc.returncode == 1, proc.stdout
    assert "MIN_TEST_ROC_AUC" in proc.stdout
    assert "MIN_TEST_ACCURACY" in proc.stdout

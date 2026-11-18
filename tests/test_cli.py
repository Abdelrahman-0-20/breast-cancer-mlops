"""M3 CLI tests: unified ``python3 -m mlops`` dispatch + exit-code API.

The charter wires one entry point over the existing modules::

    python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]

These tests pin the contract from three angles:

* **parsing** — ``--help`` lists every command and its options;
  a missing/unknown command or a missing ``--batch`` exits 2
  (argparse's usage-error code);
* **dispatch** — each subcommand forwards ``--seed``/``--batch`` to the
  existing module's ``main()`` and propagates its exit code (the
  behavior stays in the modules, never re-implemented here);
* **end to end** — the documented cycle ``train -> evaluate -> drift ->
  retrain`` through ``python3 -m mlops ...`` with the charter's exit
  codes (0 = gate passed/retrained, 1 = gate failed, 3 = no drift), plus
  the module entry points (``python3 -m mlops.evaluate|drift|retrain``)
  staying invokable.

Everything runs in ``tmp_path`` (no network, no repo pollution).
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
from mlops.cli import main
from mlops.data import load_breast_cancer, split_data

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return env


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Invoke ``python3 -m mlops ...`` exactly as documented."""
    return subprocess.run(
        [sys.executable, "-m", "mlops", *args],
        cwd=cwd,
        env=_module_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


def _run_module(
    module: str, args: list[str], cwd: Path
) -> subprocess.CompletedProcess:
    """Invoke a module entry point (``python3 -m mlops.<module> ...``)."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=_module_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture()
def workdir(tmp_path) -> Path:
    """Sandbox cwd with clean + deliberately perturbed incoming batches."""
    X, y, _ = load_breast_cancer()
    X_ref, _, _, y_ref, _, _ = split_data(X, y, seed=config.SEED)
    np.savez(tmp_path / "clean.npz", X=X_ref, y=y_ref)
    np.savez(tmp_path / "perturbed.npz", X=X_ref * 5.0 + 10.0, y=y_ref)
    return tmp_path


def _registry_versions(artifacts: Path) -> list[int]:
    payload = json.loads(
        (artifacts / "registry.json").read_text(encoding="utf-8")
    )
    return [entry["version"] for entry in payload["entries"]]


# ------------------------------------------------------------ parsing / usage


def test_help_lists_every_charter_command(workdir):
    proc = _run_cli(["--help"], cwd=workdir)
    assert proc.returncode == 0, proc.stderr
    for command in ("train", "evaluate", "drift", "retrain"):
        assert command in proc.stdout


def test_subcommand_help_documents_seed_and_batch(workdir):
    train_help = _run_cli(["train", "--help"], cwd=workdir)
    assert train_help.returncode == 0, train_help.stderr
    assert "--seed" in train_help.stdout

    drift_help = _run_cli(["drift", "--help"], cwd=workdir)
    assert drift_help.returncode == 0, drift_help.stderr
    assert "--batch" in drift_help.stdout
    assert "--seed" in drift_help.stdout

    retrain_help = _run_cli(["retrain", "--help"], cwd=workdir)
    assert retrain_help.returncode == 0, retrain_help.stderr
    assert "--batch" in retrain_help.stdout


def test_missing_command_exits_2(workdir):
    proc = _run_cli([], cwd=workdir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "required" in proc.stderr


def test_unknown_command_exits_2(workdir):
    proc = _run_cli(["frobnicate"], cwd=workdir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "invalid choice" in proc.stderr


def test_drift_without_batch_exits_2(workdir):
    proc = _run_cli(["drift"], cwd=workdir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--batch" in proc.stderr


# ------------------------------------------------------------------ dispatch


def test_train_registers_version_1(workdir, monkeypatch, capsys):
    monkeypatch.chdir(workdir)
    assert main(["train", "--seed", "42"]) == 0
    artifacts = workdir / "artifacts"
    assert (artifacts / "model-v1.joblib").is_file()
    assert (artifacts / "registry.json").is_file()
    assert _registry_versions(artifacts) == [1]

    entry = json.loads(
        (artifacts / "registry.json").read_text(encoding="utf-8")
    )["entries"][0]
    assert entry["metrics"]["seed"] == 42
    assert 0.0 <= entry["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= entry["metrics"]["roc_auc"] <= 1.0

    out = capsys.readouterr().out
    assert "model-v1.joblib" in out


def test_train_forwards_requested_seed(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    assert main(["train", "--seed", "7"]) == 0
    entry = json.loads(
        (workdir / "artifacts" / "registry.json").read_text(encoding="utf-8")
    )["entries"][0]
    assert entry["metrics"]["seed"] == 7


def test_evaluate_subcommand_gates_registered_model(
    workdir, monkeypatch, capsys
):
    monkeypatch.chdir(workdir)
    assert main(["train", "--seed", "42"]) == 0
    assert main(["evaluate"]) == 0
    assert "evaluation gate PASSED" in capsys.readouterr().out


def test_evaluate_dispatch_forwards_seed(monkeypatch):
    import mlops.evaluate as evaluate_module

    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 5  # arbitrary non-zero code must propagate unchanged

    monkeypatch.setattr(evaluate_module, "main", fake_main)
    assert main(["evaluate", "--seed", "7"]) == 5
    assert captured["argv"] == ["--seed", "7"]


def test_drift_dispatch_forwards_batch_and_seed(workdir, monkeypatch):
    import mlops.drift as drift_module

    batch = workdir / "clean.npz"
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(drift_module, "main", fake_main)
    assert main(["drift", "--batch", str(batch), "--seed", "9"]) == 0
    assert captured["argv"] == [
        "--batch", str(batch), "--seed", "9",
    ]


def test_retrain_dispatch_forwards_batch_and_seed(workdir, monkeypatch):
    import mlops.retrain as retrain_module

    batch = workdir / "perturbed.npz"
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 3

    monkeypatch.setattr(retrain_module, "main", fake_main)
    assert main(["retrain", "--batch", str(batch)]) == 3
    assert captured["argv"] == [
        "--batch", str(batch), "--seed", str(config.SEED),
    ]


# ---------------------------------------------------- in-process behaviors


def test_drift_subcommand_writes_report(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    assert main(["drift", "--batch", "clean.npz"]) == 0
    report = json.loads(
        (workdir / "drift_report.json").read_text(encoding="utf-8")
    )
    assert report["drifted"] is False
    assert report["mean_psi"] == pytest.approx(0.0, abs=1e-12)
    assert report["thresholds"]["mean_psi"] == config.PSI_MEAN_THRESHOLD
    assert report["thresholds"]["max_psi"] == config.PSI_MAX_THRESHOLD


def test_retrain_clean_batch_exits_3(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    assert main(["retrain", "--batch", "clean.npz"]) == 3
    assert not (workdir / "artifacts").exists()  # nothing to do
    report = json.loads(
        (workdir / "drift_report.json").read_text(encoding="utf-8")
    )
    assert report["drifted"] is False


def test_retrain_perturbed_batch_registers_next_version(
    workdir, monkeypatch
):
    monkeypatch.chdir(workdir)
    assert main(["train", "--seed", "42"]) == 0
    assert main(["retrain", "--batch", "perturbed.npz"]) == 0
    artifacts = workdir / "artifacts"
    assert _registry_versions(artifacts) == [1, 2]
    assert (artifacts / "model-v2.joblib").is_file()


# --------------------------------------------------- documented entry points


def test_end_to_end_cycle_via_python_m_mlops(workdir):
    """The full documented cycle through the unified CLI, exit codes and all."""
    train = _run_cli(["train", "--seed", "42"], cwd=workdir)
    assert train.returncode == 0, train.stdout + train.stderr
    assert (workdir / "artifacts" / "model-v1.joblib").is_file()
    assert (workdir / "artifacts" / "registry.json").is_file()

    evaluate = _run_cli(["evaluate"], cwd=workdir)
    assert evaluate.returncode == 0, evaluate.stdout + evaluate.stderr
    assert "evaluation gate PASSED" in evaluate.stdout

    drift_clean = _run_cli(["drift", "--batch", "clean.npz"], cwd=workdir)
    assert drift_clean.returncode == 0, drift_clean.stdout + drift_clean.stderr
    clean_report = json.loads(
        (workdir / "drift_report.json").read_text(encoding="utf-8")
    )
    assert clean_report["drifted"] is False

    drift_pert = _run_cli(["drift", "--batch", "perturbed.npz"], cwd=workdir)
    assert drift_pert.returncode == 0, drift_pert.stdout + drift_pert.stderr
    pert_report = json.loads(
        (workdir / "drift_report.json").read_text(encoding="utf-8")
    )
    assert pert_report["drifted"] is True

    retrain_clean = _run_cli(["retrain", "--batch", "clean.npz"], cwd=workdir)
    assert retrain_clean.returncode == 3, (
        retrain_clean.stdout + retrain_clean.stderr
    )

    retrain_pert = _run_cli(
        ["retrain", "--batch", "perturbed.npz"], cwd=workdir
    )
    assert retrain_pert.returncode == 0, (
        retrain_pert.stdout + retrain_pert.stderr
    )
    artifacts = workdir / "artifacts"
    assert _registry_versions(artifacts) == [1, 2]
    assert (artifacts / "model-v2.joblib").is_file()

    retrain_clean_again = _run_cli(
        ["retrain", "--batch", "clean.npz"], cwd=workdir
    )
    assert retrain_clean_again.returncode == 3, (
        retrain_clean_again.stdout + retrain_clean_again.stderr
    )

    no_command = _run_cli([], cwd=workdir)
    assert no_command.returncode == 2, no_command.stdout + no_command.stderr


def test_module_entry_points_still_invokable(workdir):
    """``python3 -m mlops.<module>`` keeps working alongside the CLI."""
    ev = _run_module("mlops.evaluate", [], cwd=workdir)
    assert ev.returncode == 0, ev.stdout + ev.stderr
    assert "evaluation gate PASSED" in ev.stdout

    dr = _run_module(
        "mlops.drift", ["--batch", "clean.npz"], cwd=workdir
    )
    assert dr.returncode == 0, dr.stdout + dr.stderr
    assert (workdir / "drift_report.json").is_file()

    rt = _run_module(
        "mlops.retrain", ["--batch", "clean.npz"], cwd=workdir
    )
    assert rt.returncode == 3, rt.stdout + rt.stderr

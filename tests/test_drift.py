"""M2 drift-monitor tests: hand-computed PSI fixture + verdict boundaries.

Charter contract (README): Population Stability Index per feature (train
reference vs incoming batch), ``drift_report.json`` written, **drifted
iff mean PSI > 0.15 OR max PSI > 0.30** with thresholds in config.

The PSI fixture is derived by hand: reference ``arange(100)`` with
``n_bins=4`` has interior quantile edges 24.75 / 49.5 / 74.25, hence
exactly 25 values per bin (expected shares 0.25 each); the candidate is
built from four constant plateaus giving shares 0.4 / 0.1 / 0.25 / 0.25,
so PSI = 0.15*ln(1.6) + 0.15*ln(2.5). Verdict boundary tests exploit
strict ``>``: a value exactly at the threshold must not drift, one ulp
above (``math.nextafter``) must.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mlops import config
from mlops.data import load_breast_cancer, split_data
from mlops.drift import (
    PSI_EPS,
    PSI_N_BINS,
    bin_edges,
    bin_proportions,
    build_report,
    load_batch,
    main,
    psi_feature,
    psi_per_feature,
    verdict,
    write_report,
)

# ------------------------------------------------------------ PSI math (hand)


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    """100 evenly spaced values: quantile bins cut at 24.75/49.5/74.25."""
    return np.arange(100, dtype=np.float64)


def test_config_thresholds_match_charter_values():
    assert config.PSI_MEAN_THRESHOLD == 0.15
    assert config.PSI_MAX_THRESHOLD == 0.30


def test_psi_parameters_are_configured_not_verdicts():
    assert PSI_N_BINS == 10
    assert PSI_EPS == pytest.approx(1e-6)


def test_bin_edges_are_open_ended_and_strictly_increasing(reference):
    edges = bin_edges(reference, n_bins=4)
    assert len(edges) == 5
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)
    assert list(edges[1:-1]) == [24.75, 49.5, 74.25]


def test_bin_proportions_hand_computed(reference):
    edges = bin_edges(reference, n_bins=4)
    shares = bin_proportions(reference, edges)
    # 0..24 | 25..49 | 50..74 | 75..99 -> exactly 25 values each
    assert shares == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_psi_hand_computed_fixture(reference):
    """Four plateaus -> shares 0.4/0.1/0.25/0.25 vs 0.25 each.

    PSI = (0.4-0.25)*ln(0.4/0.25) + (0.1-0.25)*ln(0.1/0.25)
        = 0.15*ln(1.6) + 0.15*ln(2.5)
    """
    candidate = np.concatenate(
        [
            np.full(40, 10.0),  # bin 0  (10.0 <= 24.75)
            np.full(10, 30.0),  # bin 1  (24.75 < 30.0 <= 49.5)
            np.full(25, 60.0),  # bin 2  (49.5 < 60.0 <= 74.25)
            np.full(25, 90.0),  # bin 3  (90.0 > 74.25)
        ]
    )
    assert bin_proportions(candidate, bin_edges(reference, 4)) == pytest.approx(
        [0.4, 0.1, 0.25, 0.25]
    )
    hand = 0.15 * math.log(1.6) + 0.15 * math.log(2.5)
    assert psi_feature(reference, candidate, n_bins=4) == pytest.approx(
        hand, abs=1e-12
    )
    assert hand == pytest.approx(0.20794415416798362, abs=1e-15)


def test_psi_of_identical_distributions_is_zero(reference):
    assert psi_feature(reference, reference.copy()) == pytest.approx(0.0, abs=1e-12)


def test_psi_with_empty_bin_stays_finite_and_hand_computed(reference):
    """All mass in bin 0 -> empty bins are smoothed by PSI_EPS."""
    candidate = np.zeros(100, dtype=np.float64)
    edges = bin_edges(reference, n_bins=4)
    assert bin_proportions(candidate, edges) == pytest.approx([1.0, 0, 0, 0])
    hand = 0.75 * math.log(4.0) + 3 * ((PSI_EPS - 0.25) * math.log(PSI_EPS / 0.25))
    value = psi_feature(reference, candidate, n_bins=4)
    assert math.isfinite(value)
    assert value == pytest.approx(hand, rel=1e-12)


def test_psi_is_non_negative_for_arbitrary_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=500)
    candidate = reference * 3.0 + 7.0
    assert psi_feature(reference, candidate) >= 0.0


def test_psi_per_feature_shape_and_column_isolation():
    X_ref = np.arange(3000, dtype=np.float64).reshape(100, 30)
    X_cand = X_ref.copy()
    X_cand[:, 5] = X_cand[:, 5] * 100.0 + 1000.0  # only feature 5 moves
    psi = psi_per_feature(X_ref, X_cand)
    assert psi.shape == (30,)
    assert psi[5] > config.PSI_MAX_THRESHOLD
    others = np.delete(psi, 5)
    assert others == pytest.approx(np.zeros(29), abs=1e-12)


def test_psi_per_feature_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="feature count mismatch"):
        psi_per_feature(np.zeros((10, 30)), np.zeros((10, 29)))
    with pytest.raises(ValueError, match="2-D"):
        psi_per_feature(np.zeros(10), np.zeros(10))


# ---------------------------------------------------------- verdict (strict >)


def test_verdict_at_exact_mean_threshold_is_not_drifted():
    drifted, mean_psi, max_psi = verdict(np.array([config.PSI_MEAN_THRESHOLD]))
    assert (drifted, mean_psi, max_psi) == (False, 0.15, 0.15)


def test_verdict_one_ulp_above_mean_threshold_drifts():
    value = math.nextafter(config.PSI_MEAN_THRESHOLD, math.inf)
    drifted, mean_psi, max_psi = verdict(np.array([value]))
    assert drifted is True
    assert mean_psi > config.PSI_MEAN_THRESHOLD
    assert max_psi <= config.PSI_MAX_THRESHOLD  # mean criterion triggered


def test_verdict_at_exact_max_threshold_is_not_drifted():
    # one feature exactly at the max threshold, mean far below it
    drifted, mean_psi, max_psi = verdict(
        np.array([config.PSI_MAX_THRESHOLD, 0.0, 0.0, 0.0])
    )
    assert drifted is False
    assert mean_psi == pytest.approx(0.075)
    assert max_psi == config.PSI_MAX_THRESHOLD


def test_verdict_one_ulp_above_max_threshold_drifts():
    value = math.nextafter(config.PSI_MAX_THRESHOLD, math.inf)
    drifted, mean_psi, max_psi = verdict(np.array([value, 0.0, 0.0, 0.0]))
    assert drifted is True
    assert mean_psi <= config.PSI_MEAN_THRESHOLD  # max criterion triggered
    assert max_psi > config.PSI_MAX_THRESHOLD


def test_verdict_clear_cases():
    assert verdict(np.zeros(30))[0] is False
    assert verdict(np.full(30, 12.0))[0] is True
    with pytest.raises(ValueError, match="not be empty"):
        verdict(np.array([]))


def test_verdict_reads_thresholds_from_config(monkeypatch):
    psi = np.full(4, 0.20)  # drifted under the charter's 0.15 mean threshold
    assert verdict(psi)[0] is True
    monkeypatch.setattr(config, "PSI_MEAN_THRESHOLD", 0.5)
    assert verdict(psi)[0] is False


# ------------------------------------------------------------ report contract


def test_build_report_contract():
    X_ref = np.arange(3000, dtype=float).reshape(100, 30)
    X_cand = X_ref + 1_000_000.0  # every value leaves the reference range
    names = [f"f{i}" for i in range(30)]
    report = build_report(X_ref, X_cand, names, batch="b.npz")
    assert report["drifted"] is True
    assert len(report["psi"]) == 30
    assert report["feature_names"] == names
    assert report["thresholds"] == {
        "mean_psi": config.PSI_MEAN_THRESHOLD,
        "max_psi": config.PSI_MAX_THRESHOLD,
    }
    assert report["n_reference"] == 100
    assert report["n_batch"] == 100
    assert report["batch"] == "b.npz"
    assert report["mean_psi"] == pytest.approx(float(np.mean(report["psi"])))
    assert report["max_psi"] == pytest.approx(float(np.max(report["psi"])))


def test_build_report_rejects_feature_name_mismatch():
    X = np.arange(3000, dtype=float).reshape(100, 30)
    with pytest.raises(ValueError, match="feature name count"):
        build_report(X, X, ["only", "two"])


def test_write_report_roundtrips_json(tmp_path):
    report = build_report(
        np.arange(100, dtype=float).reshape(100, 1),
        np.arange(100, dtype=float).reshape(100, 1),
        ["f0"],
        batch=None,
    )
    path = write_report(report, tmp_path / "sub" / "drift_report.json")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_load_batch_roundtrip_and_contract(tmp_path):
    X = np.arange(150, dtype=float).reshape(50, 3)
    y = np.arange(50) % 2
    path = tmp_path / "batch.npz"
    np.savez(path, X=X, y=y)
    X_loaded, y_loaded = load_batch(path)
    assert np.array_equal(X_loaded, X)
    assert np.array_equal(y_loaded, y)

    bad = tmp_path / "bad.npz"
    np.savez(bad, X=X)
    with pytest.raises(ValueError, match="must contain"):
        load_batch(bad)


# --------------------------------------------------------------- main() (CLI)


def _reference_split():
    X, y, feature_names = load_breast_cancer()
    X_ref, _, _, y_ref, _, _ = split_data(X, y, seed=config.SEED)
    return X_ref, y_ref, feature_names


def test_main_clean_batch_writes_default_report_and_exits_zero(tmp_path, monkeypatch):
    X_ref, y_ref, _ = _reference_split()
    batch = tmp_path / "clean.npz"
    np.savez(batch, X=X_ref, y=y_ref)
    monkeypatch.chdir(tmp_path)
    assert main(["--batch", str(batch)]) == 0
    report = json.loads(
        (tmp_path / "drift_report.json").read_text(encoding="utf-8")
    )
    assert report["drifted"] is False
    assert report["mean_psi"] == pytest.approx(0.0, abs=1e-12)
    assert report["max_psi"] == pytest.approx(0.0, abs=1e-12)
    assert len(report["psi"]) == 30


def test_main_perturbed_batch_reports_drift_and_exits_zero(tmp_path):
    X_ref, y_ref, _ = _reference_split()
    batch = tmp_path / "perturbed.npz"
    np.savez(batch, X=X_ref * 5.0 + 10.0, y=y_ref)
    report_path = tmp_path / "report.json"
    assert main(["--batch", str(batch), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["drifted"] is True
    assert report["mean_psi"] > config.PSI_MEAN_THRESHOLD

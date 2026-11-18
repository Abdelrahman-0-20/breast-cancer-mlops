"""Drift monitor (milestone M2): Population Stability Index per feature.

Compares the seeded **train reference** against an incoming batch and
writes a drift report (``drift_report.json`` by default, or ``--report``).

PSI per feature uses ``PSI_N_BINS`` reference-quantile bins with open
outer bins and ``PSI_EPS`` smoothing for empty bins. The verdict comes
straight from ``mlops/config.py``: **drifted iff mean PSI >
``PSI_MEAN_THRESHOLD`` OR max PSI > ``PSI_MAX_THRESHOLD``**.

Exit codes: the monitor itself always exits 0 — the verdict lives in the
report file (``mlops/retrain.py`` maps it onto exit codes 0/3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mlops import config
from mlops.data import load_breast_cancer, split_data, validate_frame

__all__ = [
    "PSI_N_BINS",
    "PSI_EPS",
    "bin_edges",
    "bin_proportions",
    "psi_feature",
    "psi_per_feature",
    "verdict",
    "build_report",
    "write_report",
    "load_batch",
    "main",
]

# Histogram granularity per feature (a parameter, not a verdict threshold).
PSI_N_BINS = 10
# Zero-proportion smoothing so PSI stays finite when a bin is empty.
PSI_EPS = 1e-6


def bin_edges(reference: np.ndarray, n_bins: int = PSI_N_BINS) -> np.ndarray:
    """Quantile bin edges of ``reference``, open-ended at both extremes.

    Returns ``[-inf, interior percentiles..., +inf]`` — strictly
    increasing, so every value (including batch values outside the
    reference range) lands in exactly one bin.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if ref.size == 0:
        raise ValueError("reference feature must not be empty")
    quantiles = np.percentile(ref, np.linspace(0.0, 100.0, n_bins + 1))
    interior = np.unique(quantiles)[1:-1]
    return np.concatenate(
        (np.array([-np.inf]), interior, np.array([np.inf]))
    )


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of ``values`` falling in each ``[edges[i], edges[i+1])`` bin."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size == 0:
        raise ValueError("values must not be empty")
    index = np.searchsorted(edges, vals, side="right") - 1
    index = np.clip(index, 0, len(edges) - 2)
    counts = np.bincount(index, minlength=len(edges) - 1).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("no values to bin")
    return counts / total


def psi_feature(
    reference: np.ndarray, candidate: np.ndarray, n_bins: int = PSI_N_BINS
) -> float:
    """Population Stability Index of one candidate feature vs reference.

    PSI = sum over bins of ``(actual - expected) * ln(actual / expected)``
    with ``expected``/``actual`` clipped at ``PSI_EPS`` so empty bins stay
    finite. Non-negative by construction; 0.0 for identical shares.
    """
    edges = bin_edges(reference, n_bins=n_bins)
    expected = np.maximum(bin_proportions(reference, edges), PSI_EPS)
    actual = np.maximum(bin_proportions(candidate, edges), PSI_EPS)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_per_feature(
    X_reference: np.ndarray,
    X_candidate: np.ndarray,
    n_bins: int = PSI_N_BINS,
) -> np.ndarray:
    """PSI for every feature column; one value per column, in order."""
    ref = np.asarray(X_reference, dtype=np.float64)
    cand = np.asarray(X_candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2:
        raise ValueError(
            f"expected 2-D feature matrices, got {ref.ndim}-D and "
            f"{cand.ndim}-D"
        )
    if ref.shape[1] != cand.shape[1]:
        raise ValueError(
            f"feature count mismatch: {ref.shape[1]} != {cand.shape[1]}"
        )
    return np.array(
        [
            psi_feature(ref[:, j], cand[:, j], n_bins=n_bins)
            for j in range(ref.shape[1])
        ],
        dtype=np.float64,
    )


def verdict(psi_values: np.ndarray) -> tuple[bool, float, float]:
    """``(drifted, mean_psi, max_psi)`` using the config thresholds.

    Drifted iff mean PSI > ``PSI_MEAN_THRESHOLD`` OR max PSI >
    ``PSI_MAX_THRESHOLD`` — both read from ``mlops/config.py`` here, so
    the comparison follows the configuration exactly (strict ``>``).
    """
    psi = np.asarray(psi_values, dtype=np.float64).ravel()
    if psi.size == 0:
        raise ValueError("PSI values must not be empty")
    mean_psi = float(np.mean(psi))
    max_psi = float(np.max(psi))
    drifted = bool(
        mean_psi > config.PSI_MEAN_THRESHOLD
        or max_psi > config.PSI_MAX_THRESHOLD
    )
    return drifted, mean_psi, max_psi


def build_report(
    X_reference: np.ndarray,
    X_candidate: np.ndarray,
    feature_names: list[str],
    *,
    batch: str | None = None,
) -> dict[str, Any]:
    """Drift report dict: verdict, per-feature PSI and the thresholds."""
    psi = psi_per_feature(X_reference, X_candidate)
    if len(feature_names) != psi.size:
        raise ValueError(
            f"feature name count {len(feature_names)} != {psi.size} features"
        )
    drifted, mean_psi, max_psi = verdict(psi)
    return {
        "drifted": drifted,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "psi": [float(value) for value in psi],
        "feature_names": list(feature_names),
        "n_bins": PSI_N_BINS,
        "thresholds": {
            "mean_psi": float(config.PSI_MEAN_THRESHOLD),
            "max_psi": float(config.PSI_MAX_THRESHOLD),
        },
        "n_reference": int(len(X_reference)),
        "n_batch": int(len(X_candidate)),
        "batch": batch,
    }


def write_report(report: dict[str, Any], path: Path | str) -> Path:
    """Write the drift report as JSON (deterministic, no timestamps)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_batch(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load an incoming batch saved as an ``.npz`` with ``X`` and ``y``."""
    with np.load(Path(path)) as data:
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(
                f"{path}: batch .npz must contain 'X' and 'y' arrays"
            )
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"])
    return X, y


def main(argv: list[str] | None = None) -> int:
    """CLI entry: always exits 0; the verdict is written to the report."""
    parser = argparse.ArgumentParser(
        prog="python3 -m mlops.drift",
        description=(
            "Population Stability Index per feature (train reference vs "
            "incoming batch); writes a drift report whose 'drifted' flag "
            "is true iff mean PSI > "
            f"{config.PSI_MEAN_THRESHOLD} or max PSI > "
            f"{config.PSI_MAX_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help=".npz incoming batch with X and y arrays",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("drift_report.json"),
        help="where to write the drift report (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="seed of the reference train split (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    X, y, feature_names = load_breast_cancer()
    validate_frame(X, y)
    X_ref, _, _, _, _, _ = split_data(X, y, seed=args.seed)

    X_batch, y_batch = load_batch(args.batch)
    validate_frame(X_batch, y_batch)

    report = build_report(
        X_ref, X_batch, feature_names, batch=str(args.batch)
    )
    write_report(report, args.report)

    print(
        f"drift report: {args.report} "
        f"(mean PSI {report['mean_psi']:.4f}, "
        f"max PSI {report['max_psi']:.4f}, "
        f"{len(report['psi'])} features)"
    )
    if report["drifted"]:
        print(
            "DRIFT DETECTED: mean PSI > "
            f"{config.PSI_MEAN_THRESHOLD} or max PSI > "
            f"{config.PSI_MAX_THRESHOLD}"
        )
    else:
        print(
            "no drift: mean PSI <= "
            f"{config.PSI_MEAN_THRESHOLD} and max PSI <= "
            f"{config.PSI_MAX_THRESHOLD}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

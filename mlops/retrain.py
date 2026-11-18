"""Auto-retrain (milestone M2): drift-gated retraining with an exit-code API.

Exit codes are an API (charter):

* **0** — drifted, retrained on reference+incoming data and registered
  the next registry version;
* **3** — no drift, nothing to do (the drift report is still written).

The drift verdict is computed exactly as in ``mlops/drift.py`` (same
report schema, thresholds from ``mlops/config.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from mlops import config
from mlops.data import load_breast_cancer, split_data, validate_frame
from mlops.drift import build_report, load_batch, write_report
from mlops.registry import register
from mlops.train import build_model

__all__ = ["EXIT_RETRAINED", "EXIT_NO_DRIFT", "main"]

EXIT_RETRAINED = 0  # drifted -> retrained + registered the next version
EXIT_NO_DRIFT = 3  # no drift -> nothing to do (distinct code, an API)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: 0 after a retrain, 3 when there is no drift."""
    parser = argparse.ArgumentParser(
        prog="python3 -m mlops.retrain",
        description=(
            "Drift-gated auto-retrain: exit "
            f"{EXIT_RETRAINED} after retraining on reference+incoming "
            f"data and registering the next version, exit "
            f"{EXIT_NO_DRIFT} when there is no drift (nothing to do)."
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
        "--artifacts-dir",
        type=Path,
        default=None,
        help="registry directory to register the next version into",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="seed of the reference split (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    X, y, feature_names = load_breast_cancer()
    validate_frame(X, y)
    X_ref, X_valid, _, y_ref, y_valid, _ = split_data(X, y, seed=args.seed)

    X_batch, y_batch = load_batch(args.batch)
    validate_frame(X_batch, y_batch)

    report = build_report(
        X_ref, X_batch, feature_names, batch=str(args.batch)
    )
    write_report(report, args.report)

    if not report["drifted"]:
        print(
            f"no drift (mean PSI {report['mean_psi']:.4f} <= "
            f"{config.PSI_MEAN_THRESHOLD}, max PSI "
            f"{report['max_psi']:.4f} <= {config.PSI_MAX_THRESHOLD}): "
            f"nothing to do, exit {EXIT_NO_DRIFT}"
        )
        return EXIT_NO_DRIFT

    # Drifted: retrain on reference + incoming, hold out the untouched
    # validation split for the recorded metrics, register the next version.
    X_train = np.concatenate([X_ref, X_batch], axis=0)
    y_train = np.concatenate([y_ref, y_batch], axis=0)
    validate_frame(X_train, y_train)

    model = build_model(seed=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "seed": int(args.seed),
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_reference": int(len(X_ref)),
        "n_incoming": int(len(X_batch)),
    }
    entry = register(
        model,
        metrics,
        feature_names,
        X_train,
        y_train,
        artifacts_dir=args.artifacts_dir,
    )
    print(
        f"drift detected (mean PSI {report['mean_psi']:.4f}, max PSI "
        f"{report['max_psi']:.4f}): retrained on reference+incoming "
        f"({len(X_train)} rows), registered version "
        f"{entry['version']} ({entry['artifact']}), exit {EXIT_RETRAINED}"
    )
    return EXIT_RETRAINED


if __name__ == "__main__":
    sys.exit(main())

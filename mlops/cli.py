"""Unified CLI (milestone M3): ``python3 -m mlops <command> ...``.

Charter contract (README, requirement 7)::

    python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]

This module only parses the command line and dispatches: every piece
of behavior (training, gating, PSI, thresholds, exit codes) lives in
the existing ``mlops/train.py``, ``mlops/evaluate.py``,
``mlops/drift.py`` and ``mlops/retrain.py``, whose own
``python3 -m mlops.<module>`` invocations keep working unchanged.

Exit codes are the underlying modules' API, preserved verbatim:

* ``train``    — 0 after training and registering the next version;
* ``evaluate`` — 0 iff every threshold in ``mlops/config.py`` is met,
  1 when one fails (the failing threshold is printed);
* ``drift``    — 0 always; the verdict lives in ``drift_report.json``;
* ``retrain``  — 0 = retrained, 3 = no drift (nothing to do);
* usage errors — 2 (argparse): missing/unknown command or option.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlops import config

__all__ = ["build_parser", "main"]

EXIT_USAGE = 2  # argparse's own exit code for usage errors


def _run_train(args: argparse.Namespace) -> int:
    """Fit the seeded model and register it as the next version."""
    from mlops import registry as registry_module
    from mlops import train as train_module

    run = train_module.run_training(seed=args.seed)
    entry = registry_module.register(
        run["model"],
        run["metrics"],
        run["feature_names"],
        run["X_train"],
        run["y_train"],
    )
    metrics = run["metrics"]
    artifact = registry_module.DEFAULT_ARTIFACTS_DIR / entry["artifact"]
    print(
        f"registered version {entry['version']} -> {artifact} "
        f"(seed {metrics['seed']}, validation accuracy "
        f"{metrics['accuracy']:.4f}, ROC-AUC {metrics['roc_auc']:.4f})"
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """Delegate to the evaluation gate module (its exit codes)."""
    from mlops import evaluate as evaluate_module

    return int(evaluate_module.main(["--seed", str(args.seed)]))


def _run_drift(args: argparse.Namespace) -> int:
    """Delegate to the drift monitor module (exit 0, verdict in JSON)."""
    from mlops import drift as drift_module

    return int(
        drift_module.main(
            ["--batch", str(args.batch), "--seed", str(args.seed)]
        )
    )


def _run_retrain(args: argparse.Namespace) -> int:
    """Delegate to the auto-retrain module (exit 0 retrained / 3 no drift)."""
    from mlops import retrain as retrain_module

    return int(
        retrain_module.main(
            ["--batch", str(args.batch), "--seed", str(args.seed)]
        )
    )


def _add_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="random seed of the split/model (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser: ``train | evaluate | drift | retrain``."""
    parser = argparse.ArgumentParser(
        prog="python3 -m mlops",
        description=(
            "Unified CLI for the breast-cancer MLOps pipeline: "
            "train -> evaluate -> drift -> retrain."
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="<command>"
    )

    train_parser = commands.add_parser(
        "train",
        help="train the seeded model and register the next version",
        description=(
            "StandardScaler + LogisticRegression on the seeded 70/15/15 "
            "split; writes artifacts/model-v{n}.joblib and updates "
            "artifacts/registry.json (exit 0)."
        ),
    )
    _add_seed(train_parser)
    train_parser.set_defaults(handler=_run_train)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="holdout evaluation gate (exit 0 iff thresholds are met)",
        description=(
            "Gate the latest registered model (or a fresh canonical run "
            "when none exists): exit 0 iff test ROC-AUC >= "
            f"{config.MIN_TEST_ROC_AUC} and accuracy >= "
            f"{config.MIN_TEST_ACCURACY} (thresholds from "
            "mlops/config.py), else 1."
        ),
    )
    _add_seed(evaluate_parser)
    evaluate_parser.set_defaults(handler=_run_evaluate)

    drift_parser = commands.add_parser(
        "drift",
        help="PSI drift report for an incoming batch (exit 0)",
        description=(
            "Population Stability Index per feature vs the seeded train "
            "reference; writes drift_report.json whose 'drifted' flag is "
            "true iff mean PSI > "
            f"{config.PSI_MEAN_THRESHOLD} or max PSI > "
            f"{config.PSI_MAX_THRESHOLD}. Always exits 0."
        ),
    )
    drift_parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help=".npz incoming batch with X and y arrays",
    )
    _add_seed(drift_parser)
    drift_parser.set_defaults(handler=_run_drift)

    retrain_parser = commands.add_parser(
        "retrain",
        help="drift-gated auto-retrain (exit 0 retrained / 3 no drift)",
        description=(
            "Retrain on reference+incoming data and register the next "
            "version when drifted (exit 0); exit 3 = no drift, "
            "nothing to do."
        ),
    )
    retrain_parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help=".npz incoming batch with X and y arrays",
    )
    _add_seed(retrain_parser)
    retrain_parser.set_defaults(handler=_run_retrain)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``) and dispatch.

    Usage errors make argparse exit 2; otherwise the exit code is the
    dispatched module's own.
    """
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())

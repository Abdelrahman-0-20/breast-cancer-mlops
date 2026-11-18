# breast-cancer-mlops

**A full MLOps project built end-to-end by an autonomous engineering pipeline** — this repo *is* the v1.0 case study of [mimo-agent-pipeline](https://github.com/Abdelrahman-0-20/mimo-agent-pipeline): a planner→builder→verifier relay with per-role model routing and verification gates, building train → evaluate → registry → drift → auto-retrain from an empty directory.

Dataset: **sklearn's built-in breast-cancer dataset** (zero network, fully deterministic with `--seed 42`).

---

## Requirements — every one of these must exist, work, and be tested

1. **Data layer** (`mlops/data.py`): load `sklearn.datasets.load_breast_cancer`, stratified split **70/15/15 with seed 42** (train/valid/test), and a `validate_frame()` contract check (expected feature count, no NaNs, target ∈ {0,1}).
2. **Training** (`mlops/train.py`): `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=seed)` — reports accuracy + ROC-AUC on the validation split; **two runs with the same seed produce byte-identical metrics** (enforced by a test).
3. **Model registry** (`mlops/registry.py`): `artifacts/model-v{n}.joblib` + `artifacts/registry.json` recording version, `trained_at`, metrics, feature names, and a sha256 of the training data; `latest()` always resolves the newest version.
4. **Evaluation gate** (`mlops/evaluate.py`): exits **0 iff** test ROC-AUC ≥ **0.95** and accuracy ≥ **0.90** (thresholds live in `mlops/config.py`, not inline); non-zero exit prints which threshold failed.
5. **Drift monitor** (`mlops/drift.py`): Population Stability Index per feature (train reference vs an incoming batch); writes `drift_report.json`; **drifted iff mean PSI > 0.15 OR max PSI > 0.30** (thresholds in config).
6. **Auto-retrain** (`mlops/retrain.py`): when drifted → retrains on reference+incoming, registers the next version; **exit 0 = retrained, exit 3 = no drift (nothing to do)** — distinct exit codes are an API.
7. **CLI** (`mlops/cli.py`): `python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]`.
8. **Tests** (`tests/`, pytest, deterministic): unit (PSI math, data contract, registry round-trip, gate thresholds, CLI parsing) + integration (train → evaluate → drift on a perturbed batch → retrain cycle). **Minimum 30 tests, exit 0.**
9. **CI**: `.github/workflows/ci.yml` running the same gates below.
10. **Docs**: this README (quickstart + architecture) plus `docs/ADR-001-seeded-determinism.md` and `docs/ADR-002-drift-thresholds.md` (decision + rejected alternatives, priced honestly).

## Gates — green after *every* milestone

```bash
./verify.sh    # compileall + pytest (exit 0) + ruff when installed
```

## Milestones — each one relay run, each one commit

| # | Scope |
|---|---|
| **M1** | data layer + training + registry + their tests |
| **M2** | evaluation gate + drift monitor + auto-retrain + their tests |
| **M3** | CLI wiring + CI workflow + README/ADRs + full final gates |

## Rules of the build

- Python 3 only (there is no bare `python` on this machine), dependencies limited to **scikit-learn, joblib, scipy, pytest** — nothing downloaded, ever.
- A milestone is done only when `./verify.sh` exits 0 — no stubs, no skipped-by-default tests, no weakened thresholds.
- Each milestone ends with one descriptive commit; the history is the case-study evidence.

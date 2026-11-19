# breast-cancer-mlops

**A full MLOps project built end-to-end by an autonomous engineering pipeline** — this repo *is* the v1.0 case study of [mimo-agent-pipeline](https://github.com/Abdelrahman-0-20/mimo-agent-pipeline): a planner→builder→verifier relay with per-role model routing and verification gates, building train → evaluate → registry → drift → auto-retrain from an empty directory.

Dataset: **sklearn's built-in breast-cancer dataset** (zero network, fully deterministic with `--seed 42`).

**Milestones: all three complete** — M1 `40b69fa`, M2 `864f9eb`, M3 (CLI + CI + docs + final gates). **133 tests, all green** (honest count: `python3 -m pytest --collect-only -q | grep -c '::'` → `133`).

---

## Quickstart — Python 3 only (there is no bare `python` on this machine)

```bash
# 1. the only allowed dependencies (numpy comes in with scikit-learn)
python3 -m pip install scikit-learn joblib scipy pytest

# 2. the gates — must exit 0 (compileall + the full test suite + ruff when installed)
./verify.sh

# 3. the unified CLI: python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]
python3 -m mlops train --seed 42     # fit + register artifacts/model-v1.joblib (exit 0)
python3 -m mlops evaluate            # exit 0 iff test ROC-AUC >= 0.95 and accuracy >= 0.90

# 4. build incoming batches (.npz with X and y) — zero network
python3 -c 'import numpy as np; from mlops.data import load_breast_cancer, split_data; X, y, _ = load_breast_cancer(); Xr, _, _, yr, _, _ = split_data(X, y, seed=42); np.savez("clean.npz", X=Xr, y=yr); np.savez("perturbed.npz", X=Xr * 5.0 + 10.0, y=yr)'

python3 -m mlops drift --batch clean.npz        # exit 0, writes drift_report.json (drifted: false)
python3 -m mlops retrain --batch clean.npz      # exit 3 — no drift, nothing to do
python3 -m mlops retrain --batch perturbed.npz  # exit 0 — retrained, artifacts/model-v2.joblib
```

The four modules keep their own documented entry points as well — the CLI dispatches to exactly this code:

```bash
python3 -m mlops.evaluate
python3 -m mlops.drift --batch clean.npz
python3 -m mlops.retrain --batch perturbed.npz
# training deliberately has no standalone module entry point:
# it goes through the unified CLI (python3 -m mlops train --seed 42)
```

### CLI exit codes — an API, not an accident

| Invocation | Exit 0 means | Other codes |
|---|---|---|
| `python3 -m mlops train [--seed N]` | trained and registered the next version | `2` = usage error |
| `python3 -m mlops evaluate [--seed N]` | test ROC-AUC ≥ 0.95 **and** accuracy ≥ 0.90 | `1` = a threshold failed (which one is printed), `2` = usage |
| `python3 -m mlops drift --batch PATH [--seed N]` | report written — the verdict lives in `drift_report.json` | `2` = usage (missing `--batch`) |
| `python3 -m mlops retrain --batch PATH [--seed N]` | drifted → retrained on reference+incoming, registered next version | `3` = no drift (nothing to do), `2` = usage |
| `python3 -m mlops` (no command) | — | `2` = usage |

---

## Architecture

```
        data ──► train ──► registry ──► evaluate ──► drift ──► retrain
         │         │           │            │          │           │
   load +      scaler +    model-v{n}.   exit 0 iff   PSI per   exit 0 = retrained
   seeded      logreg,     joblib +      thresholds   feature,  exit 3 = no drift
   70/15/15    seeded,     registry.json are met,     report ──► registers v{n+1}
   split       byte-ident  (version,     else exit 1  drift      back into registry
               metrics     metrics, sha               iff mean   when drifted
                           of data)                   > 0.15 or
                                                       max > 0.30

        ┌──────────────────────────────────────────────────────────────┐
        │ retrain --(drifted)──► registry: next version v{n+1}         │
        │ retrain --(no drift)──► exit 3, registry untouched           │
        └──────────────────────────────────────────────────────────────┘
```

Linear form: `data -> train -> registry -> evaluate -> drift -> retrain` — every threshold, seed and split fraction lives in [`mlops/config.py`](mlops/config.py), so one file documents the whole contract.

## Components

| Component | File | What it does | Exit codes |
|---|---|---|---|
| Data layer | [`mlops/data.py`](mlops/data.py) | load the sklearn dataset, `validate_frame()` contract check, stratified 70/15/15 split with the seed | — |
| Training | [`mlops/train.py`](mlops/train.py) | `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=seed)`; validation accuracy + ROC-AUC, byte-identical per seed | — |
| Model registry | [`mlops/registry.py`](mlops/registry.py) | `artifacts/model-v{n}.joblib` + `artifacts/registry.json` (version, `trained_at`, metrics, feature names, training-data sha256); `latest()` resolves the newest | — |
| Evaluation gate | [`mlops/evaluate.py`](mlops/evaluate.py) | exit 0 iff test ROC-AUC ≥ 0.95 and accuracy ≥ 0.90 (thresholds in config, never inline) | `0` pass / `1` fail / `2` usage |
| Drift monitor | [`mlops/drift.py`](mlops/drift.py) | PSI per feature (train reference vs incoming batch) → `drift_report.json`; drifted iff mean PSI > 0.15 or max PSI > 0.30 | `0` always / `2` usage |
| Auto-retrain | [`mlops/retrain.py`](mlops/retrain.py) | on drift: retrain on reference+incoming, register the next version | `0` retrained / `3` no drift / `2` usage |
| Unified CLI | [`mlops/cli.py`](mlops/cli.py) + [`mlops/__main__.py`](mlops/__main__.py) | `python3 -m mlops <train\|evaluate\|drift\|retrain> [--seed N] [--batch PATH]` dispatches to the modules above | module codes / `2` usage |
| Tests | [`tests/`](tests/) | **133 deterministic pytest tests**: unit (PSI math, data contract, registry round-trip, gate thresholds, CLI parsing) + integration (full train → evaluate → drift → retrain cycle) | `0` = green |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Python 3.12, `pip install scikit-learn joblib scipy pytest`, runs `./verify.sh` — the same gates as local | must pass |
| Gates | [`verify.sh`](verify.sh) | `compileall` + `pytest` (full suite) + `ruff check` when ruff is installed | `0` required |

---

## Requirements — every one of these must exist, work, and be tested

1. **Data layer** (`mlops/data.py`): load `sklearn.datasets.load_breast_cancer`, stratified split **70/15/15 with seed 42** (train/valid/test), and a `validate_frame()` contract check (expected feature count, no NaNs, target ∈ {0,1}).
2. **Training** (`mlops/train.py`): `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=seed)` — reports accuracy + ROC-AUC on the validation split; **two runs with the same seed produce byte-identical metrics** (enforced by a test).
3. **Model registry** (`mlops/registry.py`): `artifacts/model-v{n}.joblib` + `artifacts/registry.json` recording version, `trained_at`, metrics, feature names, and a sha256 of the training data; `latest()` always resolves the newest version.
4. **Evaluation gate** (`mlops/evaluate.py`): exits **0 iff** test ROC-AUC ≥ **0.95** and accuracy ≥ **0.90** (thresholds live in `mlops/config.py`, not inline); non-zero exit prints which threshold failed.
5. **Drift monitor** (`mlops/drift.py`): Population Stability Index per feature (train reference vs an incoming batch); writes `drift_report.json`; **drifted iff mean PSI > 0.15 OR max PSI > 0.30** (thresholds in config).
6. **Auto-retrain** (`mlops/retrain.py`): when drifted → retrains on reference+incoming, registers the next version; **exit 0 = retrained, exit 3 = no drift (nothing to do)** — distinct exit codes are an API.
7. **CLI** (`mlops/cli.py`): `python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]`.
8. **Tests** (`tests/`, pytest, deterministic): unit (PSI math, data contract, registry round-trip, gate thresholds, CLI parsing) + integration (train → evaluate → drift on a perturbed batch → retrain cycle). **Minimum 30 tests, exit 0** — shipped: **133**.
9. **CI**: `.github/workflows/ci.yml` running the same gates below.
10. **Docs**: this README (quickstart + architecture) plus [`docs/ADR-001-seeded-determinism.md`](docs/ADR-001-seeded-determinism.md) and [`docs/ADR-002-drift-thresholds.md`](docs/ADR-002-drift-thresholds.md) (decision + rejected alternatives, priced honestly).

## Gates — green after *every* milestone

```bash
./verify.sh    # compileall + pytest (exit 0) + ruff when installed
```

CI runs exactly these gates: `.github/workflows/ci.yml` sets up Python 3.12, installs the charter dependencies (`pip install scikit-learn joblib scipy pytest`) and executes `./verify.sh`.

## Milestones — each one relay run, each one commit

| # | Scope |
|---|---|
| **M1** (`40b69fa`) | data layer + training + registry + their tests |
| **M2** (`864f9eb`) | evaluation gate + drift monitor + auto-retrain + their tests |
| **M3** | CLI wiring + CI workflow + README/ADRs + full final gates |

## Rules of the build

- Python 3 only (there is no bare `python` on this machine), dependencies limited to **scikit-learn, joblib, scipy, pytest** — nothing downloaded, ever.
- A milestone is done only when `./verify.sh` exits 0 — no stubs, no skipped-by-default tests, no weakened thresholds.
- Each milestone ends with one descriptive commit; the history is the case-study evidence.

---

[![](https://img.shields.io/badge/powered_by-dsh-4D6BFE?style=flat-square&logo=deepseek&logoColor=white)](https://github.com/deepseek-ai/deepseek-harness)

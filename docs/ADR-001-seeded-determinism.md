# ADR-001: Seeded determinism

- **Status:** Accepted (charter requirement, implemented in M1, wired through the CLI in M3)
- **Scope:** `mlops/config.py`, `mlops/data.py`, `mlops/train.py`, `mlops/registry.py`, `mlops/cli.py`

## Context

The charter demands that "two runs with the same seed produce
byte-identical metrics", enforced by a test. This project is built and
gated by an automated pipeline: every run happens in a fresh process,
often in a fresh directory, with no human to reconcile surprises. A
single non-deterministic ingredient (an unseeded split, an unseeded
solver, a timestamp leaking into a metric file) would make the gates
flaky and the case study worthless as evidence.

Constraints: Python 3 only, dependencies limited to scikit-learn,
joblib, scipy and pytest (+ numpy), no network. The dataset itself is
fixed (sklearn's built-in breast-cancer data: 569 rows × 30 features),
so all run-to-run variance has to come from our own code, not from the
data source.

## Decision

One seed rules everything, declared once in `mlops/config.py` as
`SEED = 42` and used everywhere:

1. **Split** — `train_test_split(..., stratify=y, random_state=seed)`
   twice, producing the documented stratified 70/15/15 split (398
   train / 85 valid / 86 test rows for the default seed).
2. **Model** — `LogisticRegression(max_iter=1000, random_state=seed)`
   inside a `StandardScaler` pipeline; the scaler is fit on the
   training split only, so no test/valid row leaks into preprocessing.
3. **Metrics** — accuracy and ROC-AUC are computed on fixed splits and
   serialized through `metrics_to_bytes()` (sorted-key, compact JSON),
   so equal metrics are byte-equal.
4. **Registry** — the seed is recorded inside the entry's `metrics`
   dict, so every artifact is auditable back to the run that produced
   it (the `trained_at` timestamp is metadata, never part of the
   metrics).
5. **CLI** — `python3 -m mlops <command> [--seed N]` defaults to
   `config.SEED`; the same default is forwarded to every dispatched
   module, so the CLI cannot silently diverge from the modules.

**Priced honestly:**

- Determinism is guaranteed *within one installed environment*. The
  charter forbids anything beyond the four dependencies and we do not
  pin their versions (no lockfile tooling in scope), so a different
  scikit-learn/numpy/BLAS build may produce different floating-point
  last bits. We accept this: the test enforces identity run-to-run on
  the machine that runs the gates, not across all possible machines.
- A single fixed seed means the reported metrics describe **one**
  split, not a distribution. We ship no confidence intervals and make
  no claim that the numbers would survive a different split — that
  claim would need a seed sweep, which is rejected below.
- Determinism here is *enforced by tests*, not by construction: if a
  future change reintroduces randomness, the failure shows up at gate
  time rather than at review time. That is the intended cost.

## Rejected alternatives

1. **Unseeded run (`random_state=None`)** — rejected: directly
   violates the byte-identity requirement; every gate run would be a
   roll of the dice. No upside on a fixed local dataset.
2. **Time- or hash-derived seed** — rejected: a seed that changes with
   the clock or with the file contents makes the split unauditable —
   nobody can predict it from the code alone, and editing an unrelated
   line could silently reshuffle train/test membership. A single
   documented constant is greppable and reviewable.
3. **Seed sweep / k-fold averaging** — rejected: the charter requires
   one deterministic training run, one registered artifact, and
   byte-identical metrics; averaging over folds would multiply runtime
   and produce metrics no single model has. We accept having no
   variance estimate instead.
4. **Non-stratified split** — rejected: with 569 rows, each 15% slice
   is ~85 rows, so an unstratified draw can over/under-represent the
   malignant class depending on the seed, moving metrics between seeds
   more than the model itself warrants. Cost of stratification we do
   accept: split membership depends on the labels (fine in this fully
   offline setting).
5. **Persisting split indices to disk instead of seeding** — rejected:
   a second source of truth (an indices file) that must stay in sync
   with the dataset; seeding keeps the whole contract in one config
   value.

# ADR-002: Drift thresholds

- **Status:** Accepted (charter requirement, implemented in M2)
- **Scope:** `mlops/config.py`, `mlops/drift.py`, `mlops/retrain.py`

## Context

The auto-retrain step needs a binary answer: is the incoming batch
different enough from the training reference to justify retraining?
The monitor computes one Population Stability Index (PSI) value per
feature (30 features) between the seeded train reference (398 rows)
and the incoming `.npz` batch, using 10 reference-quantile bins per
feature with `1e-6` smoothing so empty bins keep PSI finite. The
verdict rule — what counts as "drifted" — is the remaining decision,
and it directly controls how often the pipeline retrains: too eager
and every noisy batch burns a refit plus a registry write; too lazy
and a shifted model keeps serving.

Constraint: we have **no labeled drift history for this dataset** — no
stream of past batches, no backtest set of "this batch was really
shifted", no measured false-alarm or miss rates.

## Decision

**Drifted iff mean PSI > 0.15 OR max PSI > 0.30** (strict `>`), with
both thresholds declared once in `mlops/config.py`
(`PSI_MEAN_THRESHOLD`, `PSI_MAX_THRESHOLD`) and read at call time by
the monitor, the retrain gate, the report and the CLI — never inlined.
The report (`drift_report.json`) records the verdict, both summary
statistics, every per-feature PSI and the thresholds used, so any
verdict can be re-checked against the config that produced it.

The `OR` combines two failure modes on purpose: the **mean** catches a
broad shift across many features, the **max** catches a single feature
spiking while the other 29 stay stable. `retrain.py` maps the verdict
onto its exit-code API (0 = retrained, 3 = no drift).

**Priced honestly:**

- **0.15 / 0.30 are conventional practitioner rules of thumb** for
  PSI (the commonly cited bands treat < ~0.10 as stable, ~0.10–0.25 as
  moderate, > ~0.25 as significant). We adopted them as documented
  heuristics — **we did not calibrate or validate them on this
  dataset**, because there is nothing here to calibrate against: no
  labeled drift events, no backtest, no measured false-alarm rate.
  Any claim that these numbers are "right for breast-cancer data"
  would be invented.
- Consequence of the `OR` rule: **one** noisy feature among 30 crossing
  0.30 triggers a retrain even when the mean is tiny. We accept more
  false-positive retrains in exchange for not missing localized shifts
  (a retrain here is a cheap deterministic refit, so the error is
  low-cost by construction).
- Consequence of the mean rule: subtle drift spread thinly across a
  handful of features may trip neither term. That miss is possible and
  we have not measured how often.
- PSI on a 398-row reference with 10 bins puts roughly 40 reference
  rows per bin, so small-sample noise exists; we report the numbers
  honestly (e.g. the identical batch scores exactly 0.0 by
  construction) rather than claiming a noise floor we never measured.
- The bin count (10) and smoothing (`1e-6`) are **parameters, not
  verdict thresholds**: they shape the PSI values but do not decide
  drift; only the two config thresholds do.

## Rejected alternatives

1. **Mean-only threshold** — rejected: a shift confined to one or two
   features is diluted by 29 stable ones and can stay under the mean
   threshold forever. Cost avoided: silent localized drift. (Keeping
   the max term is exactly why.)
2. **Max-only threshold** — rejected: with 30 per-feature values, the
   noisiest feature alone would decide every time, so the verdict
   degenerates to "did one feature get unlucky", and retrains become
   churn. (Keeping the mean term is exactly why.)
3. **Lower "textbook" cutoffs (e.g. mean > 0.10)** — rejected: we
   cannot honestly justify tighter numbers without a false-alarm
   backtest we do not have; adopting them would be decoration, not
   measurement. Cost accepted: these thresholds may under-detect
   moderate drift.
4. **Per-feature hypothesis tests (KS / chi-square with
   multiple-testing correction)** — rejected: p-values scale with
   sample size, so with a fixed 398-row reference "significance" would
   mean something different from "materially shifted", and Bonferroni
   style corrections add statistical machinery the charter does not
   ask for. PSI keeps one interpretable scale shared by report, gate
   and docs.
5. **Performance-based monitoring (drop in holdout ROC-AUC/accuracy
   after retraining)** — rejected: it answers a different question
   (has the *model* degraded, not has the *input* shifted), needs
   labeled evaluation batches to be timely, and would need its own
   threshold — which, by the same honesty standard, we also could not
   calibrate here.
6. **Per-feature thresholds learned from historical batches** —
   rejected: there is no historical stream to learn from; any such
   numbers would be invented. The single pair in `mlops/config.py` is
   at least auditable and uniformly applied.

# Phase 13 Readiness Audit

**A read-only preparatory artifact for Phase 13 (Full Experiment Program).** It records the
state of the experiment catalog so Phase 13 can be planned. It advances nothing: it does not
run experiments, regenerate results, or create configs, plotters, experiment
implementations, results, figures, or methodology documents.

- **Audit date:** 2026-07-22
- **Audited commit:** the scope-change increment that removed the benchmarking/optimization
  scope and renumbered the phases and experiments (the commit that introduces this file).
- **No experiments were run and no results were regenerated** while producing this audit. It
  was built by inspecting committed files only.

## Scope note

The benchmarking-and-optimization phase was removed from the project. The phases were
renumbered so that **Phase 13 is now the Full Experiment Program** (formerly Phase 14),
followed by Phase 14 Documentation and Technical Paper, Phase 15 Public Results Site, and
Phase 16 Final Acceptance. The experiment catalog was renumbered accordingly: the former
EXP-13 (Parallel Monte Carlo Scaling) was removed, EXP-13 is now **Cross-Method Accuracy and
Agreement** (formerly EXP-14, reframed away from performance toward accuracy and agreement),
EXP-14 is Statistical Confidence Coverage (formerly EXP-15), and EXP-15 is Numerical Edge
Cases (formerly EXP-16). The catalog now mandates **EXP-01 through EXP-15**. There is no
longer any benchmarking dependency, so nothing in the catalog is externally blocked.

## Authoritative sources inspected

- `docs/EXPERIMENT-CATALOG.MD` — the mandatory experiment definitions (EXP-01 through EXP-15)
  and the required-artifacts / final-gate criteria.
- `configs/experiment/` — stored experiment configurations and its `README.md`
  (config → experiment mapping); `configs/calibration/synthetic_surface.json`.
- `data/market/spy_options_2026-07-17.json` — the real-market surface EXP-12 depends on.
- `results/` — committed `EXP-NN.json` / `EXP-NN.csv` records and its `README.md` status table.
- `docs/figures/` — committed charts.
- `src/cli/experiment_command.cpp` — the CLI dispatch that determines which IDs are runnable.
- `src/experiments/`, `include/diffusionworks/experiments/` — experiment implementations.
- `python/` — `plot_convergence.py` (the only chart generator present).
- `docs/*METHODOLOGY*.md`, `FAILURE-MODES.MD` — methodology notes.

## Status labels

| Label | Meaning |
| --- | --- |
| `READY` | Config, committed result, committed chart, and methodology note all present. |
| `PARTIAL` | Implemented and a result is committed, but a chart and/or methodology note is missing. |
| `IMPLEMENTED_NOT_GENERATED` | Runnable (config + CLI dispatch + data present) but no committed result or chart. |
| `NOT_IMPLEMENTED` | No experiment implementation and no stored config; not runnable. |

## EXP-01 – EXP-15 readiness table

| EXP | Readiness | Required config | Implementation (CLI-dispatchable) | Stored result | Chart | Documentation | Known warning/failure | Missing dependency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 MC sampling convergence | `READY` | `convergence.json` | yes | `EXP-01.json` / `.csv` | `exp01_sampling_convergence.png` | CONVERGENCE-METHODOLOGY | pass | — |
| 02 Strong SDE convergence | `READY` | `convergence.json` | yes | `EXP-02.json` / `.csv` | `exp02_strong_convergence.png`, `exp02_local_orders.png` | CONVERGENCE-METHODOLOGY §3 | **warning** — full-range slopes are pre-asymptotic; documented, not a defect (both fits published) | — |
| 03 Weak SDE convergence | `READY` | `convergence.json` | yes | `EXP-03.json` / `.csv` | `exp03_weak_convergence.png` | CONVERGENCE-METHODOLOGY | pass | — |
| 04 Sampling vs discretization | `READY` | `convergence.json` | yes | `EXP-04.json` / `.csv` | `exp04_bias_variance.png` | CONVERGENCE-METHODOLOGY | pass | — |
| 05 Variance-reduction efficiency | `NOT_IMPLEMENTED` | none (named only in code comments) | no | none | none | none | not implemented | config + experiment implementation + plotter |
| 06 PDE stability & grid convergence | `PARTIAL` | `pde.json` | yes | `EXP-06.json` (no `.csv`) | none | PDE-CONVERGENCE-METHODOLOGY | pass | chart tooling + figure; summary `.csv` |
| 07 Barrier monitoring bias | `PARTIAL` | `barrier.json` | yes | `EXP-07.json` / `.csv` | none | BARRIER-MONITORING-METHODOLOGY §5–8 | pass; documents its oracle's breakdown (BGK up-barrier resolved wrong 23/24) and refuses the unresolved B=70 fit | chart tooling + figure |
| 08 Greek estimator comparison | `PARTIAL` | `greeks.json` | yes | `EXP-08.json` / `.csv` | none | none (reasoning only in the record + `results/README.md`) | pass; regime-specific, worst deep-OTM short-maturity | chart tooling + figure; methodology note |
| 09 Heston CF validation | `NOT_IMPLEMENTED` | none | no (not referenced) | none | none | HESTON-PRICING-METHODOLOGY exists (engine-level) | not implemented | config + experiment implementation + reference prices + plotter |
| 10 Heston variance discretization | `IMPLEMENTED_NOT_GENERATED` | `heston_simulation.json` | yes | none | none | HESTON-SIMULATION-METHODOLOGY | not yet generated | run + commit result; chart tooling + figure |
| 11 Heston calibration recovery | `IMPLEMENTED_NOT_GENERATED` | `calibration.json` (+ `configs/calibration/synthetic_surface.json`) | yes | none | none | CALIBRATION-METHODOLOGY | not yet generated | run + commit result; chart tooling + figure |
| 12 Market-surface stability | `IMPLEMENTED_NOT_GENERATED` | `calibration_stability.json` (+ `data/market/spy_options_2026-07-17.json`) | yes | none | none | CALIBRATION-METHODOLOGY | not yet generated | run + commit result; residual-by-strike/maturity charts |
| 13 Cross-method accuracy and agreement | `NOT_IMPLEMENTED` | none | no | none | none | none | not implemented (new experiment; reframed from the former accuracy-per-computation study) | config + cross-method experiment implementation + plotter |
| 14 Statistical confidence coverage | `NOT_IMPLEMENTED` | none | no | none | none | none | not implemented | config + experiment implementation |
| 15 Numerical edge cases | `NOT_IMPLEMENTED` | none | no | none | none | FAILURE-MODES exists (engine-level) | not implemented | config + experiment implementation |

## Gap synthesis

Counts by readiness label (total = 15):

| Readiness | Count | Experiments |
| --- | --- | --- |
| `READY` | 4 | EXP-01, EXP-02, EXP-03, EXP-04 |
| `PARTIAL` | 3 | EXP-06, EXP-07, EXP-08 |
| `IMPLEMENTED_NOT_GENERATED` | 3 | EXP-10, EXP-11, EXP-12 |
| `NOT_IMPLEMENTED` | 5 | EXP-05, EXP-09, EXP-13, EXP-14, EXP-15 |

```
4  ready
3  partially complete
3  implemented but not generated
5  not implemented
= 15 experiments
```

- **`READY` (4):** EXP-01–04 have config, result, chart, and methodology note. EXP-02's
  documented warning is an input to Phase 13's theory/empirics reconciliation, not a blocker.
- **`PARTIAL` (3):** EXP-06, 07, 08 are implemented with committed results but have **no
  committed charts**, and there is no plotter for PDE / barrier / greeks — only
  `plot_convergence.py` (01–04) exists. EXP-06 is also missing its summary `.csv`; EXP-08 has
  no methodology note. The catalog's "all published charts can be regenerated automatically"
  gate cannot pass for these yet.
- **`IMPLEMENTED_NOT_GENERATED` (3):** EXP-10, 11, 12 are CLI-dispatchable with stored configs
  and their data dependencies present, but have no committed results and no charts. Generating
  them is a Phase 13 action and is deliberately not done here.
- **`NOT_IMPLEMENTED` (5):** EXP-05, 09, 13, 14, 15 have no config and no CLI dispatch (05 is
  named only in code comments; 13 is the newly reframed cross-method experiment). Each needs
  new experiment code before Phase 13 can regenerate it.

The single largest cross-cutting dependency is **chart tooling**: six experiments (06, 07,
08, 10, 11, 12) need plotters that do not yet exist beyond `plot_convergence.py`.

## Phase 13 execution

Phase 13 (Full Experiment Program) runs and reconciles this entire catalog: regenerate every
mandatory experiment from stored configuration, collect raw JSON/CSV, generate all summary
tables and charts, assign pass/warning/fail/inconclusive, reconcile theory with the
empirical results, preserve disappointing and failed outcomes, and verify every headline
claim traces to generated evidence. The existing EXP-01–12 work is input to Phase 13; it does
not by itself complete it. Nothing in the catalog is externally blocked.

## Phase 13 outcome

**Re-audited after execution. Generator commit `3846fb3`; every gap above is closed.**

Everything above this heading is the *pre-execution* state, kept as written. It is the record
of what was missing, which is worth more than a document that only ever describes the present.

| Readiness | Then | Now |
| --- | --- | --- |
| `READY` | 4 | **15** |
| `PARTIAL` | 3 | 0 |
| `IMPLEMENTED_NOT_GENERATED` | 3 | 0 |
| `NOT_IMPLEMENTED` | 5 | 0 |

What closed each class of gap:

- **The five unimplemented experiments** (EXP-05, 09, 13, 14, 15) were implemented with stored
  configs, CLI dispatch, permanent tests, and methodology notes. EXP-05 measures *work-normalised*
  statistical efficiency — reciprocal variance times deterministic work units — so its ranking is
  a property of the estimators rather than of the machine that ran them.
- **The chart-tooling dependency**, called out above as the single largest cross-cutting gap, was
  closed by `python/plot_experiments.py`, which owns EXP-05–15 alongside `plot_convergence.py`
  for EXP-01–04. **21 figures** are committed and every one regenerates from a committed record,
  which is the catalog's "all published charts can be regenerated automatically" gate.
- **The `PARTIAL` trio** gained their figures; EXP-06 gained the summary CSV it lacked, and EXP-08
  gained `docs/GREEKS-METHODOLOGY.md`.
- **The ungenerated trio** (EXP-10, 11, 12) were generated and committed.

Two defects surfaced during execution and were fixed rather than documented around:

- The committed `EXP-NN.json` and `EXP-NN.csv` pairs had been produced by *separate runs* of the
  same experiment, because `--format csv` re-runs it; they disagreed on every runtime column.
  CSVs are now derived from the record's own summary table (`python/derive_csv.py`), making each
  pair provably one execution.
- Four experiments reported `pass` for completing rather than for what they measured, and two of
  those treated *finding* their target limitation as the pass condition. Status now describes the
  empirical result, which moved EXP-07, EXP-10, EXP-12, and EXP-14 to `warning`.

Final reconciliation — ten pass, five warning, none fail, none inconclusive:

```
warning: EXP-02, EXP-07, EXP-10, EXP-12, EXP-14
pass:    EXP-01, EXP-03, EXP-04, EXP-05, EXP-06, EXP-08,
         EXP-09, EXP-11, EXP-13, EXP-15
```

Every warning is a disclosed limitation with its evidence preserved, not a malfunction: a
pre-asymptotic fit (02), a refused fit and a reference measurably wrong in one geometry (07), a
decay order resolved in one regime of two (10), a well-fitting surface that does not identify its
parameters (12), and a nominal 95% interval that covers 87.75% in the small-sample skewed corner
(14). `results/README.md` states each in full.

The artifact set is inventoried in `results/MANIFEST.json` with a SHA-256 for every record, CSV,
and figure, and `python/generate_manifest.py --check` re-runs the completeness audit. Every
headline claim in `results/README.md` is mechanically re-derived from the records by
`python/verify_claims.py`.

## Staleness

The pre-execution table reflects the scope-change increment; the outcome section reflects
generator commit `3846fb3`. Any later commit — a regenerated result, a new plotter, a new
experiment — can change a readiness status, and this document is **not** updated automatically.
Re-run `python3 python/generate_manifest.py --check` and `python3 python/verify_claims.py`
against the then current commit before relying on this file.

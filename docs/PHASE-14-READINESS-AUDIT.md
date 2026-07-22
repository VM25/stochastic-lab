# Phase 14 Readiness Audit

**A read-only, Phase 13 preparatory artifact. It advances nothing.** It records the state
of the experiment catalog so Phase 14 (the full experiment program) can be planned, and it
does not run experiments, regenerate results, or create configs, plotters, experiment
implementations, results, figures, or methodology documents. Phase 13 remains open; Phase 14
must not begin until it closes.

- **Audit date:** 2026-07-22
- **Audited commit:** `2f81fd3bea1d951ddacf5cc3fdb4a8c0692095a4`
- **No experiments were run and no results were regenerated** while producing this audit. It
  was built by inspecting committed files only.

## Authoritative sources inspected

- `docs/EXPERIMENT-CATALOG.MD` — the mandatory experiment definitions (EXP-01 through EXP-16)
  and the required-artifacts / final-gate criteria.
- `configs/experiment/` — stored experiment configurations and its `README.md`
  (config → experiment mapping); `configs/calibration/synthetic_surface.json`.
- `data/market/spy_options_2026-07-17.json` — the real-market surface EXP-12 depends on.
- `results/` — committed `EXP-NN.json` / `EXP-NN.csv` records and its `README.md` status table.
- `docs/figures/` — committed charts.
- `src/cli/experiment_command.cpp` — the CLI dispatch that determines which IDs are runnable.
- `src/experiments/`, `include/diffusionworks/experiments/` — experiment implementations.
- `python/` — `plot_convergence.py`, `plot_benchmarks.py` (the only chart generators present).
- `docs/*METHODOLOGY*.md`, `BENCHMARK-PLAN.MD`, `BENCHMARK-CAPTURE-METHODOLOGY.md`,
  `FAILURE-MODES.md` — methodology notes.

## Status labels

| Label | Meaning |
| --- | --- |
| `READY` | Config, committed result, committed chart, and methodology note all present. |
| `PARTIAL` | Implemented and a result is committed, but a chart and/or methodology note is missing. |
| `IMPLEMENTED_NOT_GENERATED` | Runnable (config + CLI dispatch + data present) but no committed result or chart. |
| `NOT_IMPLEMENTED` | No experiment implementation and no stored config; not runnable. |
| `BLOCKED` | Cannot proceed until the open Phase 13 baseline is accepted. |

Note on scope: the catalog mandates **EXP-01 through EXP-16**, not only EXP-01–12. EXP-13
(parallel Monte Carlo scaling) is the Phase 13 baseline itself and lives in the benchmark
harness, not the `experiment` CLI; EXP-14/15/16 are further mandatory experiments.

## EXP-01 – EXP-16 readiness table

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
| 13 Parallel MC scaling | `BLOCKED` | benchmark harness (`scripts/capture_scaling_baseline.py`) | external to experiment CLI | none (baseline pending) | none | BENCHMARK-PLAN / BENCHMARK-CAPTURE-METHODOLOGY | environment unsuitable during the 2026-07-19 window; capture not yet accepted | accepted four-session multithread baseline + independent review (open Phase 13) |
| 14 Accuracy per unit compute | `NOT_IMPLEMENTED` | none | no | none | none | none | not implemented | config + cross-method experiment implementation |
| 15 Statistical confidence coverage | `NOT_IMPLEMENTED` | none (named only in an `online_moments.hpp` comment) | no | none | none | none | not implemented | config + experiment implementation |
| 16 Numerical edge cases | `NOT_IMPLEMENTED` | none | no | none | none | FAILURE-MODES exists (engine-level) | not implemented | config + experiment implementation |

## Corrected gap synthesis

Counts by readiness label (total = 16):

| Readiness | Count | Experiments |
| --- | --- | --- |
| `READY` | 4 | EXP-01, EXP-02, EXP-03, EXP-04 |
| `PARTIAL` | 3 | EXP-06, EXP-07, EXP-08 |
| `IMPLEMENTED_NOT_GENERATED` | 3 | EXP-10, EXP-11, EXP-12 |
| `NOT_IMPLEMENTED` | 5 | EXP-05, EXP-09, EXP-14, EXP-15, EXP-16 |
| `BLOCKED` | 1 | EXP-13 |

```
4  ready
3  partially complete
3  implemented but not generated
5  not implemented
1  blocked by Phase 13
= 16 experiments
```

- **`READY` (4):** EXP-01–04 have config, result, chart, and methodology note. EXP-02's
  documented warning is an input to Phase 14's theory/empirics reconciliation, not a blocker.
- **`PARTIAL` (3):** EXP-06, 07, 08 are implemented with committed results but have **no
  committed charts**, and there is no plotter for PDE / barrier / greeks — only
  `plot_convergence.py` (01–04) and `plot_benchmarks.py` (13) exist. EXP-06 is also missing
  its summary `.csv`; EXP-08 has no methodology note. The catalog's "all published charts can
  be regenerated automatically" gate cannot pass for these yet.
- **`IMPLEMENTED_NOT_GENERATED` (3):** EXP-10, 11, 12 are CLI-dispatchable with stored configs
  and their data dependencies present, but have no committed results and no charts. They do
  not depend on Phase 13; generating them is a Phase 14 action and is deliberately not done
  here.
- **`NOT_IMPLEMENTED` (5):** EXP-05, 09, 14, 15, 16 have no config and no CLI dispatch (05 and
  15 appear only as code comments). Each needs new experiment code before Phase 14 can
  regenerate it.
- **`BLOCKED` (1):** EXP-13 needs the accepted multithread scaling baseline that Phase 13 owes.

The single largest cross-cutting dependency is **chart tooling**: six experiments (06, 07,
08, 10, 11, 12) need plotters that do not yet exist.

## Phase 13 dependency

EXP-13 is `BLOCKED` on the open Phase 13 benchmarking-and-optimization work. The Phase 13
critical path is unchanged: **arm the hardened, single-instance-locked capture waiter during a
genuine unattended window**, obtain an accepted four-session multithread baseline, pass it
through independent review, commit the raw artifacts, then profile and (only where profiling
proves a bottleneck) optimize with before/after evidence and full numerical revalidation.
Phase 14 does not begin until Phase 13 closes.

## Staleness

This audit reflects commit `2f81fd3` only. Any later commit — a generated result, a new
plotter, a new experiment implementation, or a Phase 13 baseline — can change a readiness
status, and this document is **not** updated automatically. Re-run the audit against the then
current commit before relying on it to plan Phase 14.

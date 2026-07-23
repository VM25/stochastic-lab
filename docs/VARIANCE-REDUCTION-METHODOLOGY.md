# Variance-Reduction Efficiency Methodology

How EXP-05 compares Monte Carlo estimators, why its headline metric is efficiency rather
than variance, and what makes the comparison fair. The experiment is defined in
`docs/EXPERIMENT-CATALOG.MD` (EXP-05); this note records the measurement decisions behind
the record in `results/EXP-05.json`.

## 1. The question, and why variance is the wrong headline

The catalog asks which estimator produces the lowest error *per unit of computation*. The
qualifier is the whole point. A technique that halves an estimator's variance while doubling
its work has bought nothing: give the cheaper estimator twice the work and it reaches the
same accuracy. So the experiment reports two columns and keeps them distinct:

- `variance_reduction_ratio` = crude variance / method variance, at an equal observation
  budget. This is the variance side alone.
- `work_normalised_efficiency_gain_over_crude` = crude (variance × work) / method
  (variance × work). This is the quantity that decides which estimator to actually use.

Efficiency is `1 / (variance × work units)` in the standard variance-reduction sense, where
**work units are deterministic** -- counted from the configuration and the estimator, not
measured on a clock. This matters: the project removed all performance-benchmarking scope
(see `docs/PROJECT-IDENTITY.MD`), so ranking estimators by wall-clock time would reintroduce
exactly that. Work-normalisation makes the comparison a property of the estimators, portable
and reproducible, rather than a benchmark of one machine. The wall-clock runtime is still
recorded beside each row as a **diagnostic only**; it never enters efficiency, the ranking,
or the status.

## 1a. The deterministic work model

One work unit is one elementary operation. For an estimator producing `N` observations over
`M` monitoring steps with a control-variate pilot of `P` paths:

| Operation | Count |
| --- | --- |
| Simulated production paths | `N`, or `2N` under antithetic sampling (each observation averages an original and a reflected path) |
| Pilot paths (control variate only) | `P` |
| Path-leg simulations | `(simulated production + pilot) × M` |
| Arithmetic payoff evaluations | one per simulated path, each over `M` points |
| Geometric payoff evaluations (control only) | one per simulated path, each over `M` points |

`work_units` = path-leg simulations + arithmetic averaging + geometric averaging. Every row
reports the components (`simulated_production_paths`, `pilot_paths`, `path_leg_ops`,
`arithmetic_average_ops`, `geometric_average_ops`, `work_units`) so the count is auditable.
The model is an equal-weight elementary-operation count, a deterministic proxy for cost, not
a cycle-accurate one; it deliberately captures the structural differences that matter -- the
antithetic pair cost and the control's pilot and geometric average.

## 2. What is compared, on what

Two instruments, chosen because they exercise different controls:

- a **European call**, crude versus antithetic;
- an **arithmetic Asian call**, crude versus antithetic versus a geometric **control
  variate** versus their **combination**.

The control variate applies only to the arithmetic Asian, whose geometric counterpart has a
closed-form price and tracks the arithmetic average closely (they differ only by arithmetic
versus geometric mean), so it is the natural control. There is no comparably tight
closed-form control for the vanilla call, so the European comparison is crude versus
antithetic only.

## 3. What makes the comparison fair

- **Equal observation budget, and work counted honestly.** Every estimator in a regime runs
  on the same `paths` (observations) and the same seed set. That does *not* make the raw work
  equal -- an antithetic observation simulates two paths, and the control adds a pilot -- which
  is precisely why efficiency divides by the deterministic work units of §1a rather than
  comparing variance at equal observation count.
- **Multiple seeds.** Variance, RMSE, and efficiency are measured across independent seeds
  (`seed_count`). The across-seed standard deviation *is* the dispersion of a single
  `paths`-path estimate -- the estimator's realised error, measured rather than
  self-reported.
- **Multiple regimes.** Spots span out-of-, at-, and in-the-money and volatility is swept,
  because antithetic sampling's benefit is regime-dependent; a single regime would license a
  universal claim the catalog's exit gate forbids.
- **An unbiasedness guard.** Crude, antithetic, and control-variate averages are all exactly
  unbiased. Where an estimator's across-seed mean drifts from a reference it must reproduce
  by more than four standard errors, the record fails rather than reporting the number -- a
  resolved bias in an estimator that is unbiased in theory is a defect, not noise.

## 4. References

- **European call:** the exact Black-Scholes price. The estimators' RMSE is measured against
  it directly.
- **Arithmetic Asian:** no closed form exists, so the reference is a high-path-count
  combined-estimator run, reported per cell so its own (small) sampling error is visible. The
  RMSE at the finest scales carries the reference's uncertainty as well as the estimator's.

## 5. Validating the control's known expectation

A control variate is only unbiased if the control's expectation is genuinely known. The
`control_validation` block checks this rather than assuming it: for every regime it prices
the geometric Asian two ways -- the closed-form `GeometricAsianAnalyticEngine` and an
independent Monte Carlo estimate of the same average -- and confirms they agree within the
resolution threshold. If they did not, the control would be unsound and the record would
fail.

## 6. The finding

The geometric control dominates the arithmetic Asian, removing two to three orders of
magnitude of variance and giving by far the highest **work-normalised** efficiency. The
combined antithetic-plus-control estimator is the instructive case: at the published budget
it has the **lowest variance** of any estimator, in every regime, yet it is **less
work-efficient** than the control variate alone, in every regime, because the antithetic
layer doubles the simulated paths for only a marginal further variance cut once the control
has removed most of it. Ranked by variance the combined estimator wins; ranked by error per
unit of work the control variate alone wins. Reporting only the variance would have named the
wrong estimator best -- which is exactly why work-normalised efficiency is the metric. This
survives the switch from wall-clock to deterministic work: the antithetic layer's doubled
path cost is real either way, and it is what the work model counts.

Antithetic sampling on its own is the mild case: a real but modest gain for the monotone
European payoff that erodes out of the money, reported per regime rather than as a single
headline, and dwarfed by the control on the Asian. Because an antithetic observation costs
two simulated paths, its variance reduction must clear a factor of two before it improves
work-normalised efficiency at all.

## 7. Limitations

- Efficiency is normalised by a deterministic elementary-operation count, not a cycle-accurate
  cost. It is portable and reproducible, but it weights each operation equally, so it is a
  proxy for cost, not a measurement. Wall-clock runtime is recorded only as a diagnostic and
  is not a performance claim.
- The arithmetic Asian reference is a Monte Carlo estimate, not a closed form.
- Antithetic sampling's benefit depends on the payoff being monotone in the shock, which
  holds for these calls but not in general.
- The interpretation prose describes the result at the published configuration; a different
  configuration could shift the regime-dependent variance ordering (though not the efficiency
  ranking, which the antithetic layer's doubled path cost fixes).

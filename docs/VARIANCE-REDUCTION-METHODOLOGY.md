# Variance-Reduction Efficiency Methodology

How EXP-05 compares Monte Carlo estimators, why its headline metric is efficiency rather
than variance, and what makes the comparison fair. The experiment is defined in
`docs/EXPERIMENT-CATALOG.MD` (EXP-05); this note records the measurement decisions behind
the record in `results/EXP-05.json`.

## 1. The question, and why variance is the wrong headline

The catalog asks which estimator produces the lowest error *per unit of computation*. The
qualifier is the whole point. A technique that halves an estimator's variance while doubling
its work has bought nothing: run the cheaper estimator for twice as long and it reaches the
same accuracy. So the experiment reports two columns and keeps them distinct:

- `variance_reduction_ratio` = crude variance / method variance, at an equal path budget.
  This is the variance side alone.
- `efficiency_gain_over_crude` = crude (variance × runtime) / method (variance × runtime).
  This is the quantity that decides which estimator to actually use.

Efficiency is `1 / (variance × cost)` in the standard variance-reduction sense. It is a
*relative* number within one run on one machine, never an absolute rate; the project makes
no performance claims (see `docs/PROJECT-IDENTITY.MD`), and the seconds that enter the
efficiency ratio are a within-run cost comparison under one build, not a benchmark.

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

- **Equal path budget.** Every estimator in a regime runs on the same `paths` and the same
  seed set, so an efficiency difference is the estimator's, not the sample size's.
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
magnitude of variance and giving by far the highest efficiency. The combined
antithetic-plus-control estimator is the instructive case: at the published path budget it
has the **lowest variance** of any estimator, in every regime, yet it is **less efficient**
than the control variate alone, because layering antithetic sampling on top roughly doubles
the payoff work for only a marginal further variance cut once the control has removed most of
it. Ranked by variance the combined estimator wins; ranked by error per unit of computation
the control variate alone wins. Reporting only the variance would have named the wrong
estimator best -- which is exactly why efficiency is the metric.

Antithetic sampling on its own is the mild case: a real but modest gain for the monotone
European payoff that erodes out of the money, reported per regime rather than as a single
headline, and dwarfed by the control on the Asian.

## 7. Limitations

- The efficiency numbers are wall-clock on one machine and relative within a single run; the
  absolute seconds are not portable and not a performance claim.
- The arithmetic Asian reference is a Monte Carlo estimate, not a closed form.
- Antithetic sampling's benefit depends on the payoff being monotone in the shock, which
  holds for these calls but not in general.
- The interpretation prose describes the result at the published configuration; a different
  configuration could shift the regime-dependent variance ordering (though not the efficiency
  ranking, which the antithetic layer's cost fixes).

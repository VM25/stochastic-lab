# Cross-Method Accuracy and Agreement Methodology

How EXP-13 checks that the independent pricing methods agree where they price the same
instrument. The experiment is defined in `docs/EXPERIMENT-CATALOG.MD` (EXP-13). It replaced a
former "accuracy per unit of computation" experiment: this one asks about **agreement and
accuracy**, not cost, and makes no performance claim.

## 1. The question and the discipline

Several methods in this project can price the same contract from entirely different
mathematics: an analytic formula, a Monte Carlo simulation, a finite-difference PDE solver, a
characteristic-function integral. If they are all correct they must agree, up to the sampling
and discretization error each carries. EXP-13 checks that, under three disciplines:

- **Each method is used only where it is defined.** The finite-difference solver prices the
  European option under Black-Scholes; the characteristic-function integral and the Heston
  simulation price the European option under Heston; the control variate applies to the
  arithmetic Asian. No method is presented as a reference for a regime it cannot price.
- **Agreement is measured against the right kind of reference**, not a single universal oracle.
- **Disagreements would be explained, not hidden** — attributed to sampling or discretization
  with the evidence to back it — but here none exceeds tolerance.

## 2. The three families and their references

### European under Black-Scholes — an exact reference

The analytic Black-Scholes price is exact and is the reference. Crude and antithetic Monte
Carlo must agree with it within a standard-error band (their difference is a small fraction of
their own standard error, and the 95% confidence interval covers the analytic price), and the
Crank-Nicolson finite-difference solver must agree within a grid-based relative tolerance. The
tolerances differ by method because the error does: a standard error for the stochastic
methods, a discretization tolerance for the deterministic one.

### European under Heston — an analytic reference from a different route

The characteristic-function integration is the reference; the full-truncation Monte Carlo
simulation must agree with it within sampling uncertainty, in both a Feller-satisfying and a
Feller-violating regime. The simulation's step count is chosen so its discretization bias — the
subject of EXP-10 — is smaller than the sampling error here, so the remaining difference is
sampling, not a hidden bias. This is not a claim that full truncation is unbiased (it is not);
only that its bias is not resolved at this resolution, which is why the agreement holds.

### Arithmetic Asian under Black-Scholes — no closed form

There is no exact price, so the reference is the estimators themselves. Crude, antithetic, and
control-variate Monte Carlo estimate the same price, so they must agree with one another within
their combined standard errors. That variance reduction changes the precision and not the
answer is exactly the agreement being checked. This mutual-agreement case is kept honestly
separate from the cases where a real external reference exists.

## 3. The agreement criterion

A Monte Carlo estimate agrees with a reference when their difference is within a conservative
number of combined standard errors (`agreement_sigma`, default 5). Five sigma is deliberately
loose: an unbiased estimator effectively never fails by chance, while a real bias larger than
the sampling noise is still caught. The finite-difference method is deterministic, so it is held
to a relative tolerance justified by its grid rather than a standard error. The `pass` verdict
requires every method to agree with its reference in every regime.

The 95% Monte Carlo confidence-interval coverage of the reference is reported alongside as
agreement evidence; the rigorous coverage study, over many seeds, is EXP-14.

## 4. Limitations

- Agreement is checked at a single seed per Monte Carlo method, using the run's own standard
  error as its uncertainty. A single seed cannot detect a bias smaller than its standard error;
  the multi-seed coverage study is EXP-14 and the estimator-quality comparison is EXP-05.
- The finite-difference method is compared only where it is implemented (European under
  Black-Scholes); it is not a reference for the Asian or Heston instruments.
- The Heston Monte Carlo step count keeps the full-truncation bias below the sampling error, so
  the agreement is within sampling uncertainty — not evidence that the scheme is unbiased.
- The arithmetic Asian agreement is mutual, not against an external truth. Three estimators
  agreeing is strong evidence of a shared expectation, but a bias common to all three (the same
  discretization on the same monitoring grid) would not be caught by comparing them to each
  other.

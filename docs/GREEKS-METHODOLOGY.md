# Greek estimator methodology

Implementation note, not a specification. `docs/BUILD-PLAN.MD` Phase 8 and
`docs/EXPERIMENT-CATALOG.MD` EXP-08 state what the Greeks framework must establish;
this records how the comparison is measured, and why the textbook picture of the
finite-difference variance is wrong under the estimator this project actually uses.

## 1. The obvious measurement contradicts the textbook, and the textbook is the one at fault

The received account of a central finite-difference Greek is a bias–variance
trade-off forced by the bump `h`:

```
bias    ~ C · h^2          (central difference, smooth payoff)
variance ~ sigma^2 / h^2   (delta),  sigma^2 / h^4  (gamma)
```

so the standard error grows as `1/h` for delta and `1/h^2` for gamma as the bump
shrinks, and one is supposed to pick `h` to balance the two. That formula assumes
the two bumped prices are estimated from **independent** Monte-Carlo draws.

This project does not price the two bumps independently. It uses **common random
numbers**: the same underlying normals drive `V(S+h)` and `V(S-h)`, so the
difference `V(S+h) − V(S−h)` is a difference of two strongly positively correlated
estimators. As `h → 0` the shared-draw difference quotient converges path by path
to the pathwise derivative, whose variance is finite and bump-independent. The
`1/h` and `1/h^2` blow-ups are a property of the independent-draw estimator, not of
finite differencing.

So the experiment does not assume this — it **fits the exponent** of the across-seed
dispersion against the bump, per cell, exactly as the Phase 5 convergence studies fit
an order, and reports the fitted exponent with its interval:

| Greek | textbook `1/h^k` exponent | measured median exponent | measured range |
| --- | --- | --- | --- |
| delta | 1 | **0.006** | −0.04 → 0.24 |
| gamma | 2 | **0.527** | 0.40 → 1.96 |
| vega | 1 | **−0.001** | −0.01 → 0.02 |

The delta and vega dispersions are essentially bump-independent (exponent ~0, not
1), because under common random numbers the shared-draw estimator has already
converged to the pathwise one. Gamma sits near **0.5**, not 2: the second difference
`V(S+h) − 2V(S) + V(S−h)` retains a residual `O(h)`-scale cancellation error that
the first difference does not, but it is nowhere near the independent-draw `1/h^2`.
The single cell where gamma reaches 1.96 is deep in-the-money at short maturity,
where the smooth-payoff assumption behind the second difference is weakest — the
range is reported rather than smoothed into the median.

**This is the headline, and it is a correction to a heuristic, not a new law.** The
exponent is not universal: it ranges with moneyness and maturity, and the record
carries every per-cell value so a reader can see the spread rather than a single
number standing in for it.

## 2. What is compared, and against what reference

Four estimators of the same Greek:

- **analytic** — the closed-form Black–Scholes sensitivity, used as the reference the
  others are scored against. It is computed, not sampled, so it carries no standard
  error, and a `0.0` is never written where a measured uncertainty is absent.
- **central finite difference** — bumped revaluation under common random numbers,
  swept over bump size.
- **pathwise** — differentiating the discounted payoff along each path. Available for
  delta and vega; **not for gamma**, because the call payoff's first derivative is a
  step and its second derivative is a delta function, so the pathwise second
  derivative does not exist as a path average.
- **likelihood ratio** — differentiating the density rather than the payoff, so the
  payoff is never differentiated and the estimator survives a discontinuous payoff
  where pathwise cannot.

Bias is measured against the analytic value across independent seeds; a bias counts
as **resolved** only when it clears `bias_resolution_threshold = 5` across-seed
standard errors. Below that, the record says the bias was not resolved rather than
fitting a number to sampling noise (`FAILURE-MODES` section 6; the same discipline
as the barrier and convergence studies).

## 3. No estimator wins everywhere, and the record refuses to name one

The comparison is reported per cell, not collapsed into a ranking, because the
ordering changes with the regime:

- **Variance, delta.** Pathwise and likelihood-ratio are both unbiased for delta, and
  pathwise has the smaller standard error in **every** cell — typically about
  **2.4× smaller** (median across cells), widening to roughly **250×** in the
  deep-out-of-the-money, short-maturity corner, where the likelihood-ratio score
  weight has heavy variance and the option pays on almost no paths.
- **Coverage.** Pathwise has no gamma at all. Likelihood-ratio does, and is the only
  estimator here that keeps working on a discontinuous (digital-type) payoff, which is
  exactly where pathwise breaks.
- **Finite difference.** Cheap and general, and under common random numbers its delta
  variance does not blow up as the bump shrinks (section 1) — but it is the one
  estimator with a genuine, bump-dependent bias, and it is never the variance winner
  where an unbiased estimator exists.

The worst region for the whole comparison is **deep out-of-the-money at short
maturity**: few paths pay, so every sampled estimator is noisy and the
likelihood-ratio variance is at its worst. The record marks this region explicitly
rather than letting a favourable median hide it.

## 4. The pass condition, and what it deliberately does not assert

EXP-08 **passes** only if no estimator that is *theoretically unbiased* — pathwise
or likelihood-ratio — shows a **resolved** bias against the analytic reference. In
this run no such cell exists: every resolved bias belongs to the finite-difference
estimator, which is expected to have one. A resolved bias in a pathwise or
likelihood-ratio Greek would indict the estimator's implementation and fails the
record.

What the experiment does **not** assert:

- **No overall winner.** "Best estimator" is not a well-posed question across regimes;
  the record reports the trade-off and refuses to crown one.
- **The exponents are not a theorem.** The measured `~0` (delta) and `~0.5` (gamma)
  dispersion exponents are the behaviour of *this* common-random-number estimator at
  *these* moneyness/maturity cells and this seed count. They demonstrate that the
  textbook `1/h`, `1/h^2` picture does not apply here; they are not a claim that the
  exponent is `0` and `0.5` everywhere.
- **One model, one instrument.** Every number is Black–Scholes on a European call
  (with a digital-type payoff used only to expose the pathwise breakdown). The
  ranking is a property of these payoffs, not a universal estimator ranking.

# Validation Plan

## Purpose

Prove that DiffusionWorks produces mathematically correct, numerically defensible, statistically reliable, and reproducible results.

Validation is complete only when every required method passes justified tests or its limitations are explicitly documented.

---

## 1. Validation Principles

1. Validate before optimizing.
2. Prefer independent references.
3. Use method-specific tolerances.
4. Separate bias, variance, discretization error, and floating-point error.
5. Preserve failed and inconclusive cases.
6. Never widen tolerances merely to obtain a pass.
7. Every published claim must trace to stored validation evidence.

---

## 2. Validation Status

Every validation must return:

* `PASS`;
* `FAIL`;
* `WARNING`;
* `INCONCLUSIVE`.

A `WARNING` or `INCONCLUSIVE` result must include the cause and required follow-up.

Required `FAIL` results block project completion.

---

## 3. Black–Scholes Validation

Validate native European call and put prices against trusted closed-form reference values.

Required coverage:

* in-, at-, and out-of-the-money;
* short and long maturities;
* low and high volatility;
* zero and nonzero dividend yield;
* limiting cases.

Validate analytic:

* delta;
* gamma;
* vega;
* theta;
* rho.

Use tight, scale-aware tolerances.

---

## 4. Financial Invariants

Required checks:

### Put–Call Parity

[
C-P=S_0e^{-qT}-Ke^{-rT}
]

### Bounds

Verify call and put lower and upper bounds.

### Monotonicity

Check expected relationships with:

* spot;
* strike;
* volatility;
* barrier position.

### Barrier Relationships

Where definitions match:

[
\text{Knock-In}+\text{Knock-Out}=\text{Vanilla}
]

Any violation beyond justified tolerance is a failure.

---

## 5. GBM Simulation Validation

Validate exact GBM simulation against theoretical moments:

[
E[S_T]=S_0e^{(r-q)T}
]

and the corresponding theoretical variance.

Required checks:

* sample mean;
* sample variance;
* terminal distribution;
* reproducibility;
* independent stream behavior;
* common-random-number consistency.

Use multiple seeds and statistically justified acceptance intervals.

---

## 6. Monte Carlo Pricing Validation

Compare Monte Carlo European prices with analytic Black–Scholes values.

Validate:

* estimator mean;
* standard error;
* confidence interval;
* RMSE across seeds;
* (N^{-1/2}) sampling convergence;
* deterministic reproduction from fixed seeds.

A plausible point estimate without valid uncertainty is not a pass.

---

## 7. Confidence-Interval Validation

Repeat independent simulations and measure empirical coverage.

For nominal confidence level (1-\alpha), verify that observed coverage is statistically compatible with the target.

Investigate:

* small samples;
* skewed payoffs;
* barrier options;
* variance-reduced estimators.

Systematic undercoverage blocks completion.

---

## 8. SDE Convergence Validation

Under GBM, compare Euler–Maruyama and Milstein with exact paths using identical Brownian increments.

Validate:

* strong error;
* weak error;
* empirical convergence slope;
* slope uncertainty;
* behavior across multiple grid sizes.

Expected strong orders:

* Euler–Maruyama: approximately (0.5);
* Milstein: approximately (1.0).

Material deviations require explanation or correction.

---

## 9. Variance-Reduction Validation

For antithetic and control variates, verify:

* unbiasedness within statistical uncertainty;
* correct control expectation;
* valid covariance estimation;
* variance reduction across multiple seeds;
* net efficiency after runtime cost.

A method does not pass merely because one run has lower error.

---

## 10. PDE Validation

Compare explicit, implicit, and Crank–Nicolson prices against analytic Black–Scholes values.

Validate:

* terminal conditions;
* boundary conditions;
* tridiagonal solver;
* interpolation;
* grid convergence;
* (S_{\max}) sensitivity;
* time-step sensitivity;
* explicit-scheme instability detection.

Implicit stability must not be treated as proof of accuracy.

---

## 11. Barrier Validation

Validate barrier pricing against:

* trusted analytic values where available;
* high-resolution numerical references;
* barrier parity where applicable.

Required checks:

* monitoring convention;
* already-breached barrier;
* near-barrier behavior;
* discrete-monitoring bias;
* Brownian-bridge correction.

Unstated monitoring assumptions are a failure.

---

## 12. Greek Validation

Compare:

* analytic;
* finite difference;
* pathwise;
* likelihood ratio.

Validate:

* bias;
* variance;
* RMSE;
* bump-size sensitivity;
* common-random-number use;
* stability across moneyness and maturity.

Do not apply pathwise estimators to unsupported discontinuous payoffs without justification.

---

## 13. Heston Pricing Validation

Validate characteristic-function prices against:

* published reference values;
* optional QuantLib outputs;
* internal integration-convergence checks.

Required coverage:

* varying strikes and maturities;
* positive and negative correlation;
* high volatility of variance;
* Feller-condition pass and failure;
* difficult integration regimes.

Document:

* characteristic-function convention;
* quadrature method;
* bounds;
* tolerances;
* branch handling.

---

## 14. Heston Simulation Validation

Compare full-truncation Monte Carlo European prices with characteristic-function references.

Validate:

* correlated shocks;
* variance truncation;
* negative pre-truncation frequency;
* discretization bias;
* time-step convergence;
* finite path states;
* reproducibility.

Silent correction of invalid variance paths is prohibited.

---

## 15. Implied-Volatility Validation

Validate the solver using prices generated from known Black–Scholes volatilities.

Required checks:

* volatility recovery;
* arbitrage-bound rejection;
* deep in- and out-of-the-money cases;
* short maturities;
* non-convergence handling;
* root-bracketing behavior.

The solver must never return a plausible value when no valid root exists.

---

## 16. Calibration Validation

### Synthetic Recovery

1. Generate a Heston surface from known parameters.
2. Calibrate from multiple initial guesses.
3. Compare recovered prices, implied volatilities, and parameters.

Validate:

* volatility RMSE;
* price RMSE;
* convergence status;
* parameter error;
* dispersion across starts;
* objective reproducibility.

Accurate surface recovery may pass even when parameters are non-unique, but that limitation must be explicit.

### Market Surface

Validate:

* source and timestamp;
* input cleaning;
* arbitrage checks where feasible;
* residuals by strike and maturity;
* sensitivity to weights, bounds, and initialization.

---

## 17. External Cross-Validation

QuantLib may provide optional independent references.

Requirements:

* native and external results must be clearly separated;
* conventions must match;
* comparison configurations must be stored;
* disagreements must be investigated;
* QuantLib output cannot override mathematical reasoning automatically.

---

## 18. Edge-Case Validation

Required cases:

* (T=0) or near zero;
* (\sigma=0) or near zero;
* extreme strikes;
* barrier already breached;
* (\rho) near (\pm1);
* near-zero Heston variance;
* invalid parameter values;
* NaN and infinity inputs.

Expected behavior must be either:

* theoretically correct output;
* explicit rejection;
* explicit numerical failure.

Silent fallback is prohibited.

---

## 19. Reproducibility Validation

From a clean checkout, verify that stored commands regenerate:

* reference prices;
* validation tables;
* experiment results;
* benchmark inputs;
* published charts.

Record:

* seed;
* configuration version;
* Git commit;
* compiler;
* build flags;
* hardware;
* thread count.

Missing reproduction metadata is a failure.

---

## 20. Tolerance Policy

Tolerances must depend on:

* method;
* reference scale;
* grid resolution;
* path count;
* standard error;
* integration tolerance;
* floating-point sensitivity.

Examples:

* analytic tests: tight absolute and relative tolerances;
* Monte Carlo tests: confidence-based tolerances;
* PDE tests: convergence-based tolerances;
* calibration tests: price, volatility, and parameter criteria.

Every tolerance must be documented.

---

## 21. Required Validation Artifacts

Produce:

* machine-readable validation results;
* summary matrix;
* failed-case log;
* reference-value source map;
* tolerance registry;
* reproduction commands;
* generated tables and plots.

No manual-only validation may support a headline claim.

---

## Definition of Done

Validation is complete only when:

* all required methods have independent references;
* invariants pass;
* stochastic uncertainty is validated;
* convergence behavior is measured;
* PDE and integration errors are controlled;
* Heston simulation matches semi-analytic references;
* calibration passes synthetic recovery;
* edge cases fail safely;
* all results are reproducible;
* unresolved failures are absent.

## Completion Blockers

The project is not done if:

* outputs are accepted because they look plausible;
* methods validate against themselves;
* tolerances are arbitrary;
* stochastic results use one seed;
* confidence intervals under-cover systematically;
* convergence is inferred from insufficient data;
* Heston integration lacks independent reference checks;
* calibration lacks synthetic recovery;
* QuantLib performs core calculations;
* failed validations are hidden;
* published results cannot be regenerated.

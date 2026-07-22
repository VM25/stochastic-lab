# Experiment Catalog

## Purpose

Define the mandatory experiments that prove DiffusionWorks is mathematically correct, numerically credible, reproducible, and computationally efficient.

Every experiment must:

* run from a stored configuration;
* record seeds, parameters, compiler, hardware, threads, and Git commit;
* produce machine-readable results;
* use multiple independent seeds when stochastic;
* preserve failed and contradictory results;
* meet its exit criteria before completion.

---

## EXP-01: Monte Carlo Sampling Convergence

**Question:** Does Monte Carlo pricing error decay at the expected rate?

**Model:** Black–Scholes
**Instrument:** European call and put
**Reference:** Analytic Black–Scholes value

Vary:

* path count (N);
* moneyness;
* maturity;
* volatility.

Measure:

* absolute error;
* RMSE across seeds;
* standard error;
* confidence-interval width;
* runtime;
* empirical log-log convergence slope.

**Expected behavior:**

[
RMSE=O(N^{-1/2})
]

**Exit criteria:**

* error decreases consistently with (N);
* observed slope is defensibly close to (-0.5);
* analytic values receive statistically reasonable confidence coverage;
* deviations are explained.

**Failure conditions:**

* one seed is used;
* confidence intervals are omitted;
* increasing (N) does not improve error without explanation;
* slope is asserted visually rather than estimated.

---

## EXP-02: Strong SDE Convergence

**Question:** Do Euler–Maruyama and Milstein achieve their expected strong convergence orders?

**Model:** GBM
**Reference:** Exact GBM path using identical Brownian increments

Vary:

* time-step size (\Delta t);
* volatility;
* maturity.

Measure:

[
E\left[|S_T^{\Delta t}-S_T^{exact}|\right]
]

Expected orders:

* Euler–Maruyama: approximately (0.5);
* Milstein: approximately (1.0).

**Exit criteria:**

* empirical slopes are estimated with uncertainty;
* Milstein materially outperforms Euler at sufficiently fine grids;
* path coupling is correct and reproducible.

**Failure conditions:**

* schemes use different Brownian paths;
* only terminal means are compared;
* convergence order is inferred from too few grid levels.

---

## EXP-03: Weak SDE Convergence

**Question:** Do the schemes reproduce expected values at their theoretical weak rate?

**Model:** GBM
**Reference:** Exact moments and analytic European option prices

Test functions:

* (f(S_T)=S_T);
* (f(S_T)=S_T^2);
* European payoff.

Measure:

[
\left|
E[f(S_T^{\Delta t})]-E[f(S_T)]
\right|
]

**Exit criteria:**

* expected weak convergence is visible across multiple test functions;
* sampling uncertainty is separated from discretization bias;
* unexplained inconsistencies are investigated.

---

## EXP-04: Sampling Error Versus Discretization Bias

**Question:** When does increasing path count stop improving accuracy because time discretization dominates?

**Models:** GBM and Heston
**Methods:** Euler–Maruyama, Milstein, full-truncation Euler

Vary jointly:

* paths (N);
* time steps (M).

Measure:

* bias;
* standard error;
* RMSE;
* runtime;
* error decomposition where possible.

**Exit criteria:**

* a clear bias-dominated region is identified;
* increasing (N) at fixed coarse (M) is shown not to remove discretization error;
* efficient path-step combinations are reported.

---

## EXP-05: Variance-Reduction Efficiency

**Question:** Which estimator produces the lowest error per unit of computation?

**Instruments:**

* European option;
* arithmetic Asian option.

Compare:

* crude Monte Carlo;
* antithetic variates;
* control variates;
* combined methods.

Measure:

* estimator variance;
* RMSE;
* runtime;
* variance-reduction ratio;
* efficiency:

[
\frac{1}{Variance\times Runtime}
]

**Exit criteria:**

* comparisons use equivalent path budgets and seeds;
* control expectations are independently validated;
* gains are reported across multiple parameter regimes;
* methods that fail to improve efficiency remain documented.

---

## EXP-06: PDE Stability and Grid Convergence

**Question:** How do explicit, implicit, and Crank–Nicolson methods differ in stability, accuracy, and cost?

**Model:** Black–Scholes
**Instrument:** European call and put
**Reference:** Analytic Black–Scholes value

Vary:

* spatial grid size;
* time-step size;
* (S_{\max});
* maturity;
* volatility;
* moneyness.

Measure:

* pricing error;
* runtime;
* observed convergence;
* oscillation or instability indicators.

**Exit criteria:**

* all stable methods converge toward analytic values;
* explicit instability is detected or demonstrated;
* implicit stability is not misrepresented as accuracy;
* boundary truncation effects are measured.

---

## EXP-07: Barrier Monitoring Bias

**Question:** How much error results from discrete barrier monitoring?

**Model:** Black–Scholes
**Instrument:** Down-and-out and up-and-out calls

Compare:

* coarse discrete monitoring;
* fine monitoring;
* Brownian-bridge correction;
* trusted analytic or high-resolution reference.

Measure:

* price bias;
* barrier-hit frequency;
* runtime;
* sensitivity to barrier distance and volatility.

**Exit criteria:**

* discrete-monitoring bias is quantified;
* Brownian-bridge correction is validated;
* monitoring conventions are explicit;
* near-barrier instability is documented.

---

## EXP-08: Greek Estimator Comparison

**Question:** Which Greek estimator provides the best accuracy and stability?

**Model:** Black–Scholes
**Instrument:** European call

Compare:

* analytic;
* central finite difference;
* pathwise;
* likelihood ratio.

Measure:

* bias;
* variance;
* RMSE;
* runtime;
* sensitivity to bump size;
* stability across moneyness and maturity.

**Exit criteria:**

* finite-difference bump-size trade-offs are demonstrated;
* common random numbers are used where applicable;
* estimator limitations for discontinuous payoffs are documented;
* no method is declared superior without regime-specific evidence.

---

## EXP-09: Heston Characteristic-Function Validation

**Question:** Does native Heston integration reproduce trusted reference prices?

Vary:

* strike;
* maturity;
* correlation;
* volatility of variance;
* mean-reversion parameters;
* Feller-condition status.

Measure:

* absolute and relative pricing error;
* integration convergence;
* runtime;
* sensitivity to bounds and tolerances.

**Exit criteria:**

* prices match independent references within justified tolerances;
* branch and integration failures are explicit;
* difficult parameter regions are retained and explained.

---

## EXP-10: Heston Variance Discretization

**Question:** How does full-truncation Euler behave across stable and difficult parameter regimes?

Compare:

* naive Euler diagnostic implementation;
* full-truncation Euler;
* characteristic-function reference for European prices.

Measure:

* negative pre-truncation variance frequency;
* minimum variance;
* pricing bias;
* path failures;
* runtime;
* sensitivity to (\Delta t).

**Exit criteria:**

* invalid variance behavior is quantified;
* full truncation avoids silent invalid states;
* remaining discretization bias is measured;
* Feller-condition failure is reported but not treated as automatic invalidity.

---

## EXP-11: Heston Calibration Recovery

**Question:** Can calibration recover parameters from a synthetic surface generated by known parameters?

Procedure:

1. generate a synthetic surface;
2. calibrate from multiple initial guesses;
3. compare recovered prices, implied volatilities, and parameters.

Measure:

* objective value;
* volatility RMSE;
* parameter error;
* convergence rate;
* evaluation count;
* dispersion across initial guesses.

**Exit criteria:**

* surface fit is accurate;
* parameter recovery limitations are stated;
* multiple parameter sets with similar fits are preserved;
* optimizer non-convergence is explicit.

**Failure conditions:**

* success is judged only by objective reduction;
* calibration uses the true parameters as its only initial guess;
* synthetic recovery is omitted.

---

## EXP-12: Market-Surface Calibration Stability

**Question:** How stable are calibrated Heston parameters on a fixed documented market surface?

Vary:

* initial guesses;
* weighting schemes;
* parameter bounds;
* selected strikes and maturities.

Measure:

* residual surface;
* parameter dispersion;
* objective value;
* implied-volatility RMSE;
* runtime.

**Exit criteria:**

* data source and timestamp are documented;
* residuals are shown by strike and maturity;
* parameter instability is reported;
* no economic interpretation exceeds the evidence.

---

## EXP-13: Cross-Method Accuracy and Agreement

**Question:** Where independent methods price the same instrument, do they agree within their justified tolerances?

Compare where applicable:

* analytic;
* Monte Carlo;
* variance-reduced Monte Carlo;
* finite difference;
* Heston characteristic-function integration.

Measure:

* pairwise price differences between methods;
* agreement within each method's stated uncertainty or tolerance;
* Monte Carlo confidence-interval coverage of the analytic or reference value;
* convergence of each method toward the shared reference;
* discrepancies attributable to discretization or sampling.

**Exit criteria:**

* methods that price the same instrument agree within justified tolerances;
* each method is tested only on supported instruments;
* disagreements are explained by known bias or uncertainty, not hidden;
* no method is presented as a reference for a regime it cannot price;
* agreement is never claimed without correctness equivalence.

---

## EXP-14: Statistical Confidence Coverage

**Question:** Do reported Monte Carlo confidence intervals achieve their intended coverage?

Procedure:

* repeat independent simulations;
* test whether the analytic value lies inside each interval;
* compare observed and nominal coverage.

Measure:

* coverage rate;
* interval width;
* sample size;
* payoff skewness;
* seed count.

**Exit criteria:**

* coverage is statistically defensible;
* poor coverage at small samples or difficult payoffs is explained;
* interval methodology is revised if systematically invalid.

---

## EXP-15: Numerical Edge Cases

**Question:** Does the system fail safely or remain correct under limiting and difficult inputs?

Test:

* (T\to0);
* (\sigma\to0);
* deep in- and out-of-the-money options;
* barrier already breached;
* (\rho\to\pm1);
* near-zero Heston variance;
* extreme but valid strikes and maturities.

**Exit criteria:**

* valid limiting behavior matches theory;
* invalid configurations are rejected;
* NaN, infinity, and silent fallback values do not escape;
* every unresolved edge case blocks completion.

---

## Required Artifacts

Each experiment must generate:

* versioned configuration;
* raw JSON or CSV results;
* summary table;
* required plots;
* methodology note;
* pass, fail, warning, or inconclusive status;
* explanation of deviations;
* exact reproduction command.

## Final Experiment Gate

The experiment program is complete only when:

* all required experiments run from clean configurations;
* headline claims are generated from stored evidence;
* stochastic conclusions use sufficient independent seeds;
* failed cases remain visible;
* theoretical and empirical results are reconciled;
* no unresolved contradiction is hidden by aggregation;
* all published charts can be regenerated automatically.

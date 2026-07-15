# Mathematical Specification

## Purpose

Define the mathematical models, numerical methods, assumptions, estimators, convergence targets, and validation conditions required for DiffusionWorks.

The implementation must match this specification. If code, experiments, and equations disagree, the project is not complete.

---

## 1. Market Assumptions

Unless an experiment states otherwise:

* frictionless market;
* continuous trading;
* no arbitrage;
* constant risk-free rate (r);
* constant dividend or foreign rate (q);
* European exercise;
* risk-neutral valuation;
* deterministic maturity (T);
* sufficiently regular model parameters.

Discount factor:

[
D(0,T)=e^{-rT}
]

Risk-neutral value:

[
V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\Phi]
]

where (\Phi) is the terminal or path-dependent payoff.

---

## 2. Black–Scholes–Merton Model

Under the risk-neutral measure:

[
dS_t=(r-q)S_t,dt+\sigma S_t,dW_t
]

Exact solution:

[
S_T=S_0\exp\left[\left(r-q-\frac{1}{2}\sigma^2\right)T+\sigma\sqrt{T}Z\right]
]

where:

[
Z\sim N(0,1)
]

### European Call

[
C=S_0e^{-qT}N(d_1)-Ke^{-rT}N(d_2)
]

### European Put

[
P=Ke^{-rT}N(-d_2)-S_0e^{-qT}N(-d_1)
]

with:

[
d_1=
\frac{\ln(S_0/K)+(r-q+\frac{1}{2}\sigma^2)T}
{\sigma\sqrt{T}}
]

[
d_2=d_1-\sigma\sqrt{T}
]

---

## 3. Black–Scholes Greeks

### Call Delta

[
\Delta_C=e^{-qT}N(d_1)
]

### Put Delta

[
\Delta_P=e^{-qT}[N(d_1)-1]
]

### Gamma

[
\Gamma=
\frac{e^{-qT}\phi(d_1)}
{S_0\sigma\sqrt{T}}
]

### Vega

[
\nu=S_0e^{-qT}\phi(d_1)\sqrt{T}
]

### Call Theta

[
\Theta_C=
-\frac{S_0e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}
-rKe^{-rT}N(d_2)
+qS_0e^{-qT}N(d_1)
]

### Put Theta

[
\Theta_P=
-\frac{S_0e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}}
+rKe^{-rT}N(-d_2)
-qS_0e^{-qT}N(-d_1)
]

### Call Rho

[
\rho_C=KTe^{-rT}N(d_2)
]

### Put Rho

[
\rho_P=-KTe^{-rT}N(-d_2)
]

State whether vega and rho are reported per unit change or per one-percentage-point change.

---

## 4. Heston Model

Under the risk-neutral measure:

[
dS_t=(r-q)S_t,dt+\sqrt{v_t}S_t,dW_t^S
]

[
dv_t=\kappa(\theta-v_t),dt+\xi\sqrt{v_t},dW_t^v
]

[
dW_t^S,dW_t^v=\rho,dt
]

Parameters:

* (v_0>0): initial variance;
* (\kappa>0): mean-reversion speed;
* (\theta>0): long-run variance;
* (\xi>0): volatility of variance;
* (\rho\in[-1,1]): correlation.

Feller condition:

[
2\kappa\theta\geq\xi^2
]

The engine must report whether the Feller condition holds. It must not reject valid parameter sets solely because it fails.

### Heston Pricing

European options must be priced using the Heston characteristic function and numerical integration.

The implementation must document:

* characteristic-function convention;
* branch handling;
* integration bounds;
* quadrature method;
* numerical tolerances;
* reference values.

Equivalent valid formulations are permitted only if validated independently.

---

## 5. Payoffs

### European Call

[
\Phi_C=(S_T-K)^+
]

### European Put

[
\Phi_P=(K-S_T)^+
]

### Arithmetic Asian Call

For monitoring times (t_1,\ldots,t_m):

[
A_{\text{arith}}=\frac{1}{m}\sum_{i=1}^{m}S_{t_i}
]

[
\Phi_{\text{Asian Call}}=(A_{\text{arith}}-K)^+
]

### Arithmetic Asian Put

[
\Phi_{\text{Asian Put}}=(K-A_{\text{arith}})^+
]

### Geometric Asian Control

[
A_{\text{geo}}=
\left(\prod_{i=1}^{m}S_{t_i}\right)^{1/m}
]

Use its analytic value as a control variate where assumptions permit.

### Down-and-Out Call

[
\Phi=
(S_T-K)^+
\mathbf{1}
\left{
\min_{0\leq t\leq T}S_t>B
\right}
]

### Up-and-Out Call

[
\Phi=
(S_T-K)^+
\mathbf{1}
\left{
\max_{0\leq t\leq T}S_t<B
\right}
]

Monitoring convention must be explicit:

* continuous;
* discrete;
* Brownian-bridge corrected.

---

## 6. Monte Carlo Estimation

For discounted payoffs (X_i=e^{-rT}\Phi_i):

[
\hat{V}*N=\frac{1}{N}\sum*{i=1}^{N}X_i
]

Sample variance:

[
s^2=
\frac{1}{N-1}
\sum_{i=1}^{N}(X_i-\hat{V}_N)^2
]

Standard error:

[
SE(\hat{V}_N)=\frac{s}{\sqrt{N}}
]

Approximate confidence interval:

[
\hat{V}*N
\pm
z*{1-\alpha/2}SE(\hat{V}_N)
]

Use Student-(t) critical values when sample size is too small for the normal approximation.

Published results must aggregate multiple independent seeds unless the experiment specifically studies deterministic reproducibility.

---

## 7. GBM Discretization

For time step (\Delta t) and (Z_n\sim N(0,1)):

### Exact Transition

[
S_{n+1}
=======

S_n
\exp
\left[
\left(r-q-\frac{1}{2}\sigma^2\right)\Delta t
+
\sigma\sqrt{\Delta t}Z_n
\right]
]

### Euler–Maruyama

[
S_{n+1}
=======

S_n
+
(r-q)S_n\Delta t
+
\sigma S_n\sqrt{\Delta t}Z_n
]

### Milstein

[
S_{n+1}
=======

S_n
+
(r-q)S_n\Delta t
+
\sigma S_n\sqrt{\Delta t}Z_n
+
\frac{1}{2}\sigma^2S_n\Delta t(Z_n^2-1)
]

Use identical Brownian increments when comparing schemes.

---

## 8. Heston Discretization

Use full-truncation Euler:

[
v_n^+=\max(v_n,0)
]

[
v_{n+1}
=======

v_n+\kappa(\theta-v_n^+)\Delta t
+\xi\sqrt{v_n^+}\sqrt{\Delta t}Z_n^v
]

[
S_{n+1}
=======

S_n
\exp
\left[
\left(r-q-\frac{1}{2}v_n^+\right)\Delta t
+
\sqrt{v_n^+\Delta t}Z_n^S
\right]
]

Generate correlated shocks using:

[
Z_n^S=Z_{1,n}
]

[
Z_n^v=\rho Z_{1,n}+\sqrt{1-\rho^2}Z_{2,n}
]

where (Z_{1,n}) and (Z_{2,n}) are independent standard normals.

Report:

* negative pre-truncation variance frequency;
* minimum variance;
* discretization bias;
* parameter regimes with instability.

---

## 9. Strong and Weak Convergence

### Strong Error

For approximation (S_T^{\Delta t}) and exact solution (S_T):

[
\epsilon_{\text{strong}}(\Delta t)
==================================

\mathbb{E}
\left[
|S_T^{\Delta t}-S_T|
\right]
]

Expected GBM orders:

* Euler–Maruyama: (O(\Delta t^{1/2}));
* Milstein: (O(\Delta t)).

### Weak Error

For test function (f):

[
\epsilon_{\text{weak}}(\Delta t)
================================

\left|
\mathbb{E}[f(S_T^{\Delta t})]
-----------------------------

\mathbb{E}[f(S_T)]
\right|
]

Expected weak order:

* Euler–Maruyama: approximately (O(\Delta t));
* Milstein: approximately (O(\Delta t)).

Estimate empirical order using:

[
\log \epsilon = a+p\log \Delta t
]

where fitted slope (p) is the observed convergence order.

Results must include uncertainty around the fitted slope.

---

## 10. Sampling and Discretization Error

Total Monte Carlo pricing error contains at least:

[
\text{Total Error}
==================

\text{Sampling Error}
+
\text{Discretization Bias}
+
\text{Numerical Error}
]

Sampling error decreases approximately as:

[
O(N^{-1/2})
]

Increasing (N) cannot remove discretization bias.

Experiments must vary both:

* path count (N);
* time-step count (M).

---

## 11. Variance Reduction

### Antithetic Variates

Use paired paths generated by:

[
Z
\quad\text{and}\quad
-Z
]

Estimator:

[
\hat{V}_{\text{anti}}
=====================

\frac{1}{N}
\sum_{i=1}^{N}
\frac{X_i(Z_i)+X_i(-Z_i)}{2}
]

### Control Variates

For control (Y) with known expectation (\mu_Y):

[
\hat{V}_{CV}
============

## \bar{X}

\beta(\bar{Y}-\mu_Y)
]

Optimal coefficient:

[
\beta^*
=======

\frac{\operatorname{Cov}(X,Y)}
{\operatorname{Var}(Y)}
]

Estimate (\beta) without introducing undocumented look-ahead bias.

### Efficiency

Compare estimators using:

[
\text{Efficiency}
=================

\frac{1}
{\operatorname{Variance}\times\operatorname{Runtime}}
]

Also report RMSE per second or runtime required to reach a fixed RMSE.

---

## 12. Finite-Difference PDE

Black–Scholes PDE:

[
\frac{\partial V}{\partial t}
+
\frac{1}{2}\sigma^2S^2
\frac{\partial^2V}{\partial S^2}
+
(r-q)S
\frac{\partial V}{\partial S}
-rV
=0
]

Transform backward time using:

[
\tau=T-t
]

Required schemes:

* explicit;
* implicit;
* Crank–Nicolson.

Crank–Nicolson:

[
\left(I-\frac{\Delta t}{2}L\right)V^{n+1}
=========================================

\left(I+\frac{\Delta t}{2}L\right)V^n
]

where (L) is the spatial discretization operator.

The implementation must define:

* grid spacing;
* terminal condition;
* boundary conditions;
* maximum asset boundary;
* time direction;
* tridiagonal coefficients;
* solver tolerances.

### Vanilla Boundaries

For sufficiently large (S_{\max}):

[
C(0,t)=0
]

[
C(S_{\max},t)
\approx
S_{\max}e^{-q(T-t)}
-------------------

Ke^{-r(T-t)}
]

Equivalent put boundaries must be implemented correctly.

### Stability

The explicit method must detect or demonstrate unstable parameter-grid combinations.

Implicit and Crank–Nicolson stability must not be confused with accuracy.

---

## 13. Finite-Difference Greeks

Central delta:

[
\Delta_h
========

\frac{V(S+h)-V(S-h)}{2h}
]

Central gamma:

[
\Gamma_h
========

\frac{V(S+h)-2V(S)+V(S-h)}{h^2}
]

Central vega:

[
\nu_h
=====

\frac{V(\sigma+h)-V(\sigma-h)}{2h}
]

Experiments must measure the trade-off between:

* truncation error at large (h);
* cancellation and simulation noise at small (h).

Use common random numbers for Monte Carlo bump-and-revalue comparisons.

---

## 14. Pathwise Estimation

For differentiable payoff (\Phi(S_T)):

[
\Delta_{\text{PW}}
==================

e^{-rT}
\mathbb{E}
\left[
\frac{\partial \Phi}{\partial S_T}
\frac{\partial S_T}{\partial S_0}
\right]
]

Under GBM:

[
\frac{\partial S_T}{\partial S_0}
=================================

\frac{S_T}{S_0}
]

For a European call:

[
\Delta_{\text{PW}}
==================

e^{-rT}
\mathbb{E}
\left[
\mathbf{1}_{{S_T>K}}
\frac{S_T}{S_0}
\right]
]

Pathwise estimators must not be applied blindly to discontinuous payoffs.

---

## 15. Likelihood-Ratio Estimation

For density (f(x;\theta)):

[
\frac{\partial}{\partial\theta}
\mathbb{E}_{\theta}[g(X)]
=========================

\mathbb{E}_{\theta}
\left[
g(X)
\frac{\partial}{\partial\theta}
\log f(X;\theta)
\right]
]

Implement at least one validated likelihood-ratio Greek.

Report:

* estimator formula;
* assumptions;
* variance;
* comparison with analytic and finite-difference values;
* instability near extreme parameter regimes.

---

## 16. Heston Calibration

Parameter vector:

[
\Theta=(v_0,\kappa,\theta,\xi,\rho)
]

Primary objective:

[
\min_{\Theta}
\sum_{i=1}^{n}
w_i
\left[
\sigma_i^{\text{model}}(\Theta)
-------------------------------

\sigma_i^{\text{market}}
\right]^2
]

Alternative price-error objectives are permitted if documented.

Constraints:

[
v_0,\kappa,\theta,\xi>0
]

[
-1\leq\rho\leq1
]

Calibration must report:

* objective value;
* residuals by strike and maturity;
* parameter vector;
* initial guess;
* convergence status;
* evaluation count;
* sensitivity to initial conditions;
* synthetic recovery error.

Low pricing error does not prove unique or economically stable parameters.

---

## 17. Implied Volatility

Solve:

[
V_{BS}(\sigma_{\text{imp}})=V_{\text{target}}
]

The solver must:

* enforce valid arbitrage bounds;
* use a bracketed method or justified hybrid;
* report failure explicitly;
* define volatility bounds and tolerances;
* avoid returning a value when no valid root exists.

---

## 18. Mathematical Invariants

### Put–Call Parity

[
C-P=S_0e^{-qT}-Ke^{-rT}
]

### Call Bounds

[
\max
\left(
0,
S_0e^{-qT}-Ke^{-rT}
\right)
\leq C
\leq S_0e^{-qT}
]

### Put Bounds

[
\max
\left(
0,
Ke^{-rT}-S_0e^{-qT}
\right)
\leq P
\leq Ke^{-rT}
]

### Monotonicity

A European call should:

* increase with spot;
* decrease with strike;
* increase with volatility;
* remain non-negative.

### Barrier Relationships

Where contracts and monitoring conventions match:

[
\text{Knock-In}+\text{Knock-Out}
================================

\text{Vanilla}
]

### Limiting Cases

Required checks include:

* (T\to0);
* (\sigma\to0);
* barrier already breached;
* extreme strike;
* zero dividend yield;
* near-zero variance.

---

## 19. Numerical Error Reporting

Use:

### Absolute Error

[
E_{\text{abs}}
==============

|\hat{V}-V_{\text{ref}}|
]

### Relative Error

[
E_{\text{rel}}
==============

\frac{|\hat{V}-V_{\text{ref}}|}
{\max(|V_{\text{ref}}|,\varepsilon)}
]

### Root Mean Squared Error

[
RMSE
====

\sqrt{
\mathbb{E}
\left[
(\hat{V}-V_{\text{ref}})^2
\right]
}
]

Do not use relative error alone when the reference value is near zero.

---

## 20. Floating-Point Requirements

Use double precision as the default numeric type.

You must:

* avoid exact equality checks for floating-point outputs;
* use scale-aware tolerances;
* detect NaN and infinity;
* document cancellation-prone calculations;
* validate extreme inputs;
* prevent silent overflow or invalid square roots;
* distinguish stochastic error from floating-point error.

---

## 21. Acceptance Criteria

The mathematical implementation is complete only when:

* Black–Scholes prices and Greeks match trusted references;
* GBM exact simulation reproduces theoretical moments;
* Euler–Maruyama and Milstein convergence is measured;
* Monte Carlo confidence intervals behave statistically as expected;
* PDE solutions converge toward analytic values;
* explicit instability is demonstrated or detected;
* variance reduction improves measured efficiency where expected;
* Greek estimators are compared by bias, variance, and runtime;
* Heston prices match independent reference values;
* Heston simulation exposes variance-discretization behavior;
* synthetic Heston calibration recovers defensible prices and parameters;
* every published equation corresponds to implemented code;
* all tolerances and assumptions are documented.

---

## 22. Completion Blockers

The mathematical work is not done if:

* formulas are copied without validated implementation;
* convergence is claimed from a single grid or path count;
* Monte Carlo uncertainty is omitted;
* discretization and sampling error are conflated;
* negative Heston variance is silently ignored;
* calibration quality is judged only by objective reduction;
* unstable PDE outputs are presented as valid;
* discontinuous payoffs use unjustified Greek estimators;
* benchmark comparisons use unequal accuracy targets;
* numerical tolerances are arbitrary;
* equations, code, and reported results disagree;
* failed or contradictory experiments are removed instead of explained.

# Heston Semi-Analytic Pricing Methodology

The conventions of the Heston reference engine, documented because
`MATHEMATICAL-SPEC.MD` section 4 requires an implementation to state its
characteristic-function convention, branch handling, integration bounds, quadrature
method, numerical tolerances, and reference values. The engine is
`HestonAnalyticEngine`; the model is `HestonModel`.

## 1. The model

Under the risk-neutral measure:

```
dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
dW_t^S dW_t^v = rho dt
```

The variance is a CIR process. As with the Black–Scholes model, `r` and `q` are
market inputs (in `MarketState`), so one market state serves every model.

## 2. The Feller condition is reported, never enforced

`2 kappa theta >= xi^2` is the condition under which the variance stays strictly
positive. When it fails, the variance can touch zero. The model **reports** whether
it holds — through `feller_ratio()` (the number `2 kappa theta / xi^2`) and
`satisfies_feller()` — and **never rejects** a parameter set for violating it. A
Feller-violating set is a legitimate, frequently-calibrated regime; refusing it
would encode a numerical preference as a validity rule.

The pricing integral is well defined whether or not the condition holds, so the
price is unaffected by the violation. The consequences are the *simulation's* problem
(Phase 10), and the engine surfaces the violation as a warning on the priced result
so a caller carrying the number forward sees it.

## 3. Characteristic-function convention: the little Heston trap

The engine uses the **"little Heston trap"** formulation of Albrecher, Mayer,
Schoutens and Tistaert (2007), not Heston's original (1993) form. The two are
algebraically identical, but they write the complex discriminant `d` and the `g`
factor differently, and that difference is not cosmetic.

For each of the two probabilities `P_1` (with `u_j = +1/2`, `b = kappa - rho xi`) and
`P_2` (`u_j = -1/2`, `b = kappa`):

```
d       = sqrt( (rho xi i u - b)^2 - xi^2 (2 u_j i u - u^2) )
g       = (b - rho xi i u - d) / (b - rho xi i u + d)      <-- little trap: -d in the numerator
C(u)    = (r-q) i u T + (kappa theta / xi^2) [ (b - rho xi i u - d) T
                                               - 2 ln( (1 - g e^{-dT}) / (1 - g) ) ]
D(u)    = (b - rho xi i u - d) / xi^2 * (1 - e^{-dT}) / (1 - g e^{-dT})
phi_j(u) = exp( C(u) + D(u) v_0 + i u ln S_0 )
```

**Why the trap matters.** Heston's original writes `g` with `+d` in the numerator and
uses `e^{+dT}`. As the maturity `T` grows, the term `ln((1 - g e^{dT})/(1 - g))`
crosses a branch cut of the complex logarithm, and the principal-branch value that a
standard `log` returns becomes the *wrong* branch. The price then jumps
discontinuously and is silently wrong for long maturities. The little-trap form keeps
`e^{-dT}` (which decays rather than grows) and the matching `-d` sign, so the
principal branch stays correct for every maturity. This engine's
`StaysCorrectAtLongMaturityWhereTheTrapBites` test checks T = 5, 10, 15 against
mpmath references — exactly where the naive form fails.

## 4. Pricing formula and integration

The call is the two-probability decomposition

```
C = S e^{-qT} P_1 - K e^{-rT} P_2,    P_j = 1/2 + (1/pi) integral_0^inf Re[ e^{-i u ln K} phi_j(u) / (i u) ] du.
```

A **put** is priced from the call by put–call parity (`P = C - S e^{-qT} + K e^{-rT}`),
which is exact and model-independent — no second integral.

**Integration bounds.** The semi-infinite domain `[0, inf)` is mapped to `[0, 1)` by
the substitution `u = x / (1-x)`, `du = dx / (1-x)^2`. No truncation point is chosen
and no tail is discarded: the transform carries the whole domain. The `u -> 0`
removable singularity of the integrand is never evaluated, because the Gauss–Legendre
nodes are strictly interior.

**Quadrature.** Gauss–Legendre on the transformed interval, with the nodes computed
(not tabulated) so the count is a free parameter. The default is 1024 nodes.

## 5. The convergence check: failures never return a plausible number

The price is computed at `N` nodes **and** at `2N`, and their difference is the
reported `integration_error`. If it exceeds `convergence_tolerance` (default `1e-8`),
the price is **refused** with a `ConvergenceFailure`, not returned. A non-finite
intermediate is refused with `NonFiniteValue`, and a materially negative price
(impossible for the contract) likewise.

This is the exit gate's "failures never return plausible values" made concrete. The
semi-infinite integrand's tail is the slow part, and under-resolving it produces a
*smooth, plausible, wrong* price that only the doubling check exposes.

**What converges and what does not.** The default resolves the ordinary and
moderately-hard regimes — across moneyness out to twice the strike, short and long
maturity, Feller-satisfying and not, including deep out-of-the-money at quarter-year
maturity — to well below the tolerance. `IntegrationErrorFallsMonotonicallyWithNodes`
shows the error falling from 6.8e-3 (128 nodes) to 6.2e-11 (1024) for a deep-OTM
quarter-year option. The **pathological corner** — very short maturity (weeks) with
large vol-of-variance (`xi ~ 2`) — does not converge at any practical node count, and
is refused rather than approximated. That refusal is correct behaviour, not a defect.

## 6. Reference values

Validated against three routes that share no code with this engine:

| Case | S,K,T,r,q,kappa,theta,xi,rho,v0 | Reference | Source |
|---|---|---|---|---|
| Fang–Oosterlee | 100,100,1,0,0, 1.5768,0.0398,0.5751,−0.5711,0.0175 | 5.78515543437619 | published COS benchmark |
| Feller-violating | 100,100,1,0.05,0, 2,0.09,1,−0.3,0.09 | 13.1365327960895 | mpmath + QuantLib |
| Long-dated T=10 | 100,100,10,0.03,0, 1.5,0.04,0.6,−0.7,0.04 | 36.44207657516968 | mpmath |

- The **published Fang–Oosterlee (2008) COS benchmark** for the first case;
- a **40-digit mpmath** integration of the same little-trap formula (exact arithmetic,
  but shares this project's reading of the convention);
- **QuantLib's `AnalyticHestonEngine`** — independent production code, a different
  quadrature and a different branch convention. If both had misread the formulation
  the same way, only a disagreement here would reveal it.

The engine matches mpmath and QuantLib to 1e-13/1e-14 on the T = 1 cases and the
published benchmark to 1e-10.

## 7. Limitations

* **European calls and puts under Heston, only.** Path-dependent and American
  payoffs are out of scope for the semi-analytic engine.
* **xi = 0 is refused.** At zero vol-of-variance the variance is deterministic and the
  model is Black–Scholes with the integrated variance — a different computation, not
  this characteristic-function integral. The model admits `xi = 0` (it is a valid
  parameter set); the semi-analytic engine declines it rather than dividing by
  `xi^2`.
* **The reference values are model outputs**, not market observations. The engine is
  validated to be a correct evaluation of the Heston price, not a correct model of any
  market.
* **The pathological short-maturity/large-vol-of-variance corner is refused**, not
  priced. A different method (e.g. a contour-shifted or asymptotic-tail integration)
  would be needed to reach it; this engine reports that it cannot rather than
  guessing.

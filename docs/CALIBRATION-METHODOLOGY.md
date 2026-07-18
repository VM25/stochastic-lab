# Implied Volatility and Calibration Methodology

The conventions of the implied-volatility solver and the Heston calibrator,
documented because `MATHEMATICAL-SPEC.MD` sections 16 and 17 fix the objective, the
constraints, the solver requirements, and what a calibration must report. The engines
are `ImpliedVolatility`, `HestonCalibrator` (`calibrate_heston`), and the experiments
`EXP-11` and `EXP-12`.

## 1. Implied volatility: a bracketed, safeguarded inversion

The solver inverts the Black-Scholes price for the volatility that reproduces a target
price. Because the price is strictly increasing in volatility, a unique root exists
inside the no-arbitrage window, and the method is a **safeguarded Newton iteration
inside a maintained bracket**: a Newton step is taken when it stays in the bracket and
vega is usable, a bisection step otherwise. The bisection fallback guarantees
termination even in the deep wings where vega underflows and Newton stalls.

It refuses rather than invents:

- A target **outside the no-arbitrage price window** — below the discounted intrinsic
  value or above the discounted forward (a call cannot be worth more than the prepaid
  asset) — has no implied volatility. Returned as `RootNotBracketed`, with the window.
- A **non-positive maturity** has no implied volatility: `UnsupportedCombination`.
- A target above the price at the **volatility ceiling** is refused rather than matched
  by an unbounded volatility.
- **Non-convergence** within the iteration budget is `ConvergenceFailure`, never a
  returned best-so-far.

A target at or below the price of the volatility floor is an option quoted at
intrinsic; its volatility is zero, and the solver reports the floor with an explicit
flag rather than chasing a root into a region where the price is flat to machine
precision. That last point matters for the deep out-of-the-money, low-vol, short-
maturity corner, where the price underflows and *no* method can recover the volatility
— the inversion is ill-conditioned there, and the solver says so instead of returning a
confident wrong number.

## 2. Constrained parameters: an unconstrained search that stays feasible

The Heston parameters are constrained (`v0, kappa, theta, xi > 0`, `-1 <= rho <= 1`;
`MATHEMATICAL-SPEC` section 16). A derivative-free optimizer that stepped in the raw
parameters would have to be told about the box at every step. Instead each coordinate
is mapped to the whole real line by a **logit against its bound range**,

```
x = log((p - lo) / (hi - p)),        p = lo + (hi - lo) / (1 + e^{-x}),
```

so the optimizer searches an unconstrained space and every point it proposes maps back
strictly inside the box. The inverse is total — any real point lands in the box — so no
step can leave the feasible region. The forward map requires the point strictly inside
and refuses a parameter sitting on a bound, which has no finite image.

## 3. The objective, and why it is finite everywhere

The primary objective is the **weighted sum of squared implied-volatility residuals**
(`MATHEMATICAL-SPEC` section 16); a weighted price-error objective is also available and
is cheaper, since it needs no per-evaluation inversion, but it weights the deep,
high-priced quotes far more heavily than the wings.

The objective is made **finite everywhere feasible**, deliberately. A parameter set our
quadrature cannot resolve at some quotes — common at high vol-of-variance, where the
characteristic-function integrand is oscillatory and the doubling-convergence check
fails — would otherwise return an infinite objective and wall a derivative-free
optimizer onto a plateau it cannot leave. Instead **each unpriceable or uninvertible
quote adds a finite penalty**, large against a typical squared residual but finite, so
the optimizer descends on the quotes that do resolve. The true parameters price
cleanly, so the penalty never moves the minimum — only shapes the path to it — and the
penalised-quote count is reported so a fit that leaned on penalties is not read as
clean. This was not a cosmetic choice: before it, three of four diverse starts on a
20-quote surface froze on the infinite plateau and never moved.

## 4. The optimizer: derivative-free and deterministic

Calibration uses a classical **Nelder-Mead downhill simplex**: no derivatives, no
randomness, so a calibration is reproducible from its configuration. A non-finite
objective is treated as `+infinity` so the simplex retreats from it, and
non-convergence within the budget is a reported status (`MaxIterationsReached`) with
the best point returned — never an error to swallow, because "did not converge" is a
finding the calibration gate turns on. The evaluation count is reported as the honest
cost of the search.

**Multiple initial guesses** are the intended use, not an option. A single guess can
land in a local minimum; several diverse guesses find the global one and, by landing in
different places or the same, are the evidence on whether the fit is unique.

## 5. Four things kept apart

The calibrator reports, and never conflates, the four things the catalog insists on
separating. A low objective is **not** on its own read as success.

- **Optimizer convergence** — a per-start Nelder-Mead status.
- **Surface-fit quality** — the weighted objective, the price and implied-volatility
  RMSEs, and the residual surface by strike and maturity.
- **Parameter recovery** — against a synthetic truth, the normalised distance of the
  fit from it (`EXP-11`).
- **Identifiability** — whether materially different parameter vectors fit about as
  well. Starts whose objective is within a factor of the best yet sit far apart in
  normalised parameter space are flagged, and their parameters preserved as evidence.

## 6. EXP-11: synthetic recovery, demonstrated blind

`EXP-11` generates a synthetic surface from known parameters and calibrates it from
several diverse guesses. The catalog names three failure conditions —
success judged by objective reduction alone, the true parameters used as the only
guess, and synthetic recovery omitted — and the experiment excludes all three
structurally: at least two guesses are required, at least one must sit away from the
truth, and the **recovery verdict is read only from a "blind" start** that did not begin
near the answer. A surface fit well by parameters far from the truth is reported as a
**warning about non-identifiability**, not a pass; no blind start converging is
**inconclusive**, not a wrong answer.

Recovery is not uniform across the five parameters. On a surface of European prices,
`v0` and `theta` trade off against each other and `kappa` against `xi`, so the
correlation and the variance level are pinned far more tightly than the mean reversion.
The per-parameter errors show which, and this is stated as a limitation, not hidden.

## 7. EXP-12 and the static surface: stability, not recovery

`EXP-12` calibrates **one fixed surface** under several deliberate variations of the
setup — weighting the quotes uniformly, toward the money, and toward the wings;
tightening the bounds; dropping the wing strikes — and reports the **dispersion of the
calibrated parameters** across them. A parameter that barely moves is stably identified
by this surface; one that swings is weakly identified, and that is reported rather than
smoothed into a single fit. Every scenario's residual surface is kept.

The surface is the one stored in `configs/calibration/market_surface.json` and
reproduced by `EXP-12`: a **documented synthetic reference**, generated from a known
Heston model and frozen with a stated timestamp, with its implied volatilities rounded
to four decimals. It is emphatically **not real market data**, and the record says so —
no economic reading is placed on it, only a statement about how this estimator behaves
on this surface. On a synthetic surface generated by Heston itself, the fit can be
nearly exact and the dispersion tiny; a real surface carries model error that this
study, by construction, does not.

## 8. Limitations

- Synthetic recovery bounds the calibration's accuracy under the model's own
  assumptions. A real surface is not generated by Heston, so a good synthetic recovery
  does not promise a good real-market fit.
- The stability scenarios are a fixed, deliberately small menu; a wider set of
  weightings and bounds could expose instability this one does not.
- Stability across setups is not identifiability. Parameters can move together in a way
  that keeps the fit good, so a small per-scenario RMSE with a large dispersion is
  exactly the weak-identification signal the study exists to surface — the two are
  reported separately for that reason.

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

## 7. The two datasets: a synthetic one for recovery, a real one for stability

The phase uses **two** surfaces, and they are not interchangeable.

The **synthetic** surface (`configs/calibration/synthetic_surface.json`, and the grid
`EXP-11` generates) exists for **controlled recovery**: it is produced by a known
Heston model, so the right answer is known and the question is whether the calibrator
gets it back. A near-perfect fit is expected there, and it is explicitly labelled
synthetic — never "market".

The **real** surface (`data/market/spy_options_2026-07-17.json`) is a genuine market
snapshot: SPY option closing prices observed at the 2026-07-17 close, retrieved from a
market-data provider, with its underlying, observation timestamp, spot, rate and
dividend assumptions, source, and licensing all documented in the file. SPY options are
American; only out-of-the-money options are included (puts below spot, calls above), the
side where the early-exercise premium is small, and they are treated as European — a
documented approximation.

### Dataset validation

Before any calibration, the real dataset is validated (`validate_dataset`) against the
documented cleaning rules, and every exclusion is reported with its reason rather than
dropped: a non-positive price is missing; a quote whose day volume is below the
threshold is stale; a price outside the no-arbitrage window is arbitrageable; a price
that cannot be inverted has no implied volatility; and an implied volatility outside a
sane band is rejected. The calibration runs on exactly the surviving quotes, and the
excluded-quote count and coverage travel with the record, so the surface a reader audits
is the surface the fit used. (Because the dataset carries last-trade closes rather than
bid/ask, there is no bid/ask crossing to check; that is noted.)

### EXP-12: stability on the real surface

`EXP-12` calibrates the validated real surface under variations of **all four axes the
catalog names** — the initial guesses, the weighting of the quotes, the parameter
bounds, and the strike/maturity subset — and reports the **dispersion of the calibrated
parameters** across them. For each scenario it reports, separately: optimizer
convergence, the fit RMSE, the residual structure by strike and maturity, and the
penalized-quote count. Because the surface is *not* generated by Heston, the fit carries
genuine model error: the residuals are systematic, with Heston underfitting the steep
put wing and overfitting the call wing of a real equity skew, and the RMSE is a real
number, not a rounding artifact.

## 8. The penalty, its sensitivity, and clean-success gating

The finite penalty (§3) must do two things at once, and both are required and tested.
It must **prevent the optimizer freezing** on an infinite plateau where the quadrature
cannot resolve some quotes — which it does, because the priceable quotes still give a
descent direction. And it must **not make a bad region look competitive**: a fit that
leaves `k` quotes unpriceable carries at least `k · quote_penalty` in its objective, and
with the penalty set well above a good fit's squared-residual objective, any fit that
leans on penalties scores far worse than a clean one, so the clean fit wins. The penalty
is configurable (`quote_penalty`) precisely so this trade-off can be studied rather than
assumed.

The consequence is a hard rule: **a calibration that relied on penalties is never
reported as a clean success.** The result carries a `relied_on_penalties` flag — true
whenever the best fit left any quote unpriceable — and the `calibrate` command and
`EXP-12` both drop such a fit out of clean/pass status into a warning, with the
penalized-quote count shown. A low objective bought partly with penalty mass is not a
good fit, and the pipeline refuses to present it as one.

## 9. Limitations

- Synthetic recovery bounds the calibration's accuracy under the model's own
  assumptions. A real surface is not generated by Heston, so a good synthetic recovery
  does not promise a good real-market fit — which is exactly why the real surface is
  calibrated separately.
- The real dataset is a small end-of-day snapshot of last-trade closes, not a
  synchronous bid/ask quote set; closes across strikes are not perfectly simultaneous,
  and that non-synchroneity adds noise the residual RMSE partly reflects.
- The stability scenarios are a fixed menu across the four axes; a wider set could
  expose instability this one does not.
- Stability across setups is not identifiability. Parameters can co-move in a way that
  keeps the fit good, so a small per-scenario RMSE with a large dispersion is exactly
  the weak-identification signal the study exists to surface — the two are reported
  separately for that reason.

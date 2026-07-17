# Barrier Monitoring Methodology

How DiffusionWorks measures the cost of observing a barrier at fixed dates rather
than continuously, and why several of the obvious ways to measure it are wrong.

Companion to `CONVERGENCE-METHODOLOGY.md` and `PDE-CONVERGENCE-METHODOLOGY.md`.
The evidence is `results/EXP-07.json`.

## 1. The monitoring convention is a contract term

A barrier option's price depends on when the barrier is looked at. This is not a
numerical detail that a fine enough grid removes — it is part of what the contract
says, and two contracts differing only in it are two different contracts with two
different prices.

DiffusionWorks therefore makes the convention explicit in the instrument
(`MonitoringConvention`), refuses to default it, and refuses to let an engine
silently substitute one for another:

| Convention | Meaning | Priced by |
|---|---|---|
| `Continuous` | Watched at every instant | `BarrierAnalyticEngine` (Reiner–Rubinstein) |
| `Discrete` | Watched at `m` equally spaced dates | `BarrierMonteCarloEngine` |
| `BrownianBridge` | Discrete dates, plus the probability of crossing between them | `BarrierMonteCarloEngine` |

Almost no traded barrier is continuously monitored; most are observed at daily
closes. The textbook closed forms price the continuous contract. The gap between
them is what EXP-07 measures, and it is larger than most modelling choices.

`BarrierMonteCarloEngine` **refuses** `Continuous`. No simulation observes a
barrier at every instant, and monitoring on a very fine grid and reporting the
result as continuous would report a discretely monitored price under the wrong
name. Section 4 shows how far off "very fine" actually is.

## 2. Two discretisation errors, and why conflating them makes both unmeasurable

A simulated barrier price carries two distinct errors:

1. **Path discretisation** — the simulated path is not the true path.
2. **Monitoring discretisation** — the barrier is observed at `m` dates rather
   than continuously.

Only the second is EXP-07's subject. The engine eliminates the first *entirely*
rather than making it small: log-GBM is Brownian motion with drift, so the exact
scheme samples the true joint law of the price at any finite set of dates, however
coarse. Stepping exactly at the `m` monitoring dates therefore introduces no path
error at all.

What remains is monitoring bias and Monte Carlo noise, nothing else. An Euler path
on a finer grid would add a bias of its own, and the monitoring bias would then be
measured *through* it.

This is a limitation as well as a design: EXP-07 says nothing about a production
simulation that uses a finer path grid or an inexact scheme.

## 3. The Brownian-bridge correction

Between two observed log-prices `a` and `b` over an interval of length `dt`, the
minimum of the bridge falls below a log-barrier `lb` with probability

```
p = exp( -2 (a - lb)(b - lb) / (sigma^2 dt) )
```

when both endpoints are above `lb`, and 1 when either is at or below it. The
up-barrier case is the reflection, and since both factors change sign the product —
and hence the formula — is unchanged.

**This is exact for this model, not an approximation.** Log-GBM *is* Brownian
motion with drift, and the drift cancels out of the bridge law once both endpoints
are conditioned on. That is why the correction converges to the continuous price
rather than to something near it, and why it works at coarse frequencies rather
than only fine ones. Under Heston — or any model whose log-price is not Brownian
motion with constant coefficients — the bridge law is an approximation and EXP-07
says nothing about how good a one.

It corrects the discretisation of the **observation**, not of the **path**. A
bridge applied to an Euler path would be correcting the wrong one.

### The bridge needs its own random stream

`StreamPurpose::BarrierBridge = 2`, not a continuation of `AssetShock`.

The crossing probability is a *function of* the asset shocks that produced the
interval's endpoints. Drawing its uniform from that same stream would test each
interval's excursion against a number derived from that same interval. The
dependence would be invisible in any single price — the result would look
perfectly reasonable — and would corrupt the correction exactly where it matters:
near the barrier, where `p` is neither 0 nor 1.

## 4. Measuring the bias

### Multi-seed, not single-seed

The bias is a difference of two numbers, one of which carries sampling error. A
single seed cannot separate a real bias from a lucky draw.

This is not a formality. A single-seed exploration at 400k paths showed the bridge
drifting positive at fine monitoring: +0.024, +0.037, +0.035 at m = 50, 100, 250.
Each was individually within 1.6 standard errors, but they *trended*, which reads
as a real effect. Across 16 seeds the same cells measure +0.007, +0.007, +0.010 —
all within 2 standard errors, no trend. The apparent drift was one seed's draw
appearing three times, because the same seed produces correlated estimates across
monitoring frequencies.

EXP-07 replicates every cell across 16 seeds and reports the **across-seed**
standard error, which is the estimator's realised dispersion rather than its
self-reported one.

### Resolution, and refusing to fit noise

A bias is **resolved** when it clears 3 across-seed standard errors. Below that,
the experiment has measured nothing, and saying otherwise reports a draw.

**No order is fitted unless every level is resolved.** A power law fitted to
unresolved levels returns a slope and an interval regardless, and both are
meaningless.

This is not hypothetical. At B = 70 (1.78 σ√T below spot) the bias never clears
2.1 standard errors at any frequency. The first EXP-07 run fitted all six levels
anyway and published:

```
order -0.1866, 95% interval [-0.6998, +0.3267]
```

which reads as a confident finding that the bias *grows* as the barrier is watched
more often — fitted entirely to six draws from zero. A reader has no way to tell
that from a measurement. The fit is now refused and the record states that the
bias was not resolved.

## 5. Why the fitted order is not the measured order

Theory (Broadie–Glasserman–Kou) says the discrete-monitoring bias decays as
`O(1/sqrt(m))`: an order of 0.5 against the monitoring interval.

Almost no measured interval contains 0.5 — and, revealingly, **they miss it from
both sides**:

| Barrier | Distance | Full-range fit | Asymptotic window (finest 3) |
|---|---|---|---|
| down 70 | 1.78 σ√T | *refused — unresolved* | — |
| down 80 | 1.12 σ√T | 0.248 [0.152, 0.343] | 0.308 [−0.451, 1.068] ✓ |
| down 90 | 0.53 σ√T | 0.395 [0.368, 0.422] | 0.433 [0.419, 0.448] |
| down 95 | 0.26 σ√T | 0.437 [0.425, 0.449] | 0.455 [0.404, 0.505] ✓ |
| up 105 | 0.24 σ√T | 0.630 [0.595, 0.665] | 0.580 [0.571, 0.589] |
| up 110 | 0.48 σ√T | 0.552 [0.543, 0.560] | 0.543 [0.456, 0.630] ✓ |
| up 120 | 0.91 σ√T | 0.487 [0.475, 0.499] | 0.500 [0.427, 0.572] ✓ |
| up 140 | 1.68 σ√T | 0.426 [0.395, 0.456] | 0.463 [0.302, 0.624] ✓ |

(✓ = the 95% interval contains 0.5.)

Taken at face value this is a refutation of the theory. It is not one, and the
distinction matters more than the numbers do.

**The two-sided miss is itself the argument.** Down-barriers come in low and close
up-barriers come in high. A study of one direction only would have read its own miss
as a signed correction to theory — "the real order is about 0.4" — and been wrong in
a way no amount of extra care *within* that direction would have exposed. What is
actually happening is that the higher-order terms the asymptotic statement discards
carry opposite signs for the two geometries.

**The evidence that the range is pre-asymptotic, not the theory wrong:**

1. **The local orders climb monotonically toward 0.5 without arriving.** At B = 90:
   0.342, 0.375, 0.390, 0.431, 0.435. An error of `C·h^k + higher-order terms` has
   local orders that climb toward `k` as the contamination drains away. This is the
   same signature seen in Phases 5 and 6.

2. **The asymptotic windows recover 0.5 at five of the seven fitted arms**, including
   both extremes of barrier distance.

3. **The misses fall on both sides of 0.5**, which no single-signed error in the
   method could produce.

4. **The B = 90 window still excludes 0.5 despite being well resolved**, because its
   residuals are small enough that the interval is narrower than the model's own
   misspecification. A tight interval around a slightly wrong slope is not evidence
   that the slope is right; it is evidence that a straight line is the wrong model
   for this data. The same effect appeared in Phase 6's interpolation study.

**What settles it is a check that does not depend on a fit at all.** See section 6.

The reportable statement is: *the bias decays at a rate approaching 0.5 from below
over the frequencies tested*. Not that it decays at 0.5, and not that theory is
wrong.

Reaching the asymptotic range would need frequencies well beyond daily, which no
traded contract uses.

## 6. The continuity correction: predicting the size, not the rate

Broadie–Glasserman–Kou also predict the bias's **magnitude** in closed form. A
barrier observed at `m` dates behaves, to leading order, like a continuously
monitored barrier moved *away* from the spot:

```
B_effective = B * exp( -beta * sigma * sqrt(dt) )     (lower barrier)
B_effective = B * exp( +beta * sigma * sqrt(dt) )     (upper barrier)

beta = -zeta(1/2) / sqrt(2*pi) = 0.5825971579390106
```

The reason is overshoot: a discretely observed path that is going to breach is
first *seen* past the barrier, not at it, and the mean overshoot is
`beta·sigma·sqrt(dt)`.

This is a far sharper test than a fitted slope, for two reasons. It predicts the
actual number rather than a trend, and DiffusionWorks did not fit it to the data —
it comes from `-zeta(1/2)/sqrt(2*pi)` and a barrier shift.

Measured, **for down-barriers**: the continuity-corrected reference lands within
**2 across-seed standard errors** of the discrete price at **16 of the 18 resolved
cells**.

**A theory with the rate wrong could not predict the size to within its noise
across three barriers and six frequencies.** The fitted slope is contaminated; the
theory is not.

For up-barriers it does not land, and that is section 6.2. This is why section 5's fits are reported as a pre-asymptotic
measurement rather than a refutation — and the point generalises: when a fitted
order disagrees with theory, an independent prediction of the *magnitude*
distinguishes "my fitting range is bad" from "the theory is wrong", where more grid
levels often cannot.

### Read the residual, not the explained fraction

`bias_explained_fraction` is intuitive and unreliable. It divides by the bias, so
where the bias is small the ratio is noisy and misleads. At B = 80 it ranges from
67% to 109% across frequencies, which reads as the correction breaking down — while
every residual there is within 1.9 standard errors of zero and therefore consistent
with the correction being exactly right.

`continuity_corrected_residual_over_se` is the statistic that decides. Both are
published; only one of them means anything.

## 6.2 Where the correction fails, and why that is the good part

The correction is itself an approximation with error `o(1/sqrt(m))`. Its residual is
expected to be resolved *somewhere*, and where that happens is a measurement of the
correction's own accuracy rather than an embarrassment.

**Down-barriers.** Two of the 18 resolved cells disagree, both at B = 95 — the
closest barrier, watched least often. The residual across the six frequencies:

```
m       5      12      25      50     100     250
res  +33.0    +6.2    -1.1    -0.4    +0.6    +1.0    (across-seed SE)
```

At m = 5 the correction is off by +0.184, unmistakably resolved at 33 standard
errors, and it drains away exactly as an `o(1/sqrt(m))` remainder must.

**Up-barriers: the correction is resolved as wrong at 23 of the 24 cells.**
Systematically negative — it over-predicts — and still −5.4 σ at daily monitoring on
the nearest barrier:

```
B=105   m:    5      12      25      50     100     250
           -275.0  -150.4   -52.9   -30.4   -10.8    -5.4   (across-seed SE)
```

The one cell that does not disagree is B = 140 at m = 250 (−1.5 σ) — the farthest
up-barrier at the finest monitoring, exactly where an `o(1/sqrt(m))` remainder should
have drained away. The exception is the theory behaving as advertised, not a let-off.

**This is not a sign error.** The shift direction was checked independently against
mpmath: moving the barrier *away* from the spot gives 0.01531 against a measured
0.01471, while reversing it gives 0.00459 — off by an order of magnitude rather than
a few percent. The sign is right; the remainder is real.

Two things make it visible, and they are worth separating:

1. **The noise collapses, so the same error becomes resolvable.** An up-and-out call
   near the spot knocks out on almost every path and pays zero, so the estimator's
   variance is tiny. The absolute residual is 0.0006–0.008 — comparable to the
   down-barrier case — but it is 0.1% of a down-barrier price and 4% of this one.
   *Significance measures what the run can see, not how large the error is.*

2. **The geometry is genuinely harder.** An up-and-out call pays `B − K` *just below*
   its barrier and zero at it: the value is **discontinuous across the boundary**. A
   down-and-out call struck above its barrier approaches zero smoothly there. A
   continuity correction built on an overshoot argument is weakest exactly where the
   value jumps.

So EXP-07 does not merely use the continuity correction as an oracle. It locates
where the oracle stops being trustworthy — and the answer is not where a
down-barrier-only study would have suggested. **Do not use the continuity-corrected
reference for an up-and-out call near the spot.** Use the bridge, which is exact for
this model.

## 7. What the numbers say

At S = K = 100, r = 0.05, q = 0, T = 1, down-and-out call, 200 000 paths × 16 seeds.

**Discrete monitoring is biased high in every one of the 42 resolved cells, and the
size is worst exactly where the contract is cheapest:**

| Barrier | Distance | Continuous price | Bias at daily (m=250) | As % of price | Resolution |
|---|---|---|---|---|---|
| down 70 | 1.78 σ√T | 10.45 | not resolved | — | max 2.1 SE |
| down 80 | 1.12 σ√T | 10.35 | +0.029 | +0.28% | 4 SE |
| down 90 | 0.53 σ√T | 8.67 | +0.261 | +3.01% | 37 SE |
| down 95 | 0.26 σ√T | 5.64 | +0.569 | +10.09% | 83 SE |
| up 105 | 0.24 σ√T | **0.0088** | +0.0059 | **+67.4%** | 53 SE |
| up 110 | 0.48 σ√T | 0.119 | +0.036 | +30.1% | 69 SE |
| up 120 | 0.91 σ√T | 1.176 | +0.150 | +12.8% | 106 SE |
| up 140 | 1.68 σ√T | 5.711 | +0.219 | +3.8% | 44 SE |

The up-and-out at B = 105 is the sharpest statement in the experiment. It knocks out
in precisely the states where it would have paid — the payoff window is only
S_T ∈ (100, 105) — so what survives is worth 0.0088, and **daily monitoring biases
that by two thirds of its own value**. The monitoring convention's cost is not
uniform: it concentrates in the cheap, nearly-knocked-out contracts, where a relative
error is largest and intuition is weakest.

**Volatility moves it as directly as frequency does** (B = 90, daily):

| σ | Bias | As % of price |
|---|---|---|
| 0.1 | +0.020 | +0.30% |
| 0.2 | +0.261 | +3.01% |
| 0.4 | +0.974 | +10.03% |

What matters is `sigma·sqrt(dt)` — the distance the price can wander unobserved —
not the calendar.

**The bridge removes the bias rather than shrinking it, in both directions.** No
bridge cell shows a bias this run can resolve; the largest across all 48 is 2.4
across-seed standard errors, against a discrete arm reaching 563. It holds at m = 5 as well as at
m = 250: five observations with the bridge beat 250 without it.

## 8. The PDE arm: pricing the continuous contract directly

Everything above is about *monitoring* — the gap a discrete observation schedule
opens against the continuously monitored contract, and the bridge that closes it.
The PDE arm asks a different question and must not be confused with it: **it prices
the continuous contract itself**, by solving the Black–Scholes equation with an
absorbing Dirichlet boundary at the barrier (`V(B, τ) = 0`), and measures how fast
that solve converges to the analytic reference as the grid refines.

The distinction is the discretisation being studied. The Monte Carlo arm
discretises the *observation* — how often the barrier is looked at. The PDE arm
discretises the *solution* — the grid and the time step. One is a property of the
contract; the other is numerical error against a fixed continuous price. Conflating
"the monitoring is coarse" with "the grid is coarse" would be exactly the kind of
category error `EXP-07` exists to prevent, so the two arms are labelled and reported
separately, and anchored to one number: the same Reiner–Rubinstein value the
monitoring arm measures its bias against.

**Measured (Crank–Nicolson, Rannacher smoothing, node count and time steps refined
together, so the fitted order is the joint spatial-and-temporal one):**

| Barrier | Fitted order [95% CI] | Finest relative error |
|---|---|---|
| down 70 | 2.001 [2.000, 2.002] | 1.5e-5 |
| down 80 | 2.000 [2.000, 2.001] | 1.7e-5 |
| down 90 | 2.001 [1.999, 2.004] | 1.4e-5 |
| down 95 | 2.525 [1.223, 3.826] | 8.8e-6 |
| up 105  | 1.661 [−0.086, 3.409] | 8.5e-5 |
| up 110  | 2.339 [1.490, 3.188] | 8.5e-6 |
| up 120  | 2.000 [1.998, 2.003] | 3.9e-6 |
| up 140  | 2.003 [1.827, 2.179] | 4.6e-6 |

Six of the eight land on 2.000 with tight intervals — the barrier introduces no
order loss, which is the point: a misplaced Dirichlet boundary would show up here as
first-order convergence, and it does not. The barrier alignment being *mandatory*
(section on `AssetGrid::with_barrier_on_node`) is what buys that.

The two ragged fits, down 95 and up 105, are the barriers nearest the spot
(0.26 and 0.24 σ√T), and the raggedness is in the *coarse* levels, not the fine
ones — up 105's local orders run 0.07, 2.23, 2.49. An up-and-out struck at the spot
with its barrier 5% above pays only on `S_T ∈ (100, 105)` and is worth 0.0088; that
payoff window is a handful of cells wide on a coarse grid, so the coarse error is
dominated by barely resolving it, and the order only settles once the grid is fine
enough to see the window. Both still clear the second-order floor the arm enforces,
and both reach relative errors below 1e-4. This is the PDE mirror of the
monitoring arm's finding that the near-spot contracts are where every method is
most strained.

**The arm is a validation, not just a measurement.** It fails the whole `EXP-07`
record if any fitted order drops below 1.5, because that is what a real defect —
a barrier boundary half a cell out of place, a wrong live-index range — would look
like. The arm passing is a standing regression on the engine, not a one-time check.

## 9. Limitations

* **Calls only.** Puts are a separate set of cases the analytic engine does not
  implement, so no trusted continuous reference exists for them.
* **The reference is a formula, not a market.** Reiner–Rubinstein agrees with
  mpmath to 1e-15 and QuantLib to 1e-9, so it is a trustworthy statement about the
  model — but every bias here is measured against a model output.
* **The 48 bridge cells share 16 seeds**, so they are not 48 independent tests.
  Their sampling errors are correlated across barriers and frequencies — visibly
  so: the largest positive deviation sits at m = 50 for three of the four barriers.
  The evidence that the bridge is unbiased is weaker than the cell count suggests.
* **B = 70 measured nothing.** The bias there is real but below this run's
  resolution. Read as "negligible at these path counts", not "zero".
* **Barriers very close to the spot are not characterised.** The nearest tested are
  0.24–0.26 σ√T. Closer barriers knock out nearly every path, leaving a surviving
  sample small enough that the estimator's variance is dominated by rare paths. The
  up-and-out at B = 105, worth 0.0088, is the edge of what this run resolves.
* **The bridge estimator draws a Bernoulli decision per interval** rather than
  accumulating the conditional survival probability. Both are unbiased; the
  conditional-expectation form would have strictly lower variance. The bridge arm's
  intervals are wider than necessary as a consequence of that choice, not of the
  correction.

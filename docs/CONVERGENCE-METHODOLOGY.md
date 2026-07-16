# Convergence methodology

Implementation note, not a specification. `docs/EXPERIMENT-CATALOG.MD` states what
EXP-01 through EXP-04 must establish; this records *how* they establish it, and why
several obvious approaches do not work. Written because the failures below are not
visible in the results — a wrong method here produces a clean plot and a confident
number.

## 1. A fitted slope needs an uncertainty, and an uncertainty needs the right instrument

`FAILURE-MODES` section 6 names a convergence order judged by eye as a completion
blocker. So every order in this project is fitted by ordinary least squares on
log–log axes and reported with a Student-t interval on the slope.

**Student-t, not normal.** A convergence study has a handful of grid levels. At
four points there are two degrees of freedom, where the t critical value is 4.303
against the normal's 1.960 — the interval is more than twice as wide. Using the
normal quantile would produce intervals that exclude the true order routinely.

**Centred sums, not the textbook identity.** `Sxx = Σx² − n·x̄²` is algebraically
correct and numerically wrong here: log-spaced abscissae are large in magnitude and
narrow in spread, exactly the case where those two terms cancel. The same applies to
`SSres = Syy − b·Sxy`, which on a near-perfect fit — what a clean convergence study
is — can return a small negative number whose square root is `NaN`.

**Three points minimum.** Two points determine a line exactly, leaving no residual
and a standard error of zero. That slope would appear to be the best-determined
quantity in the study while resting on no evidence at all, so it is refused rather
than returned.

## 2. The regression interval is the wrong test for exact data

The weak errors for `f(S)=S` and `f(S)=S²` are computed in closed form (§4). Fitting
them gives slope **0.9997** against a true order of exactly **1** — and the verdict
initially came back `inconsistent`.

The machinery was wrong, not the scheme. A regression interval is built from the
residual scatter, which estimates *sampling noise*. Exact data has none: the
residual is at rounding level, the interval collapses to a point, and containment
becomes a test no true order could ever pass.

Exact studies are therefore judged against a tolerance (`kAnalyticOrderTolerance =
0.01`) sized by the only real source of slack — the neglected higher-order term,
which contributes O(dt) to the fitted slope and stays below 1e-3 on these grids. It
sits 50× below the 0.5 gap between the candidate orders, so it cannot confuse order
½ with order 1.

The general principle: **match the error bar to the error's actual source.** A
number carries a sampling interval only if it was sampled.

## 3. Pre-asymptotic levels are arithmetic, not a defect

Fitted over the full grid, the measured strong orders fall short of theory and their
intervals *exclude* it:

| Scheme | Full-range fit | Asymptotic window | Theory |
| --- | --- | --- | --- |
| Euler–Maruyama | 0.4932 [0.4884, 0.4979] | **0.5001 [0.4988, 0.5013]** | 0.5 |
| Milstein | 0.9637 [0.9436, 0.9839] | **0.9941 [0.9873, 1.0010]** | 1.0 |

Neither interval is wrong. A regression interval quantifies how well the *line* is
determined, and the line really does have slope 0.4932 — the model `e = C·dt^k` is
simply misspecified at coarse dt, where the neglected terms are not negligible.

The evidence that this is pre-asymptotic contamination rather than a wrong order is
the **local order** between adjacent levels, which climbs monotonically toward
theory as the grid refines (Milstein: 0.854 → 0.906 → 0.945 → 0.979 → 0.988 →
1.005). A scheme that genuinely converged at order 0.96 would show local orders
scattered *about* 0.96, not marching to 1.

Three consequences, all load-bearing:

1. **The asymptotic window is declared in the configuration**, before the run.
   Choosing it after seeing which window lands on the theoretical value would be
   choosing the answer.
2. **Both fits are always published**, and the verdict `consistent_asymptotically`
   exists precisely to say "the full range disagrees and here is why."
3. **The grid runs to M = 1024 because Milstein needs it.** A higher-order scheme
   enters its asymptotic regime *later* — at M = 256 Milstein still reads 0.985.
   Truncating the grid to save time yields an honest measurement of a
   pre-asymptotic slope and an apparent contradiction with theory.

## 4. The weak error must not be measured the obvious way

The natural approach — simulate, compare the mean against Black–Scholes — does not
work. The weak bias is O(dt); the Monte Carlo standard error is O(1/√N). At M = 64
the bias is ~2e-3 against a standard error of ~5e-2. **The measurement reports
noise.** Fitting it yields slopes of −0.61 and +0.32 for a quantity whose true order
is 1.

Three fixes, in ascending order of strength:

**Pairing (9× better).** The exact scheme samples the true terminal law, so
`E[f(S_T^exact)] = E[f(S_T)]` identically. Therefore

```
E[f(S^Δ)] − E[f(S_T)]  =  E[ f(S^Δ) − f(S^exact) ]
```

Same estimand; but on shared Brownian paths the common noise cancels. This is a
control variate with β = 1 and an analytically known control mean — the Phase 4
machinery arriving at the same place from a different direction.

**Restricting the range.** Even paired, the resolution depends on the scheme's
*strong* order, and for Euler it degrades as the grid refines:

| Scheme | Paired difference | Standard error | bias / se |
| --- | --- | --- | --- |
| Euler–Maruyama | O(√dt) | O(√dt/√N) | O(√(dt·N)) — **falls with dt** |
| Milstein | O(dt) | O(dt/√N) | O(√N) — **constant in dt** |

Measured, at 4×10⁶ paths: Euler's resolution falls 13.6 → 7.0 across M = 8 → 64,
while Milstein's sits at 783.0, 783.0, 783.2, 783.5 — flat to four figures. The
machinery reproduces its own prediction, which is the best evidence available that
the estimator does what the derivation says.

The consequence is that **Euler sets the usable range**, not Milstein. The
call-payoff arm is fitted over 8 ≤ M ≤ 64 for that reason, and the
`noise_dominated` verdict exists to refuse a fit rather than report a slope through
unresolved points.

**Closed forms (exact).** For `f(S)=S` and `f(S)=S²` no simulation is needed at all.
Each scheme's step multiplies the state by a factor `u` independent of the state, so
`E[S_M^k] = S_0^k·(E[u^k])^M`, and:

```
Euler:    E[u] = 1 + a·dt          E[u²] = (1+a·dt)² + σ²·dt
Milstein: E[u] = 1 + a·dt          E[u²] = (1+a·dt)² + σ²·dt + (σ⁴/2)·dt²
```

The bias then halves per grid doubling to four significant figures. That
**establishes** weak order 1 rather than estimating it.

### The result worth reading twice

**Euler and Milstein share their first moment exactly.** The Milstein correction
`(σ²/2)(ΔW²−Δt)` has mean zero, so it cannot move `E[S]` at any step count.
Milstein's strong order 1 buys **nothing whatever** in the weak error of the mean.
It is pinned as an exact equality in `tests/experiments/scheme_moments_test.cpp`
rather than left as a remark, and the published records show the two schemes'
`E[S]` errors agreeing bit-for-bit at every grid level.

It gets sharper. On the call payoff Milstein's weak error is not merely no better
but **about six times worse**, and of the opposite sign:

| Scheme | Weak error at M = 8 | Fitted order |
| --- | --- | --- |
| Euler–Maruyama | +1.40 × 10⁻² | 0.92 |
| Milstein | −8.23 × 10⁻² | 0.99 |

Both converge at order 1; the *constants* differ, and here they favour the scheme
with the worse strong order. Nothing is contradictory about this — strong order
measures pathwise tracking, weak order measures how well an expectation is
reproduced, and a scheme can excel at one while trailing at the other.

The practical statement: **choose a scheme for the kind of accuracy the
application needs, not for the larger of its two order numbers.** Milstein is the
right choice for a pathwise problem — a barrier, or a Greek by common random
numbers. For a plain expectation it costs an extra term per step and, on this
payoff, returns a worse answer.

## 5. Nested sweep levels break the fit's independence assumption

The first run of EXP-01 **failed**. Every one of eight scenarios fitted a slope
steeper than the theoretical −0.5, and four intervals excluded it outright:

```
atm_call   -0.5551  [-0.5834, -0.5269]   inconsistent
```

The estimator was fine. **The interval was lying.**

A run of N paths at a given seed uses path indices `0..N-1`. Sweeping N with the
seed held fixed therefore *nests* the levels — the N=4000 run re-uses every path
the N=1000 run used. The levels' errors are then correlated, which violates the
independence OLS assumes of its residuals, and the reported interval comes out too
narrow around whatever slope the correlated draw happened to produce.

Measured across ten independent master seeds:

| Levels | Mean slope | SD across masters | Intervals covering 0.5 |
| --- | --- | --- | --- |
| Nested | 0.5111 | 0.0225 | 10/10 |
| Disjoint | 0.5118 | **0.0070** | 10/10 |

Both are unbiased — the rate was never wrong — but nesting inflates the slope's
real dispersion by **3.2×**, none of which a nested run's own interval knows about.
The original failure was a narrow interval around a fluke.

The fix costs nothing: the stream is addressed by coordinates, so giving each level
its own seed region is just a different address. Each level now records the seed
range it occupied, and `SamplingConvergenceLevelsDoNotSharePaths` asserts the
ranges are disjoint — checked exactly on the addresses rather than through a
statistic that could agree by chance.

This also removed an inconsistency: EXP-02 had always drawn a fresh path per grid
level, and its limitations said so, while EXP-01's levels were nested.

The general lesson is the same as §2. **An interval is only as honest as the
assumptions behind it**, and a violated assumption does not announce itself — it
produces a confident number.

## 6. Coupling is what makes a strong error a strong error

The strong error compares a discretised path against the exact path **driven by the
same Brownian increments** — same seed, same path index, same stream coordinates,
differing only in the stepping rule. `EXPERIMENT-CATALOG` names schemes using
different Brownian paths as a failure condition for EXP-02.

The counter-based generator makes this structural rather than a convention: a path
is a pure function of `(seed, purpose, path, position)`, so two generators at the
same coordinates *cannot* consume different shocks. Were the paths decoupled, the
"strong error" would be the expected absolute difference of two independent samples
— an O(1) quantity that does not converge at all, and which would still produce a
plot, a slope, and an interval.

`tests/experiments/convergence_test.cpp` verifies it by ratio: the coupled
difference is >100× smaller than the same difference across independent seeds.

## 7. What is not established

- **GBM only.** Both strong orders are theoretical results under globally Lipschitz
  coefficients. GBM satisfies that; Heston's square-root diffusion does not. Nothing
  here transfers to the Heston arm of EXP-04, which is completed in Phase 14.
- **Terminal values only.** A pathwise supremum error would be the stronger
  statement and is not measured. It is what a barrier payoff is sensitive to
  (EXP-09).
- **Rates, not constants.** Two methods can share an order and differ by orders of
  magnitude in the error at any fixed resolution. That comparison belongs to EXP-05.
- **Levels are independent, not nested.** Each level draws its own paths, which is
  what the fit assumes (§5). It also means the study cannot exploit the variance
  reduction a nested design would offer if the correlation were modelled properly.
- **EXP-04's work metric is a proxy.** `paths × steps` ignores per-path overhead and
  memory traffic, so its efficient frontier is indicative, not a performance claim.
  Cost is measured properly in Phase 13.

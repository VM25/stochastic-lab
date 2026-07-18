# Heston Simulation Methodology

The conventions of the Heston path simulator, documented because the discretisation
of a CIR variance is not unique and the choice changes the answer. The engine is
`HestonMonteCarloEngine`; the model is `HestonModel`; the study is `EXP-10`. The
semi-analytic engine documented in `HESTON-PRICING-METHODOLOGY.md` is the reference
this simulator is measured against.

## 1. The model and the difficulty

Under the risk-neutral measure the spot and its variance evolve as

```
dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
dW_t^S dW_t^v = rho dt
```

The variance is a CIR process. In continuous time it is non-negative, and strictly
positive when the Feller condition `2 kappa theta >= xi^2` holds. A naive Euler
discretisation preserves neither property: a single large negative variance shock can
send the discretised variance below zero, and then `sqrt(v)` is not a real number.
This is the whole problem the scheme below exists to solve, and the reason a Heston
simulator cannot be written by pointing an Euler stepper at the SDE.

## 2. Full-truncation Euler

The engine uses the **full-truncation** scheme of Lord, Koekkoek and van Dijk (2010).
Writing `v^+ = max(v, 0)`, the variance is stepped as

```
v_{n+1} = v_n + kappa (theta - v_n^+) dt + xi sqrt(v_n^+ dt) Z^v
```

and the spot in log form, with the same truncated variance in the Itô correction and
the diffusion,

```
ln S_{n+1} = ln S_n + (r - q - v_n^+ / 2) dt + sqrt(v_n^+ dt) Z^S.
```

The **state** `v_n` is allowed to go negative and is carried forward unchanged; only
its truncation `v_n^+` is ever fed to a square root or a drift. Two consequences
follow. First, no square root ever sees a negative argument, so a negative variance
can never produce a non-finite path — the scheme is closed. Second, because the state
is not itself floored, the truncation does not inject the upward drift that absorption
(flooring the state at zero) or reflection (taking `|v|`) do, so full truncation has
the smallest bias of the simple fixes. Lord et al. compare the family and full
truncation wins; this engine takes that result rather than re-deriving it.

The log-Euler spot step keeps the discounted spot a martingale in expectation: given
the variance at the start of a step, `E[exp(-v^+ dt/2 + sqrt(v^+ dt) Z^S)] = 1`, so
`E[S_{n+1} | F_n] = S_n e^{(r-q) dt}` exactly. Put-call parity therefore holds up to
the sampling error in the estimate of `E[S_T]`, which is a regression test on the
engine.

## 3. Correlation

`Z^S` and `Z^v` carry the model's correlation `rho`. They are **not** drawn from one
stream. Each path draws two independent standard normals, one from a variance-shock
stream and one from an asset-shock stream, and the pair is combined by the Cholesky
factor: `Z^v` passes through unchanged and `Z^S = rho Z^v + sqrt(1 - rho^2) Z_perp`.
Keeping the variance's own driving draw fixed as `rho` varies means a study that turns
correlation on or off changes only the coupling, not the variance path's noise — the
same stream-purpose separation the rest of the project uses (`StreamPurpose`).

## 4. The diagnostics: invalid variance behaviour is measured, not hidden

Every priced result carries the state of the variance process that produced it:

- **`negative_variance_fraction`** — the fraction of variance steps whose
  pre-truncation value came out negative and was floored. Zero does not certify
  correctness, but a large fraction is the signal that the discretisation is straining,
  and it rises sharply as the Feller condition is violated.
- **`minimum_variance`** — the most negative pre-truncation variance any path-step
  reached. It is the depth of the excursion the truncation had to absorb.
- **`non_finite_paths`** — paths that produced a non-finite spot or variance.

The pre-truncation values are recorded **before** any flooring, so these describe the
scheme's real excursion rather than its cleaned-up state.

## 5. A non-finite path blocks the price

If any path goes non-finite, the price is **blocked** — the engine returns a failure,
not an average over the survivors. Averaging the survivors would report a clean number
for a simulation that diverged on part of its sample, which is exactly the silent
failure `FAILURE-MODES.MD` forbids. Under full truncation this never happens from a
negative variance; the block exists for genuine overflow and as the honest response
when the naive scheme (below) is asked to price a regime it cannot.

The simulation is also exposed through `simulate()`, which returns the diagnostics
alongside an optional price: absent when a path went non-finite. This lets `EXP-10`
quantify a scheme's failures without the failure erasing the evidence, while the
pricing entry point `price()` still treats a non-finite path as the hard error it is.

## 6. The naive scheme is a diagnostic, never a price

The engine can also run an unguarded Euler scheme (`HestonVarianceScheme::NaiveEuler`)
that feeds the raw, possibly-negative variance straight into the square root. It
exists only as `EXP-10`'s baseline — the thing whose failure justifies full
truncation — and is never used to price. The CLI pricing path is always full
truncation.

Its failure in EXP-10 is **regime- and configuration-specific**, not a universal
impossibility. The scheme fails whenever a step drives the variance negative and the
next square root is taken; that is common in the tested regimes and path counts, but a
milder regime, a finer step, or fewer paths could avoid a negative variance entirely
and let the naive scheme return a price. The claim is "it failed for the tested
configurations," never "a naive Euler can never price a Heston path."

## 7. The bias is real, empirically first-order here, and reported

Full truncation is biased: the discretised price differs from the true Heston price by
an amount that decays with the step. `EXP-10` measures it against the semi-analytic
reference across a Feller-satisfying and a Feller-violating regime, and fits the decay
order **only over the steps where the bias clears the sampling noise** — fitting the
noise-dominated fine levels would fit the sampling error, not the bias, which the
"never fit unresolved data" discipline refuses.

In the default study (`configs/experiment/heston_simulation.json`, spot 100, strike
100, r 5%, T 1, v0 = theta = 0.04, kappa 2, rho -0.7):

- **Feller-satisfying (xi = 0.3, ratio 1.78):** the bias is below the sampling noise
  at every step tested, so no decay order is fitted — reported as unresolved, not as
  zero. Full truncation prices it with no non-finite paths; the naive scheme fails to
  price it at any step count, because even a rare negative variance loses the path.
- **Feller-violating (xi = 1.0, ratio 0.16):** the bias is of order one at five steps
  and falls to within a sampling error or two of the reference by 320 steps. The
  fitted decay order is ≈ 1.09 with a 95% interval of about [1.0, 1.2] — consistent
  with first order. This is **empirical evidence for this regime and step range, not a
  universal convergence theorem** for full-truncation Euler under Heston; a different
  regime or asymptotic window could measure a different order. The pre-truncation
  variance is floored on a quarter to a half of steps and dips well below zero, and
  full truncation absorbs it with zero path failures. The naive scheme loses up to
  ~95% of its paths here.

The Feller condition is reported, never treated as automatic invalidity: the violating
regime is priced, with its bias measured and its uncertainty attached.

## 8. Limitations

- Full truncation is one of several fixes for the CIR positivity problem. This study
  establishes that it avoids the naive scheme's failures and that its bias decays; it
  does not rank it against reflection, the quadratic-exponential (QE) scheme, or an
  exact Broadie–Kaya sampler.
- The bias decay order is fitted only where the bias resolves above the sampling
  noise. A regime whose bias never clears the noise yields no order — a statement
  about resolution at that path count, not evidence the scheme fails to converge.
- The naive scheme's collapse is a statement about the unguarded square-root Euler
  scheme specifically, the one a first attempt reaches for — not about every
  conceivable naive implementation.

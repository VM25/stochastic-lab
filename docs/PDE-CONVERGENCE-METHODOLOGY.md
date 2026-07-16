# PDE convergence methodology

Implementation note, not a specification. `docs/BUILD-PLAN.MD` Phase 6 and
`docs/EXPERIMENT-CATALOG.MD` EXP-06 state what the finite-difference engine must
establish; this records how, and why the obvious measurement is wrong.

## 1. A joint sweep measures neither order

The finite-difference error has two independent sources:

```
error ≈ C_S · dS^p  +  C_t · dtau^q
```

Refining `dS` while holding `N_t` fixed does **not** measure `p`. It measures a
mixture, and the mixture happens to look clean whenever `C_t · dtau^q` is small —
which is a property of the parameters chosen, not of the scheme.

The first sweep run here made exactly that mistake. It held `N_t = 200` and refined
`N_S`, saw error ratios of 3.78, 4.02, 4.01, and concluded "second order in dS."
The ratios were real; the conclusion was not licensed, because nothing had
established that the temporal error at `N_t = 200` was negligible against the
spatial error being measured.

So the two are separated:

**Space sweep.** Hold `N_t` large enough that the temporal error is negligible,
refine `N_S`, fit `log(error)` against `log(dS)`. Then **check the premise**:
double `N_t` and confirm the answer barely moves. If it moves by an appreciable
fraction of the smallest measured error, the sweep was a mixture and its order is
not a spatial order.

**Time sweep.** Hold `N_S` fixed and measure error against a reference **on the
same spatial grid** at a very large `N_t`. The spatial error then cancels exactly
rather than sitting underneath as a floor the fit would mistake for a convergence
stall. Fit `log(error)` against `log(dtau)`.

Every order below is fitted with `statistics/regression`, with a Student-t interval
on the slope, exactly as the Phase 5 studies are. An order read off a ratio is an
order judged by eye, which `FAILURE-MODES` section 6 names as a completion blocker.

## 2. Measured orders

Configuration: `S_0 = K = 100`, `r = 0.05`, `q = 0`, `sigma = 0.2`, `T = 1`,
`S_max = 400` (4K), strike aligned to a node. Analytic value
**10.450583572186**.

### Space, at N_t = 8000 (dtau = 1.25e-4)

| Scheme | dS = 4 → 0.5 | Fitted order | 95% CI | R² |
| --- | --- | --- | --- | --- |
| Crank–Nicolson | 3.98e-2 → 6.18e-4 | **2.003** | [1.998, 2.009] | 1.00000 |
| Implicit | 3.99e-2 → 7.49e-4 | 1.916 | [1.779, 2.053] | 0.99945 |

**The negligibility check passes for Crank–Nicolson and fails for implicit.**
Doubling `N_t` moves the Crank–Nicolson answer by 2.9e-09 — nothing, against a
smallest spatial error of 6.2e-04. It moves the implicit answer by **6.6e-05**,
about 9% of its smallest spatial error. The implicit scheme is first order in time,
so its temporal error decays slowly and is still present at `N_t = 8000`.

The implicit figure of 1.916 is therefore **a mixed spatial-temporal order, not a
spatial one**.

Driving the contamination out recovers the true order, and the measured order
tracks the contamination as it goes:

| N_t | dtau leakage at N_S=801 | Fitted space order | 95% CI | R² |
| --- | --- | --- | --- | --- |
| 8,000 | 8.8% of the finest error | 1.9159 | [1.779, 2.053] | 0.99945 |
| 32,000 | 2.5% | 1.9799 | [1.939, 2.021] | 0.99995 |
| 128,000 | **0.7%** | **1.9974** | [1.983, 2.012] | 0.99999 |

So **both schemes are second order in space**. The implicit scheme's apparent
shortfall was temporal leakage throughout, and the negligibility check is what
established that rather than leaving it as a plausible story. Reporting 1.916 as
"the implicit spatial order" would have been reporting a property of the chosen
`N_t`.

### Time, at N_S = 401, against a same-grid reference

| Scheme | dtau = 0.1 → 0.00625 | Fitted order | 95% CI | R² |
| --- | --- | --- | --- | --- |
| Implicit | 1.04e-1 → 6.51e-3 | **1.001** | [0.997, 1.004] | 1.00000 |
| Crank–Nicolson | 1.18e-1 → 9.71e-6 | 3.551 | [2.019, 5.083] | **0.948** |
| Crank–Nicolson + 2 Rannacher steps | 1.76e-2 → 7.20e-5 | **1.984** | [1.976, 1.992] | 1.00000 |

The implicit scheme is textbook first order in time.

**Plain Crank–Nicolson is not cleanly second order in time on this payoff**, and
the fit says so rather than hiding it: R² = 0.948 and a 95% interval spanning
[2.02, 5.08] is a regression reporting that it has been handed something that is
not a straight line. The errors show why —

```
dtau  0.1000   0.0500   0.0250   0.0125   0.00625
err   1.18e-1  1.28e-2  1.39e-4  3.88e-5  9.71e-6
ratio     9.2     92.1      3.6      4.0
```

— a factor of **92** between the second and third points, then ratios of 3.6 and
4.0. Two regimes, not one. The coarse end is the startup oscillation: the payoff's
kink is non-smooth data, and Crank–Nicolson's amplification factor tends to −1 for
the highest-frequency modes, so those modes are oscillated rather than damped.

Restricted to the asymptotic window past the ringing (`N_t` 40 to 160), the fit
gives **1.920 [1.336, 2.505], R² = 0.99943** — consistent with second order, and
with an interval wide enough to be honest about resting on three points. Both fits
are reported, as in the Phase 5 studies: the full-range number is not wrong, it is
a faithful description of a curve that is not a straight line.

## 3. Rannacher smoothing, measured rather than assumed

Two fully implicit steps before switching to Crank–Nicolson restore clean second
order in time: **1.984 [1.976, 1.992], R² = 1.00000**, with monotone errors and no
regime change. The implicit scheme's amplification factor tends to 0, so those
first steps annihilate the high-frequency modes; Crank–Nicolson then proceeds on
data that is effectively smooth.

It is an **explicit, opt-in option**, defaulting to zero, recorded in the
diagnostics, and refused on schemes it does not apply to. A Crank–Nicolson run that
quietly took implicit steps is not Crank–Nicolson, and reporting its order as
Crank–Nicolson's would be reporting a measurement of a different method.

The cost is visible in the table: at `dtau = 0.1` the smoothed scheme's error is
1.76e-2 against plain Crank–Nicolson's 1.18e-1, but the smoothed error at
`dtau = 0.00625` is 7.20e-5 against plain's 9.71e-6. Rannacher is better where it
matters (the coarse, practical end) and worse in the deep asymptotic limit, because
those two first-order steps never stop contributing their first-order local error.
Neither scheme dominates; which is better depends on the step count.

## 4. What none of this establishes

- **Stability is not accuracy.** The implicit scheme steps a year in one step, stays
  bounded and smooth, and is wrong by more than 0.1. Unconditional stability means
  the iteration does not diverge. It says nothing about the answer.
- **An M-matrix is not accuracy.** The sign structure guarantees a non-negative
  price from a non-negative payoff. A monotone, positivity-preserving scheme can be
  monotone, positivity-preserving, and badly wrong.
- **A small residual is not accuracy.** The tridiagonal residual is rebuilt
  independently and is a genuine check on the solve, but on an ill-conditioned
  system it can be tiny while the error is large.
- **These orders are one configuration.** They are measured at one strike, one
  volatility, one maturity, with the strike aligned to a node and `S_max = 4K`.
  EXP-06 sweeps those; the numbers here are the method, not the result.

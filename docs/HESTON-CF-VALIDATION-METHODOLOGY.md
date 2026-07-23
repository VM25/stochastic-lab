# Heston Characteristic-Function Validation Methodology

How EXP-09 validates the Heston characteristic-function pricing, and — the point the
catalog name insists on — how it validates the **characteristic function itself**, not only
the prices built from it. The experiment is defined in `docs/EXPERIMENT-CATALOG.MD` (EXP-09);
the engine and its conventions are in `docs/HESTON-PRICING-METHODOLOGY.md`.

## 1. Why the characteristic function directly

The price is a quadrature of the characteristic function. Validating only the price leaves
the object it is built from unchecked: a price can be right for the wrong reasons, or a
characteristic function can be subtly mis-normalised in a way a single price hides. So the
engine now exposes `HestonAnalyticEngine::log_price_characteristic_function` — evaluating
`phi_j(u) = E^j[e^{i u ln S_T}]` for the two-probability decomposition, with a **complex**
argument so identities off the real axis can be checked — and the experiment tests it
directly.

## 2. Three kinds of evidence, kept separate

The catalog's exit gate is that prices match independent references. It is tempting to call
every check an "independent oracle", but they are not the same kind of evidence, and the
record labels them honestly:

### A. Internal analytic properties of the characteristic function

Properties `phi` must have *because* it is the characteristic function of a real log-price
under a risk-neutral measure. These are self-consistency checks, not external oracles:

- `phi_j(0) = 1` for both measures;
- conjugate symmetry `phi(-u) = conj(phi(u))` (the log-price is real);
- the martingale identity `phi_2(-i) = S_0 e^{(r-q)T}` — the forward, which pins the drift;
- `|phi| <= 1`, finiteness, and continuity across a dense frequency grid.

They hold to near machine precision across a wide regime sweep (strike, maturity, correlation,
vol-of-variance, mean-reversion), including the Feller-violating regimes. Passing them says
the function is self-consistent and correctly normalised — **not** that it prices correctly.

The continuity check is grid-resolution-aware: the smooth adjacent step is `~|E[ln S_T]| * du`
from the `e^{i u ln S_0}` carrier, so the bound scales with the grid spacing `du`. A genuine
branch-cut jump — the "trap" the little-trap formulation avoids — is `O(1)` and does not shrink
with `du`, so it is caught as the grid refines.

### B. External references (price recovery)

Whether the engine prices correctly is the separate question, answered against trusted
external values, each labelled by where it comes from:

- **Published:** the Fang–Oosterlee (2008) COS-method benchmark, `5.78515543437619`. This is
  the one genuine literature value.
- **Independently generated:** the Feller-violating and long-maturity "trap" cases, computed
  to 40 digits with mpmath and cross-checked against QuantLib's `AnalyticHestonEngine`. These
  are strong external references but **not** literature benchmarks, and are not labelled as
  such. QuantLib provides the cross-check but is not linked into the experiment binary; the
  agreement was established when the constants were produced, and the live QuantLib
  cross-validation lives in the dedicated external-validation target.

### C. Internal numerical checks

- **Analytic invariants** (a sub-kind, neither external prices nor CF identities): put-call
  parity, exact by construction, and the Black-Scholes limit as vol-of-variance vanishes.
- **Integration-node convergence:** the doubling error falls as the quadrature is refined,
  down to the floating-point floor (a wiggle at the floor is rounding noise, not
  non-convergence, and is allowed).
- **Failure handling:** a pathological deep-out-of-the-money, very short maturity, large
  vol-of-variance corner that does not converge at a practical node count is **refused** with
  the exact `ConvergenceFailure` status, rather than returned as a plausible number. A failure
  that stops is the correct behaviour there.

## 3. Status

`pass` requires all of it: the characteristic-function identities within their tolerances,
every external reference within its stated tolerance, put-call parity and the Black-Scholes
limit within theirs, integration convergence to the floor, and the pathological corner refused
with the exact status. Any one failing fails the record.

## 4. Limitations

- Only one reference is a published literature value; the rest are independently generated to
  high precision and labelled accordingly.
- The characteristic-function properties are checked on a finite, if wide, regime sweep and a
  finite frequency grid. They are exact identities, so passing everywhere sampled is strong
  evidence, but the sweep does not prove them for every parameter.
- The martingale identity is evaluated at the single complex point `u = -i`; it pins the first
  moment (the drift), not higher moments, which the price recovery covers instead.
- The pathological corner shows the integral has regimes it cannot resolve; the experiment
  documents that the engine refuses them rather than hiding it.

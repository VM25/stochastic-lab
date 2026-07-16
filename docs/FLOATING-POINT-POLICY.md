# Floating-Point and Reproducibility Policy

## Status

This is an **implementation note**, not a specification. It records what the code
actually guarantees, as measured. The authoritative requirements live in
`PROJECT-SPEC.MD`, `TECHNICAL-DESIGN.MD`, and `VALIDATION-PLAN.MD`; this document
says how far those requirements are met and where the limits are.

It exists because "reproducible" is not one property, and stating it without
qualification would claim more than the project has earned.

---

## 1. Build settings

Floating-point contraction is **disabled** globally:

| Toolchain | Flag |
|---|---|
| GCC, Clang | `-ffp-contract=off` |
| MSVC | `/fp:precise` |

Set in `CMakeLists.txt` via `CMAKE_CXX_FLAGS`, so it reaches the published
`build_flags` metadata on every result. A test
(`BuildInfoTest.DisclosesThatFloatingPointContractionIsDisabled`) asserts it is
present, so the guarantee is checked rather than trusted.

No fast-math, no reassociation, no flush-to-zero. These are not enabled anywhere
and must not be.

### Why contraction is off

Two reasons, one of them discovered rather than anticipated.

**Cross-compiler reproducibility.** GCC defaults to `-ffp-contract=fast` and
Clang to `=on`. Both decide *per expression* whether to fuse `a*b+c` into an FMA.
The same source then produces different low-order bits on different toolchains,
while CI builds and compares both. A result that depends on which compiler
produced it is not reproducible (VALIDATION-PLAN section 19).

**It changes convergence, not merely the last digit.** The incomplete beta
continued fraction in `src/statistics/distributions.cpp` is a fixed-point
iteration. Under contraction its residual stalls near `1e-14` and never reaches
the convergence criterion, so Student-*t* at large degrees of freedom failed
outright — an ordinary 100,000-path Monte Carlo run could not obtain a confidence
interval. Measured: contraction on, no convergence in 300 iterations at ν = 10⁵;
contraction off, converged in 35.

### Cost

FMA is unavailable in numeric kernels. **Not yet measured.** Phase 13 measures it
and reports the figure here. Until then no claim is made about the cost, in
either direction.

---

## 2. Reproducibility tiers

Different parts of the pipeline carry different guarantees. The tests assert each
at its actual strength; see `tests/random/reproducibility_test.cpp` and
`data/references/random_stream_golden.json`.

### Tier 1 — bit-exact on every conforming platform

| Component | Why |
|---|---|
| `Philox4x64::generate` | Integer arithmetic only: multiply, xor, add on `uint64`. No floating point is involved, so no platform, compiler, or optimisation level can change it. |
| `uniform_from_bits` | Every step is exact. The shift is integer; converting a 52-bit integer to `double` is exact; adding `0.5` is exact in that binade (spacing `0.5`); scaling by `2^-52` is a power of two. Nothing rounds. |

These are asserted with **exact equality**, and validated bit-for-bit against the
published Random123 known-answer vectors and independently against
`numpy.random.Philox`.

### Tier 2 — bit-exact on any IEEE-754 platform, given the build settings above

| Component | Why |
|---|---|
| `inverse_norm_cdf`, central branch (`|p − 0.5| ≤ 0.425`) | A rational polynomial using only `+ − * /`. Each is correctly rounded by IEEE-754, and with contraction disabled the evaluation order is fixed by the expression tree. |
| Black–Scholes price/Greek arithmetic, excluding its `exp`/`log`/`erfc` calls | Same reasoning. |

**Measured: 85.0% of normal draws take the central branch** (1,000,000 draws;
`|u − 0.5| ≤ 0.425` by construction of the uniform).

### Tier 3 — bit-exact within a fixed platform and toolchain; tolerance across

| Component | Dependency |
|---|---|
| `inverse_norm_cdf`, tail branches (~15% of draws) | `std::log` |
| `norm_cdf` | `std::erfc` |
| `student_t_cdf`, `student_t_quantile` | `std::lgamma`, `std::log`, `std::exp` |
| Everything downstream: normals, prices, statistics | inherits the above |

IEEE-754 specifies correct rounding for `+ − * / sqrt` and **not** for the
transcendental functions. glibc, macOS libm, and musl may each return a different
last bit. Typical disagreement is under 1 ulp, but it is not zero and it is not
guaranteed.

Consequently:

* **A stored result reruns bit-for-bit on the same platform and toolchain.** This
  is the guarantee that matters for reproducing a published number, and it is
  tested (`EverythingIsBitReproducibleWithinOneBuild`).
* **Across platforms, results agree to a tolerance**, not bit-for-bit. Tests
  compare tier-3 quantities with a documented relative tolerance (~1e-14 for
  normals).

---

## 3. What is *not* claimed

Stated explicitly, because the absence of a claim is easy to miss:

* **Not** bit-identical results across operating systems or libm
  implementations. Only tiers 1 and 2 are platform-independent.
* **Not** full IEEE-754 reproducibility of the whole engine. IEEE-754 does not
  specify the transcendental functions the engine depends on, so no build setting
  can deliver this.
* **Not** reproducibility across different reduction orders. Floating-point
  addition is not associative; summing paths in a different order changes the
  last bits. Phase 12 fixes the reduction order deterministically and measures
  the residual difference.
* **Not** a measured performance cost for disabling contraction. Phase 13.

Where a result must be reproduced exactly, reproduce it on the recorded platform.
Every artifact carries its compiler, version, flags, OS, and CPU for this reason.

---

## 4. Coordinate space and collision safety

A random draw is a pure function of four coordinates:

```
value = f(master_seed, purpose, path, position)
```

mapped onto Philox as:

```
key     = (master_seed, purpose)
counter = (position / 4, path, 0, 0)
lane    = position mod 4
```

**Collision-freedom is structural, not statistical.** Philox is a bijection on
`(counter, key)`, so distinct pairs give distinct 256-bit outputs. The map above
is injective: `master_seed`, `purpose`, `path`, and `position / 4` are each
recoverable from the pair, with the lane selecting a word inside the block. Two
distinct coordinate tuples therefore cannot address the same value. There is no
birthday bound, no seeding heuristic, and no assumption that separately seeded
generators are independent.

Tested directly in `StreamCoordinatesTest.DistinctCoordinatesGiveDistinctDraws`
and `NeighbouringCoordinatesDoNotAlias` — the latter including the transposition
case `(seed, path)` vs `(path, seed)`, which a symmetric mixing would alias.

### Limits

| Coordinate | Maximum | Note |
|---|---|---|
| `master_seed` | 2⁶⁴ − 1 | full word |
| `purpose` | 2⁶⁴ − 1 | full word; values frozen, see below |
| `path` | 2⁶⁴ − 1 | 1.8×10¹⁹ paths; ~585 years at 10⁹ paths/second |
| `position` | 2⁶⁴ − 1 | per path |

**No narrowing occurs.** Every coordinate is a full `uint64_t` and none is cast to
a narrower type. `position / 4` cannot overflow `counter[0]`: dividing a `uint64`
by 4 yields at most 2⁶² − 1, two bits inside the word. Asserted in
`StreamCoordinatesTest.LimitsSpanTheFullWordAndDoNotOverflow`.

### Frozen purpose identifiers

| Purpose | Value |
|---|---|
| `AssetShock` | 0 |
| `VarianceShock` | 1 |
| `Diagnostic` | 1000 |

These are **part of the reproducibility contract**. Changing one silently changes
every stored result that used it while leaving the code looking correct. New
purposes take new values; existing ones are never renumbered and gaps are never
reused. Pinned by literal in
`StreamCoordinatesTest.PurposeIdentifiersAreFrozen`.

---

## 5. Known numerical limitations

Measured, not assumed. Each has a test asserting its shape, so a regression that
worsened it would fail rather than pass quietly.

### Deep out-of-the-money option prices underflow

For sufficiently extreme moneyness the true price falls below
`numeric_limits<double>::denorm_min()` (≈4.9e-324) and the engine returns exactly
zero. This is the correctly rounded result — no double represents the value — and
is established by a log-domain Mills-ratio bound rather than inferred from the
computed price.

**Limitation:** at that moneyness the engine cannot distinguish a worthless option
from one worth 1e-1146. Resolving them needs a log-domain or extended-precision
pricer, which Version 1.0 does not provide and does not claim.

### Student-*t* accuracy degrades with degrees of freedom

| ν | Worst relative error |
|---|---|
| ≤ 100 | ~1e-15 |
| 1000 | ~1e-12 |
| 10000 | ~5e-12 |

Cause: cancellation in the prefactor `lgamma(a+b) − lgamma(a) − lgamma(b)`. At
ν = 10⁴ the terms are ≈37582 with relative error ~1e-16, hence absolute error
~4e-12; their difference is only ≈3.6, so that becomes a ~1e-12 relative error,
which `exp` carries through.

**Immaterial in use:** a confidence half-width is itself a statistical estimate
carrying ~0.2% relative uncertainty at 100,000 paths, so a 1e-12 error in the
critical value is nine orders of magnitude below the noise it multiplies. It is
asserted anyway, per-regime, in
`StudentTAccuracyDegradesPredictablyWithDegreesOfFreedom`. **No normal-approximation
fallback is used**; such a fallback would need to be designed and validated on its
own terms before it could be trusted.

### The right tail of `norm_cdf` saturates

`N(x)` reaches exactly 1.0 from x ≈ 8.3, correctly: the spacing of doubles near 1
is 2.2e-16 while `N(−8.3)` ≈ 5.2e-17, so no double lies between the true value and
1. Past that point `1 − N(x)` is exactly zero, while `N(−x)` still resolves the
same probability 180 orders of magnitude further down. The degradation begins
earlier — at x = 8, `1 − N(8)` is already ~7% wrong.

This is why the engine evaluates `N(−d)` directly rather than `1 − N(d)`, and why
put delta is `−e^{−qT}N(−d₁)` rather than the algebraically identical
`e^{−qT}[N(d₁) − 1]`.

---

## 6. Where each guarantee is tested

| Claim | Test |
|---|---|
| Contraction disabled and disclosed | `BuildInfoTest.DisclosesThatFloatingPointContractionIsDisabled` |
| Philox bit-exact | `PhiloxKatTest.ReproducesEveryKnownAnswerVector`, `ReproducibilityTierTest.PhiloxWordsAreBitExact` |
| Uniforms bit-exact | `ReproducibilityTierTest.UniformsAreBitExact` |
| Normals within tolerance of the true quantile | `ReproducibilityTierTest.NormalsMatchTheTrueQuantilesWithinTolerance` |
| Bit-reproducible within one build | `ReproducibilityTierTest.EverythingIsBitReproducibleWithinOneBuild` |
| Coordinates do not collide | `StreamCoordinatesTest.DistinctCoordinatesGiveDistinctDraws` |
| No aliasing of neighbours or transpositions | `StreamCoordinatesTest.NeighbouringCoordinatesDoNotAlias` |
| No narrowing or overflow | `StreamCoordinatesTest.LimitsSpanTheFullWordAndDoNotOverflow` |
| Purpose values frozen | `StreamCoordinatesTest.PurposeIdentifiersAreFrozen` |
| Student-*t* degradation bounded per regime | `DistributionReferenceTest.StudentTAccuracyDegradesPredictablyWithDegreesOfFreedom` |
| Deep-OTM underflow is graceful and provably correct | `BlackScholesLimitsTest.UnrepresentablyDeepOutOfTheMoneyCallUnderflowsToZero` |

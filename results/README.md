# Experiment results

Machine-readable records for the fifteen experiments in `docs/EXPERIMENT-CATALOG.MD`.
Each `EXP-NN.json` is a complete record — configuration, results, summary table,
status, interpretation, limitations, reproduction command, and the build provenance
that produced it. `EXP-NN.csv` is the same summary table, for plotting.
`MANIFEST.json` records every artifact with a SHA-256 checksum and names the commit
that generated them.

The whole set was regenerated in one pass from the stored configurations at generator
commit `3846fb3`, from a clean tree. Every record carries that commit and
`git_dirty: false`, so the fifteen results are mutually comparable rather than a
sediment of runs from different revisions.

## Regenerating

```
./build/release/src/diffusionworks experiment \
    --config configs/experiment/convergence.json --id EXP-02 \
    --output results/EXP-02.json
```

Each record names its own `reproduction_command`. The CSV is derived from the record
rather than produced by a second run:

```
python3 python/derive_csv.py results/EXP-02.json
```

That matters. `diffusionworks experiment --format csv` re-runs the experiment, so a
CSV produced that way describes a *different execution* than the JSON beside it and
disagrees with it on every runtime column. Deriving the CSV from the record's own
summary table makes the pair provably one run.

The figures, all of which regenerate from the committed records:

```
python3 python/plot_convergence.py results/EXP-*.json --outdir docs/figures
python3 python/plot_experiments.py results/EXP-*.json --outdir docs/figures
```

`plot_convergence.py` owns EXP-01–04; `plot_experiments.py` owns EXP-05–15. Each
renders what it recognises and skips the rest, so both are run over the whole set.
Both are reporting only (ADR-002): every number is read from a record, never
recomputed, so a plotting bug cannot change a published result.

Auditing the set, and re-deriving the prose below from the artifacts:

```
python3 python/generate_manifest.py --check
python3 python/verify_claims.py
```

## Status

Ten pass, five warn, none fail.

| Experiment | Status | Headline |
| --- | --- | --- |
| EXP-01 | pass | Monte Carlo RMSE decays as N^(−1/2) across all eight scenarios; every fitted interval covers −0.5. |
| EXP-02 | **warning** | Euler–Maruyama attains strong order ½ and Milstein order 1 in the asymptotic window (0.4996, 0.9957). The full-range fits fall short (0.4974, 0.9838) — the coarse levels are pre-asymptotic. |
| EXP-03 | pass | Both schemes converge weakly at order 1. Euler and Milstein share their first moment *exactly*. |
| EXP-04 | pass | The bias floor exists and is located: on coarse grids more paths stop helping. |
| EXP-05 | pass | Work-normalised efficiency, not wall-clock. The control variate is worth ~1730× crude on the arithmetic Asian; antithetic sampling is worth ~3.3× on the European, where no control exists. |
| EXP-06 | pass | Crank–Nicolson reaches order 2.0034 in space; Rannacher restores order 1.9838 in time where plain CN oscillates. The explicit stability bound is sufficient, not necessary — stable to ratio 1.5, divergent at 2. |
| EXP-07 | **warning** | Daily monitoring is not continuous monitoring: the bias reaches 67% of price for an up-and-out call near the spot, and 792% at the coarsest frequency. The Brownian bridge removes it. But one arm's bias never resolved and the continuity correction is measurably wrong for up-barriers — see below. |
| EXP-08 | pass | No Greek estimator wins everywhere. Under common random numbers the finite-difference **delta** dispersion exponent is 0.01, not 1; **gamma** is 0.53, not 2. Pathwise beats likelihood-ratio delta by 2.4× at the median but has no gamma; likelihood-ratio survives a discontinuous payoff. |
| EXP-09 | pass | The characteristic function satisfies φ(0)=1, conjugate symmetry to machine zero, and the martingale identity to 5.6e−16 across 162 regimes; it reproduces the published COS value and four independently generated references to ~1e−15 relative. |
| EXP-10 | **warning** | Full truncation prices every regime with **zero** non-finite paths where the unguarded naive Euler fails everywhere. Its bias decays at order 1.09 in the Feller-violating regime — but the Feller-satisfying regime never resolved, so no order is fitted there. |
| EXP-11 | pass | Calibration recovers the generating parameters from a **blind** start: normalised distance 2.0e−10, 4 of 4 starts converged, no competing fit at materially different parameters. |
| EXP-12 | **warning** | **A good fit is not an identified parameter set.** On a real SPY surface all seven scenarios converge and fit (implied-vol RMSE 0.0015–0.0106, no penalty reliance), yet mean reversion varies by 145% of its mean and long-run variance by 95%. |
| EXP-13 | pass | Analytic, PDE, and Monte Carlo agree within their combined standard errors on every European cell; Heston Monte Carlo agrees with the characteristic function; the Asian estimators agree pairwise. |
| EXP-14 | **warning** | The nominal 95% interval is not worth 95% everywhere: at K/S=1.6 with payoff skewness 17.3, N=2,000 covers 87.75%, 4.4σ below nominal. More paths fix it. |
| EXP-15 | pass | 21 of 21 edge cases either produce the correct limiting value or refuse explicitly. No non-finite value escapes. |

## The five warnings

A warning here does not mean the experiment malfunctioned. Every one of the five ran
to completion and answered its question; the warning says the answer carries a caveat
that changes what may be quoted from it. **An experiment does not earn a pass for
successfully discovering a limitation** — if it did, the status would describe the
harness rather than the finding.

**EXP-02** records that the full-range slopes exclude their theoretical values while
the asymptotic-window slopes cover them, because the error is `C·dt^k` only to
leading order. The local orders climb monotonically toward theory as the grid
refines, which is what distinguishes this from a wrong order. Both fits are
published; the reasoning is in `docs/CONVERGENCE-METHODOLOGY.md` §3.

**EXP-07** carries two caveats that a bare pass would invite a reader to ignore.
First, it **refuses to fit** the B=70 arm, where the bias never clears its own
across-seed noise. An earlier run fitted it anyway and published an order of −0.19
with a 95% interval of [−0.70, +0.33] — a confident-looking claim that the bias
*grows* as the barrier is watched more often, fitted entirely to six draws from zero.
Second, it **measures where its own reference stops working**. The
Broadie–Glasserman–Kou continuity correction predicts the bias in closed form and
lands within noise for down-barriers, disagreeing at only 2 of 18 resolved cells. For
up-barriers it is resolved as *wrong* at 23 of 24 — not a sign error (reversing the
shift misses by an order of magnitude) but the correction's own o(1/√m) remainder,
made visible because an up-and-out call near the spot pays on almost no paths and so
has almost no noise to hide under. The record says not to use it as an oracle there.
The fitted orders also miss the theoretical 0.5 **from both sides** — down-barriers
low, up-barriers high — which is the argument that the miss is a higher-order
artefact rather than a correction to theory. See
`docs/BARRIER-MONITORING-METHODOLOGY.md` §5–8.

The separate **PDE arm** prices the continuous contract directly with an absorbing
boundary and converges to the analytic reference at order ~2, with five of eight
cells landing on 2.000 with tight intervals; the near-spot barriers are ragged at
coarse grids but clear the second-order floor the arm enforces. It fails the whole
record if any fitted order drops below 1.5, which is what a misplaced barrier
boundary would look like — a standing regression on the engine, not a one-time check.

**EXP-10** establishes that full truncation prices both regimes without a single
non-finite path while the naive scheme fails at every step count tested, and that the
remaining bias decays at order 1.09 where it is measurable. It warns because that
order is measured in *one* of the two regimes. In the Feller-satisfying regime the
bias is genuinely tiny — below the sampling noise at every step — so the record
reports the decay order as unresolved rather than fitting a slope to noise. "Full
truncation converges at first order" is therefore established for the regime that
resolved and not for the other, and requiring only *some* regime to resolve would let
one measured order stand in for both.

**EXP-12** is the result that most deserves not to be rounded off. On a **real** SPY
surface — 18 quotes, three maturities, none excluded — all seven scenarios converge
and every one fits, with implied-vol RMSE from 0.0015 to 0.0106 and no scenario
leaning on a penalty. A reader who stopped at the fit quality would conclude the
model was pinned down. It is not. Across those same scenarios the **mean reversion
spans 0.26 to 11.21** (145% of its own mean) and the **long-run variance 0.038 to
0.496** (95%), while the correlation (−0.72 to −0.55) and initial variance hold to
about 10%. The parameters governing the short end are determined by this surface; the
pair governing the long-run variance level is not — κ and θ trade off along a valley
the data does not resolve. This is why the record reports objective value, surface
fit, parameter dispersion, and penalty reliance as four separate things: collapsing
them into "the calibration converged" would state the one that is true and hide the
one that matters. Contrast EXP-11, where a *synthetic* surface generated by known
Heston parameters is recovered to 2.0e−10 with no competing fit — the difference
between the two records is model error, not solver quality.

**EXP-14** asks whether the reported confidence intervals cover at the rate they
claim, and the answer is: not everywhere. At the large sample the observed coverage
sits within a couple of its own standard errors of nominal at every moneyness,
including the severely right-skewed deep-out-of-the-money call — that is the central
limit theorem working. At N=2,000 with payoff skewness 17.3 the interval covers
87.75%, which is 4.4σ below the nominal 95%. The option pays on a small fraction of
paths, so the payoff distribution is far from normal, the estimated standard error is
itself unreliable, and the interval comes out too narrow. The degradation is expected
and explained, and it is still a warning, because a reader quoting a 95% interval in
that regime would be quoting something this experiment disproved.

## Reading a record

Fields absent from a record are absent on purpose. An analytic level carries no
`error_standard_error` because it was computed rather than sampled, and a `0.0` there
would claim a measured certainty. EXP-04's cells carry no `non_positive_states`
because the pricing engine does not surface the path generator's diagnostics — the
question was not asked, so it is not answered.

`limitations` is populated on every record, including the passing ones.

Five experiments pin no seed — EXP-06, EXP-09, EXP-11, EXP-12, EXP-15. That is not an
omission: none of them samples. The PDE sweeps, the characteristic-function
quadrature, both calibrations (analytic surfaces and a deterministic
Levenberg–Marquardt), and the edge cases are exactly reproducible without one, and
`MANIFEST.json` records `deterministic` as their seed policy. The audit infers which
experiments sample from their stored configuration rather than from a hand-kept list,
so an experiment that samples and loses its seed is caught.

## Provenance

Every record embeds the compiler, flags, git commit, and hardware that produced it.
This is not decoration: the project builds with `-ffp-contract=off` precisely because
floating-point contraction changes answers, so a slope produced under different flags
is a different result.

`MANIFEST.json` names the generator commit and carries a SHA-256 for every record,
CSV, and figure. Its `artifact_commit` field is null by construction — a manifest
cannot name the commit that adds it, and back-filling it would change the file its own
checksums cover. `git log --follow results/MANIFEST.json` resolves it after the fact.

Re-running on a different machine will reproduce the analytic values bit-for-bit and
the simulated ones to within the tiers documented in `docs/FLOATING-POINT-POLICY.md`.

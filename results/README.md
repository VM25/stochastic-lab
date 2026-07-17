# Experiment results

Machine-readable records for the experiments in `docs/EXPERIMENT-CATALOG.MD`. Each
`EXP-NN.json` is a complete record — configuration, results, summary table,
status, interpretation, limitations, reproduction command, and the build
provenance that produced it. `EXP-NN.csv` is the same summary table, for plotting.

Regenerate any of them:

```
./build/release/src/diffusionworks experiment \
    --config configs/experiment/convergence.json --id EXP-02 \
    --output results/EXP-02.json
```

and the figures in `docs/figures/`:

```
python3 python/plot_convergence.py results/*.json --outdir docs/figures
```

## Status

| Experiment | Status | Headline |
| --- | --- | --- |
| EXP-01 | pass | Monte Carlo RMSE decays as N^(−1/2) across all eight scenarios; every fitted interval covers −0.5. |
| EXP-02 | **warning** | Euler–Maruyama attains strong order ½ and Milstein order 1 in the asymptotic window. The full-range fits fall short — the coarse levels are pre-asymptotic. |
| EXP-03 | pass | Both schemes converge weakly at order 1. Euler and Milstein share their first moment *exactly*. |
| EXP-04 | pass | The bias floor exists and is located: on coarse grids more paths stop helping. |
| EXP-06 | pass | Crank–Nicolson reaches order 2.0034 in space; Rannacher restores order 1.98 in time where plain CN oscillates. The explicit stability bound is sufficient, not necessary — stable to ratio 1.60, divergent at 1.70. |
| EXP-07 | pass | Daily monitoring is not continuous monitoring: the bias reaches **67% of price** for an up-and-out call near the spot, and is biased high in all 42 resolved cells. The Brownian bridge removes it at every frequency, in both directions. A separate PDE arm prices the continuous contract directly and converges to the analytic reference at order ~2.0. |
| EXP-08 | pass | No Greek estimator wins everywhere. Under common random numbers the finite-difference **delta** variance is bump-independent (fitted exponent ~0, not 1/h); **gamma** grows only as ~bump^(−0.5), not 1/h². Pathwise beats likelihood-ratio delta by 1.8–2.4× in standard error but has no gamma; likelihood-ratio is the one that survives a discontinuous payoff. |

The EXP-02 warning is not a defect. It records that the full-range slopes (0.4974,
0.9838) exclude their theoretical values while the asymptotic-window slopes
(0.4996, 0.9957) cover them, because the error is `C·dt^k` only to leading order.
The local orders climb monotonically toward theory as the grid refines, which is
what distinguishes this from a wrong order. Both fits are published; the reasoning
is in `docs/CONVERGENCE-METHODOLOGY.md` §3.

EXP-07 passes while reporting fitted orders that *exclude* their theoretical 0.5 —
and they miss it **from both sides**: down-barriers low (0.395 at B=90) and close
up-barriers high (0.630 at B=105). That two-sided miss is the argument. A
single-direction study would have read its own miss as a signed correction to theory
and been wrong in a way no extra care within that direction could expose; what is
really happening is that the discarded higher-order terms carry opposite signs for
the two geometries. The asymptotic windows recover 0.5 at five of seven fitted arms.
See `docs/BARRIER-MONITORING-METHODOLOGY.md` §5–6.

EXP-07 also **measures where its own oracle stops working**. The Broadie–Glasserman–Kou
continuity correction predicts the bias's magnitude in closed form, and for
down-barriers it lands within 2 standard errors at 16 of 18 resolved cells without
having been fitted to the data. For up-barriers it is resolved as *wrong* at 23 of 24
cells — not a sign error (verified independently; reversing the shift misses by an
order of magnitude) but the correction's own o(1/√m) remainder, made visible because
an up-and-out call near the spot pays on almost no paths and so has almost no noise
to hide under. The record says not to use it as an oracle there.

EXP-07 also **refuses to fit** the B=70 arm, where the bias never clears 2.1
across-seed standard errors. An earlier run fitted it anyway and published an order
of −0.19 with a 95% interval of [−0.70, +0.33] — a confident-looking claim that the
bias *grows* as the barrier is watched more often, fitted entirely to six draws from
zero. The record now says the bias was not resolved, which is what happened.

The **PDE arm** is a distinct question from the monitoring arms and is anchored to
the same continuous price. It solves the Black–Scholes equation with an absorbing
boundary at the barrier and converges to the analytic reference at order ~2.0 across
both directions (six of eight cells land on 2.000 with tight intervals; the two near-
spot barriers are ragged at coarse grids but clear the second-order floor the arm
enforces). It fails the whole record if any fitted order drops below 1.5, which is
what a misplaced barrier boundary would look like — so it is a standing regression on
the engine, not a one-time check. See `docs/BARRIER-MONITORING-METHODOLOGY.md` §8.

EXP-08's headline is a correction to a textbook heuristic. The naive picture is that a
finite-difference Greek's variance grows as 1/h (delta) or 1/h² (gamma) as the bump h
shrinks, forcing a bias-variance compromise. **Under common random numbers that is not
what happens.** The experiment fits the dispersion-versus-bump exponent per cell and
finds a median near **0.0** for delta (its variance is essentially bump-independent,
because the shared-draw estimator converges to the pathwise one) and near **0.5** for
gamma — not 2.0, and not a universal law (it ranges 0.4–2.0 with moneyness). The
rankings are equally regime-specific: pathwise delta beats likelihood-ratio delta by
1.8–2.4× in standard error, but has no gamma, and likelihood-ratio is the only method
that survives a discontinuous payoff. The experiment reports these per cell rather than
declaring a winner, and marks its worst region (deep out-of-the-money, short maturity)
explicitly. It fails the record only if a *theoretically unbiased* estimator (pathwise,
likelihood-ratio) shows a resolved bias, which none does.

## Reading a record

Fields absent from a record are absent on purpose. An analytic level carries no
`error_standard_error` because it was computed rather than sampled, and a `0.0`
there would claim a measured certainty. EXP-04's cells carry no
`non_positive_states` because the pricing engine does not surface the path
generator's diagnostics — the question was not asked, so it is not answered.

`limitations` is populated on every record, including the passing ones.

## Provenance

Every record embeds the compiler, flags, git commit, and hardware that produced
it. This is not decoration: the project builds with `-ffp-contract=off` precisely
because floating-point contraction changes answers, so a slope produced under
different flags is a different result.

Records here were produced on the commit named in each file's `build_metadata`.
Re-running on a different machine will reproduce the analytic values bit-for-bit
and the simulated ones to within the tiers documented in
`docs/FLOATING-POINT-POLICY.md`.

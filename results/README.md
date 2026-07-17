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
| EXP-07 | pass | Daily monitoring is not continuous monitoring: the bias reaches 10.1% of price at a barrier 0.26 σ√T away. The Brownian bridge removes it at every frequency tested. |

The EXP-02 warning is not a defect. It records that the full-range slopes (0.4974,
0.9838) exclude their theoretical values while the asymptotic-window slopes
(0.4996, 0.9957) cover them, because the error is `C·dt^k` only to leading order.
The local orders climb monotonically toward theory as the grid refines, which is
what distinguishes this from a wrong order. Both fits are published; the reasoning
is in `docs/CONVERGENCE-METHODOLOGY.md` §3.

EXP-07 passes while reporting fitted orders that *exclude* their theoretical 0.5
(0.395 at B=90, 0.437 at B=95). The same pre-asymptotic reading as EXP-02 applies —
but here it is not left resting on the climbing local orders alone. The
Broadie–Glasserman–Kou continuity correction predicts the bias's *magnitude* in
closed form, and it accounts for 94–101% of the measured bias without having been
fitted to it. A theory with the rate wrong could not predict the size to within a
percent, which is what distinguishes a contaminated fitting range from a wrong
order. See `docs/BARRIER-MONITORING-METHODOLOGY.md` §5–6.

EXP-07 also **refuses to fit** the B=70 arm, where the bias never clears 2.1
across-seed standard errors. An earlier run fitted it anyway and published an order
of −0.19 with a 95% interval of [−0.70, +0.33] — a confident-looking claim that the
bias *grows* as the barrier is watched more often, fitted entirely to six draws from
zero. The record now says the bias was not resolved, which is what happened.

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

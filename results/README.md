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

The EXP-02 warning is not a defect. It records that the full-range slopes (0.4974,
0.9838) exclude their theoretical values while the asymptotic-window slopes
(0.4996, 0.9957) cover them, because the error is `C·dt^k` only to leading order.
The local orders climb monotonically toward theory as the grid refines, which is
what distinguishes this from a wrong order. Both fits are published; the reasoning
is in `docs/CONVERGENCE-METHODOLOGY.md` §3.

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

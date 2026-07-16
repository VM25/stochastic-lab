# Experiment configurations

Every experiment runs from a stored configuration, per EXPERIMENT-CATALOG's
requirement that results be reproducible from a recorded input rather than from a
number typed into a source file.

```
diffusionworks experiment --config configs/experiment/convergence.json --id EXP-02
```

## `convergence.json`

Drives EXP-01 through EXP-04. Fields are optional and fall back to the defaults in
`ConvergenceExperimentConfig`; unknown keys are rejected rather than ignored, so a
misspelled `volatilty` fails loudly instead of silently leaving the default in
place.

| Field | Meaning |
| --- | --- |
| `spot`, `rate`, `dividend_yield`, `volatility`, `maturity`, `strike` | The Black–Scholes market and instrument. |
| `master_seed` | Base seed. Independent replications are derived from it; `--seed` on the command line overrides it. |
| `seed_count` | Independent replications for the stochastic claims (EXP-01). A single seed's error is one draw from a distribution centred near zero, so it cannot establish a rate. |
| `step_counts` | Grid levels for the discretisation studies, coarse to fine. At least three: two points can be joined by a line of any slope. |
| `asymptotic_level_count` | How many of the finest levels form the asymptotic window. |
| `strong_paths` | Paths per grid level in EXP-02. |
| `path_counts` | Path counts swept in EXP-01. |

### Why `asymptotic_level_count` is in the configuration

Because it must be fixed **before** the results are seen.

The theoretical convergence orders describe the limit `dt → 0`. On a finite grid
the error is `C·dt^k` only to leading order, and the neglected terms bend the
log–log plot at the coarse end — so a fit over the whole grid understates the
order. Restricted to the asymptotic window, Euler-Maruyama measures 0.5001 and
Milstein 0.9941; fitted over everything they read 0.4932 and 0.9637.

That gap is real and must be reported, not tuned away. Choosing the window after
seeing which one lands on the theoretical value would be choosing the answer, so
the window lives here, both fits are always published, and the verdict says
explicitly when the two disagree.

The defaults run to `M = 1024` because Milstein needs it. A higher-order scheme
enters its asymptotic regime *later*: at `M = 256` Milstein still measures 0.985.
Truncating the grid to save time produces an honest measurement of a
pre-asymptotic slope and an apparent contradiction with theory.

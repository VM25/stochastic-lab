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

## `heston_simulation.json`

Drives EXP-10, the Heston variance-discretization study. Fields fall back to the
defaults in `HestonSimulationExperimentConfig`; unknown keys are rejected. The
methodology is `docs/HESTON-SIMULATION-METHODOLOGY.md`.

| Field | Meaning |
| --- | --- |
| `spot`, `strike`, `rate`, `dividend_yield`, `maturity` | The market and instrument. |
| `initial_variance`, `mean_reversion`, `long_run_variance`, `correlation` | The Heston parameters shared by every regime. |
| `vol_of_variance` | One regime per entry. Everything else is held fixed, so this is the single knob that moves the Feller ratio across 1; the default `[0.3, 1.0]` straddles it (ratios 1.78 and 0.16). |
| `step_counts` | Time steps per path, coarse to fine. At least three, to see the bias decay. |
| `paths` | Paths per cell. |
| `seed_count` | Independent replications. The bias is measured across seeds, so one seed cannot separate it from a lucky draw. |
| `master_seed` | Base seed; `--seed` overrides it. |
| `bias_resolution` | How many across-seed standard errors a bias must clear before its decay order is fitted rather than treated as noise. |

## `calibration.json`

Drives EXP-11, the Heston calibration-recovery study. The methodology is
`docs/CALIBRATION-METHODOLOGY.md`. Heston parameter vectors are objects with the five
named fields (`initial_variance`, `mean_reversion`, `long_run_variance`,
`vol_of_variance`, `correlation`).

| Field | Meaning |
| --- | --- |
| `spot`, `rate`, `dividend_yield` | The market. |
| `strikes`, `maturities` | The grid the synthetic surface is priced on. |
| `true_parameters` | The parameters the surface is generated from — the answer recovery must reach without being told it. |
| `initial_guesses` | Diverse starting points, at least two, none the truth. The recovery verdict is read only from a start that did not begin near the answer. |
| `objective` | `implied_volatility` (primary) or `price`. |
| `quadrature_nodes`, `max_iterations` | Pricing accuracy per evaluation and the optimizer budget. |
| `recovery_tolerance`, `fit_tolerance_iv_rmse` | The normalised parameter distance and IV RMSE below which recovery and fit count as accurate. |
| `truth_guess_epsilon` | A guess this close to the truth is treated as the truth and excluded from the recovery verdict. |

## `calibration_stability.json`

Drives EXP-12, the market-surface stability study. It calibrates one fixed surface
under five built-in scenarios (uniform / at-the-money / wing weighting, tighter bounds,
and a reduced strike set) and reports the dispersion of the calibrated parameters.

| Field | Meaning |
| --- | --- |
| `spot`, `rate`, `dividend_yield`, `strikes`, `maturities` | The market and grid of the fixed surface. |
| `surface_parameters` | The Heston parameters the documented synthetic surface is generated from. Matches `configs/calibration/market_surface.json`. |
| `as_of` | The timestamp the surface is documented with. |
| `initial_guesses` | The starting points every scenario calibrates from. |
| `objective`, `quadrature_nodes`, `max_iterations` | As above. |

The surface here is a **documented synthetic reference, not real market data**;
`configs/calibration/market_surface.json` is the same surface as an explicit
implied-volatility quote set runnable through the `calibrate` command.

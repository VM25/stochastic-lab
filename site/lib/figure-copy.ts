// Text alternatives and captions for the committed figures. These describe what each
// chart shows — its axes, method, and what to look for — deliberately without
// restating specific result numbers, which belong to the records and are rendered
// from them elsewhere on the page. This keeps every displayed number traceable to an
// artifact while giving each figure the accessible description DESIGN-SPEC requires.

export interface FigureCopy {
  alt: string;
  caption: string;
}

export const FIGURE_COPY: Record<string, FigureCopy> = {
  "sampling-convergence.png": {
    alt: "Log-log plot of Monte Carlo root-mean-square pricing error against path count, showing a straight negative-slope line for each scenario alongside a reference slope of minus one half.",
    caption:
      "Monte Carlo sampling error versus path count, log-log, one line per scenario, against a reference slope of −1/2. Straight parallel lines are the expected N^(−1/2) decay.",
  },
  "strong-convergence.png": {
    alt: "Log-log plot of strong error against time-step size for Euler–Maruyama and Milstein, each with a full-range and asymptotic-window fitted slope, against reference slopes of one half and one.",
    caption:
      "Strong error versus time step on common Brownian paths, one panel per parameter variation. Fitted convergence orders and intervals, with the measured standard errors shown as error bars.",
  },
  "local-orders.png": {
    alt: "Plot of local convergence order between adjacent grid levels, rising monotonically toward the theoretical order as the grid refines.",
    caption:
      "Local order between adjacent step sizes. The monotone climb toward theory is why a full-range slope understates the order — the coarse levels are pre-asymptotic, not wrong.",
  },
  "weak-convergence.png": {
    alt: "Log-log plot of weak error against time-step size for three test functions, both schemes converging at order one.",
    caption:
      "Weak error versus time step, one panel per test function. For f(S)=S and f(S)=S² the error is computed in closed form; the call payoff is sampled.",
  },
  "bias-variance.png": {
    alt: "Plot of RMSE against path count at several step counts, the coarse-grid curves flattening onto a discretisation-bias floor.",
    caption:
      "Sampling error versus discretisation bias. On coarse grids the RMSE flattens onto a floor set by the bias, where more paths stop helping.",
  },
  "work-normalised-efficiency.png": {
    alt: "Plot of work-normalised efficiency gain over crude sampling for each variance-reduction estimator, per instrument, on a log scale.",
    caption:
      "Work-normalised efficiency — reciprocal variance times deterministic work units, not wall-clock — for each estimator, per instrument. The ranking is a property of the estimators.",
  },
  "space-convergence.png": {
    alt: "Log-log plot of PDE pricing error against asset-grid spacing for each time-stepping scheme, against a reference slope of two.",
    caption:
      "Spatial convergence: pricing error versus asset spacing ΔS, log-log, against a second-order reference, with the fitted orders shown.",
  },
  "time-convergence.png": {
    alt: "Log-log plot of PDE pricing error against time step for implicit, plain Crank–Nicolson, and Rannacher-smoothed schemes, with the plain scheme kinking at coarse steps.",
    caption:
      "Temporal convergence: implicit (first order), plain Crank–Nicolson whose full-range fit is inflated by coarse-grid oscillation, and Rannacher smoothing that restores clean second order.",
  },
  "explicit-stability.png": {
    alt: "Semi-log scatter of pricing error against Courant ratio, stable points near zero error and an unstable point exploding past the stability bound.",
    caption:
      "Explicit-scheme stability: error against the Courant ratio, with the coefficient-derived bound marked. The scheme is stable past the bound and divergent beyond — sufficient, not necessary.",
  },
  "monitoring-bias.png": {
    alt: "Two panels, down-and-out and up-and-out, plotting relative bias against monitoring frequency for discrete monitoring (biased) and the Brownian bridge (near zero).",
    caption:
      "Discrete-monitoring bias versus frequency, per barrier direction. Discrete monitoring (solid, filled) is biased; the Brownian bridge (dashed, open) removes it. The bias is two-sided.",
  },
  "pde-convergence.png": {
    alt: "Log-log plot of PDE barrier pricing error against grid refinement for each barrier, converging at order two toward the continuous reference.",
    caption:
      "The absorbing-boundary PDE solution converging to the analytic continuous-barrier price at order ~2, in both directions.",
  },
  "fd-variance-scaling.png": {
    alt: "Two panels for delta and gamma plotting the fitted dispersion-versus-bump exponent per cell, against the textbook exponents of one and two and the measured medians near zero and one half.",
    caption:
      "Under common random numbers the finite-difference variance does not follow the textbook 1/h (delta) or 1/h² (gamma). The fitted exponent per cell sits near the measured medians, not the textbook lines.",
  },
  "estimator-standard-error.png": {
    alt: "Plot comparing the across-seed standard error of the pathwise and likelihood-ratio delta estimators cell by cell, pathwise consistently lower.",
    caption:
      "Standard error of the two unbiased delta estimators, cell by cell. Both are unbiased; pathwise has the smaller standard error everywhere, and likelihood-ratio survives a discontinuous payoff.",
  },
  "cf-validation.png": {
    alt: "Two panels: quadrature convergence to machine precision, and relative error of the characteristic-function price against published and independently generated references, all far below tolerance.",
    caption:
      "Left: the integration converging to machine precision, a self-consistency check. Right: agreement with published and independently computed reference values, kept as separate kinds of evidence.",
  },
  "variance-discretization.png": {
    alt: "Two panels per regime plotting full-truncation bias against step count, with a fitted decay order in the Feller-violating regime and an unresolved marker in the satisfying regime.",
    caption:
      "Full-truncation bias versus step count per regime. The bias decays measurably in the Feller-violating regime; in the satisfying regime it never clears the noise, so no order is fitted.",
  },
  "recovery.png": {
    alt: "Bar chart of normalised distance to the generating parameters for each optimiser start, blind and seeded, all recovering the truth to near machine precision.",
    caption:
      "Parameter recovery from a synthetic surface, per optimiser start. The headline is read from a blind start — one that did not begin at the answer.",
  },
  "parameter-dispersion.png": {
    alt: "Five small panels, one per Heston parameter, plotting the calibrated value across seven scenarios, tight for correlation and initial variance and widely scattered for mean reversion and long-run variance.",
    caption:
      "The same real surface, seven scenarios, all converged and all fitting. The short-end parameters are determined; mean reversion and long-run variance are not.",
  },
  "residual-surface.png": {
    alt: "Plot of the implied-volatility residual, model minus market, by strike and maturity for the base calibration scenario.",
    caption:
      "Implied-volatility residual by strike and maturity. A real surface is not a Heston surface, so the residual is genuine model error, reported rather than summarised away.",
  },
  "agreement.png": {
    alt: "Bar chart of pairwise method disagreement in combined standard errors for each cell, every bar well below the five-sigma agreement gate.",
    caption:
      "Cross-method disagreement in units of combined standard error, per cell, against the 5σ agreement gate. Disagreement is judged against sampling noise, not an absolute tolerance.",
  },
  "coverage.png": {
    alt: "Plot of observed coverage of the nominal ninety-five percent interval per cell, one cell in red below nominal where the sample is small and the payoff is severely skewed.",
    caption:
      "Observed coverage of the nominal 95% interval, per cell. Red marks a resolved under-coverage in the small-sample, high-skew corner; more paths restore it.",
  },
  "edge-cases.png": {
    alt: "Horizontal bar matrix of twenty-one edge cases, each grouped by category and marked as behaving correctly, either producing the limiting value or refusing.",
    caption:
      "Every edge case either produces the correct limiting value or refuses explicitly. Refusing is a correct outcome — a plausible number from a degenerate input would be the failure.",
  },
};

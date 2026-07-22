# Build Plan

## Purpose

Build DiffusionWorks in dependency order. Complete each phase only after its functionality, tests, validation, and reproducibility gates pass.

Do not advance because code compiles or produces plausible output. If a gate fails, continue debugging, redesigning, or reimplementing until it passes.

---

## Phase 0: Project Foundation

### Goal

Create a clean, reproducible C++ project skeleton.

### Build

* CMake project and presets;
* debug, release, and sanitizer builds;
* dependency management;
* testing framework;
* formatting and static-analysis configuration;
* CI for GCC and Clang;
* initial folder structure;
* CLI executable shell;
* versioned JSON configuration schema;
* structured result and error types.

### Exit Gate

* clean checkout configures and builds;
* tests run through CMake;
* CI passes;
* sanitizer preset works;
* malformed configuration fails explicitly;
* no pricing logic exists outside C++.

---

## Phase 1: Domain Model and Analytic Black–Scholes

### Goal

Establish the mathematical and software baseline.

### Build

* market and option domain types;
* validated constructors;
* normal PDF and CDF;
* European call and put payoffs;
* Black–Scholes prices;
* analytic delta, gamma, vega, theta, and rho;
* `price` CLI command;
* JSON and console outputs.

### Validate

* trusted reference prices;
* analytic Greek references;
* put–call parity;
* bounds;
* monotonicity;
* limiting cases.

### Exit Gate

* all analytic tests pass with justified tolerances;
* invalid inputs fail safely;
* results contain full metadata;
* no bare numerical result is returned without status and diagnostics.

---

## Phase 2: Statistics and Randomness

### Goal

Create trustworthy stochastic infrastructure before Monte Carlo pricing.

### Build

* deterministic master seed;
* reproducible stream partitioning;
* standard-normal generation;
* correlated-normal generation;
* online mean, variance, covariance, and confidence intervals;
* multi-seed aggregation;
* thread-local accumulators.

### Validate

* fixed-seed reproducibility;
* normal moments;
* correlation recovery;
* statistical accumulator accuracy;
* independent stream behavior.

### Exit Gate

* repeated fixed configurations reproduce results;
* statistical tests are stable;
* no shared mutable RNG exists;
* no flaky stochastic tests remain.

---

## Phase 3: GBM Simulation and Monte Carlo Pricing

### Goal

Implement correct, reproducible Monte Carlo pricing under GBM.

### Build

* exact GBM transition;
* Euler–Maruyama;
* Milstein;
* path generation;
* European and arithmetic Asian payoffs;
* Monte Carlo estimator;
* standard error and confidence interval;
* `simulate` support;
* single-thread baseline.

### Validate

* theoretical GBM moments;
* European prices against Black–Scholes;
* confidence coverage;
* fixed-seed regression results.

### Exit Gate

* exact GBM reproduces theoretical moments;
* Monte Carlo prices converge toward analytic values;
* uncertainty estimates are statistically defensible;
* non-finite paths fail explicitly.

---

## Phase 4: Variance Reduction

### Goal

Improve estimator efficiency without changing expected value.

### Build

* antithetic variates;
* common random numbers;
* geometric Asian control variate;
* combined variance-reduction modes.

### Validate

* control expectation;
* unbiasedness within uncertainty;
* variance reduction across multiple seeds;
* runtime-adjusted efficiency.

### Exit Gate

* each method has multi-seed evidence;
* methods that do not improve efficiency remain documented;
* no favorable single run supports a claim.

---

## Phase 5: Convergence Framework

### Goal

Turn numerical methods into reproducible research experiments.

### Build

* experiment runner;
* parameter sweeps;
* strong-error calculation;
* weak-error calculation;
* convergence-slope regression;
* uncertainty estimates;
* CSV and JSON outputs;
* automated plotting inputs.

### Run

* Monte Carlo sampling convergence;
* Euler–Maruyama strong convergence;
* Milstein strong convergence;
* weak convergence;
* sampling-error versus discretization-bias experiment.

### Exit Gate

* empirical slopes are estimated, not visually guessed;
* common Brownian paths are used correctly;
* theoretical and observed behavior are reconciled;
* unresolved contradictions block progression.

---

## Phase 6: Finite-Difference Engine

### Goal

Implement and compare Black–Scholes PDE methods.

### Build

* asset and time grids;
* terminal and boundary conditions;
* native tridiagonal solver;
* explicit method;
* implicit method;
* Crank–Nicolson;
* interpolation;
* stability diagnostics.

### Validate

* solver unit tests;
* European prices against Black–Scholes;
* grid convergence;
* (S_{\max}) sensitivity;
* explicit instability detection.

### Exit Gate

* stable schemes converge toward references;
* unstable explicit cases are detected or demonstrated;
* boundary effects are measured;
* stability is not confused with accuracy.

---

## Phase 7: Barrier Options

### Goal

Demonstrate path dependence, discontinuity, and monitoring bias.

### Build

* down-and-out call;
* up-and-out call;
* discrete monitoring;
* Brownian-bridge correction;
* PDE barrier boundaries.

### Validate

* trusted barrier references;
* already-breached behavior;
* near-barrier cases;
* monitoring-frequency bias;
* barrier parity where applicable.

### Exit Gate

* monitoring conventions are explicit;
* Brownian-bridge correction is validated;
* near-barrier failures are documented;
* barrier prices never exceed corresponding vanilla values without explanation.

---

## Phase 8: Greeks Framework

### Goal

Compare Greek estimators by accuracy, variance, stability, and cost.

### Build

* central finite-difference delta, gamma, and vega;
* common-random-number bumping;
* pathwise delta;
* at least one likelihood-ratio estimator;
* structured Greek results.

### Validate

* analytic Black–Scholes Greeks;
* bump-size experiments;
* estimator bias and variance;
* moneyness and maturity sensitivity.

### Exit Gate

* finite-difference bump trade-offs are demonstrated;
* unsupported estimator-payoff combinations are rejected;
* no estimator is declared universally superior;
* results include uncertainty and runtime.

---

## Phase 9: Heston Semi-Analytic Pricing

### Goal

Create an independently validated Heston reference engine.

### Build

* Heston parameters and validation;
* characteristic function;
* numerical quadrature;
* European pricing;
* Feller-condition diagnostics;
* integration failure reporting.

### Validate

* published reference values;
* optional QuantLib comparisons;
* integration-bound and tolerance sensitivity;
* difficult parameter regimes.

### Exit Gate

* reference prices match independently trusted values;
* branch handling is documented;
* failures never return plausible values;
* Feller-condition failure is reported, not automatically rejected.

---

## Phase 10: Heston Simulation

### Goal

Implement and study stochastic-volatility path simulation.

### Build

* correlated Heston shocks;
* full-truncation Euler;
* variance diagnostics;
* European Monte Carlo pricing;
* path-failure detection.

### Validate

* comparison with semi-analytic Heston prices;
* time-step convergence;
* negative pre-truncation frequency;
* difficult parameter regimes.

### Exit Gate

* simulation bias is quantified;
* invalid variance behavior is never hidden;
* prices approach semi-analytic references as resolution improves;
* unresolved non-finite paths block completion.

---

## Phase 11: Implied Volatility and Calibration

### Goal

Implement defensible Heston calibration rather than optimizer demonstration.

### Build

* bracketed implied-volatility solver;
* constrained parameter transformations;
* weighted calibration objective;
* derivative-free optimizer;
* multiple initial guesses;
* residual export;
* explicit convergence status.

### Validate

* known-volatility recovery;
* invalid-price rejection;
* synthetic Heston surface recovery;
* parameter and surface-fit stability;
* static documented market surface.

### Exit Gate

* synthetic pricing surface is recovered accurately;
* parameter non-uniqueness is reported;
* multiple initializations are tested;
* calibration failure cannot appear successful;
* results do not rely on the true parameters as the only initial guess.

---

## Phase 12: Multithreading

### Goal

Improve Monte Carlo throughput without damaging correctness or reproducibility.

### Build

* deterministic path partitioning;
* `std::jthread` workers;
* thread-local buffers and accumulators;
* deterministic reduction;
* thread-count configuration.

### Validate

* one-thread equivalence;
* reproducibility at fixed thread count;
* race and deadlock testing;
* ThreadSanitizer where supported;
* valid behavior across thread counts.

### Exit Gate

* no shared mutable RNG exists;
* no locks or allocations remain in hot path loops without justification;
* race detection is clean;
* numerical differences from reduction order are documented.

---

## Phase 13: Full Experiment Program

### Goal

Execute every required experiment and reconcile the results.

### Run

All experiments in `experiment-catalog.md`.

### Produce

* raw results;
* summary tables;
* charts;
* pass, fail, warning, or inconclusive status;
* interpretation;
* limitations;
* reproduction commands.

### Exit Gate

* all mandatory experiments execute from stored configurations;
* all headline claims trace to generated evidence;
* failed and disappointing outcomes remain visible;
* theoretical and empirical results are reconciled;
* unresolved required failures block progression.

---

## Phase 14: Documentation and Technical Paper

### Goal

Make the mathematics, implementation, results, and limitations independently auditable.

### Produce

* completed API documentation;
* architecture diagrams;
* mathematical-method notes;
* experiment interpretations;
* validation report;
* technical paper;
* exact reproduction instructions.

### Exit Gate

* equations match implemented code;
* reported results match generated artifacts;
* no planned feature is described as implemented;
* every major claim includes evidence;
* limitations and failures are explicit.

Do not write `README.md` yet.

---

## Phase 15: Public Results Site

### Goal

Present the completed technical work without turning it into another dashboard-first project.

### Build

* project overview;
* methods;
* convergence results;
* validation results;
* architecture;
* technical paper access;
* repository links.

### Exit Gate

* the site contains no independent pricing logic;
* charts come from stored engine outputs;
* mobile and desktop layouts are usable;
* presentation does not conceal failures;
* the mathematical and C++ work remains the dominant signal.

---

## Phase 16: Final Acceptance

### Goal

Verify that DiffusionWorks satisfies every completion requirement.

### Run

* clean checkout;
* all build presets;
* complete test suite;
* sanitizers;
* static analysis;
* validation suite;
* experiment regeneration;
* documentation consistency audit;
* website evidence audit.

### Exit Gate

The project is complete only when:

* every required model, method, instrument, and experiment exists;
* all completion-blocking tests and validations pass;
* results are reproducible;
* no core work depends on Python or QuantLib;
* unresolved numerical failures are absent;
* documentation matches reality;
* the final product clearly proves stochastic mathematics and C++ engineering.

Only after this gate passes may you write the final `README.md`.

---

## Scope-Control Rule

Do not add:

* new models;
* American options;
* SABR;
* jump diffusion;
* rough volatility;
* rates or credit models;
* GPU support;
* custom allocators;
* WebAssembly

until every mandatory Version 1.0 gate passes.

A new feature cannot compensate for unfinished correctness, validation, or documentation.

---

## Iteration Rule

At every phase:

1. implement;
2. test;
3. validate;
4. inspect failures;
5. revise;
6. rerun;
7. document evidence;
8. pass the exit gate.

Continue iterating until the gate passes or a better validated result is achieved. Never mark a phase complete through explanation alone.

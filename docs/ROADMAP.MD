# Roadmap

## Purpose

Define the release path for DiffusionWorks and prevent scope expansion from replacing unfinished core work.

Version 1.0 is complete only when all mandatory milestones pass their acceptance gates.

---

## Version 1.0: Core Numerical System

### Milestone 1 — Foundation

Deliver:

* CMake project;
* CLI shell;
* configuration and output schemas;
* CI, sanitizers, formatting, and static analysis;
* validated domain types.

**Gate:** Clean builds and tests pass on supported compilers.

### Milestone 2 — Black–Scholes Baseline

Deliver:

* European options;
* analytic prices and Greeks;
* invariants and limiting-case tests.

**Gate:** Native results match trusted references within justified tolerances.

### Milestone 3 — Monte Carlo Infrastructure

Deliver:

* deterministic randomness;
* exact GBM;
* Euler–Maruyama;
* Milstein;
* statistics and confidence intervals;
* Asian pricing;
* variance reduction.

**Gate:** Pricing, uncertainty, and convergence are statistically validated.

### Milestone 4 — Finite-Difference Methods

Deliver:

* explicit;
* implicit;
* Crank–Nicolson;
* native tridiagonal solver;
* vanilla and barrier pricing.

**Gate:** Stable schemes converge correctly and unstable configurations are detected.

### Milestone 5 — Greek Estimation

Deliver:

* finite-difference;
* pathwise;
* likelihood-ratio estimators;
* bias, variance, bump-size, and runtime comparisons.

**Gate:** Every estimator is validated against analytic references.

### Milestone 6 — Heston Model

Deliver:

* characteristic-function pricing;
* full-truncation simulation;
* variance diagnostics;
* implied-volatility solver;
* synthetic and market-surface calibration.

**Gate:** Prices match independent references and calibration passes synthetic recovery.

### Milestone 7 — Multithreading

Deliver:

* deterministic multithreading;
* deterministic reductions;
* thread-count controls;
* race-safety tests.

**Gate:** One-thread equivalence, fixed-count reproducibility, and clean race detection hold; no throughput, scaling, or speedup claim is made.

### Milestone 8 — Research Evidence

Deliver:

* all mandatory experiments;
* raw results;
* charts;
* validation report;
* technical paper.

**Gate:** Every headline conclusion traces to generated evidence.

### Milestone 9 — Public Presentation

Deliver:

* results site;
* architecture presentation;
* technical paper access;
* final repository documentation;
* final `README.md`.

**Gate:** Presentation matches the completed engine and contains no unsupported claims.

---

## Version 1.0 Required Scope

Required:

* Black–Scholes and Heston;
* European, Asian, and barrier options;
* analytic, Monte Carlo, PDE, and characteristic-function pricing;
* required Greeks;
* variance reduction;
* Heston calibration;
* validation and experiments;
* CLI, tests, documentation, paper, and results site.

Nothing outside this scope may delay completion.

---

## Deferred Features

Eligible only after Version 1.0 passes final acceptance:

* Sobol quasi-Monte Carlo;
* Heston Quadratic Exponential simulation;
* knock-in barrier variants;
* automatic differentiation;
* SIMD;
* WebAssembly demonstration;
* additional quadrature or optimization methods.

These are possible Version 1.1 features, not hidden Version 1.0 requirements.

---

## Future Research Directions

Consider only when they create a distinct validated research question:

* American exercise;
* local volatility;
* SABR;
* jump diffusion;
* rough volatility;
* interest-rate models;
* adjoint differentiation;
* GPU acceleration.

Do not add a model merely to increase feature count.

---

## Scope Admission Rule

A new feature may enter the roadmap only when:

1. all current milestone gates pass;
2. it answers a clear research question;
3. its validation reference is available;
4. required tests and experiments are defined;
5. it does not weaken reproducibility or architecture;
6. it improves the project’s quantitative or C++ signal.

Otherwise, reject or defer it.

---

## Completion Rule

Version 1.0 is not complete if:

* required work is replaced by stretch features;
* validation or experiments remain unresolved;
* documentation describes unfinished behavior;
* the website advances ahead of the engine;
* published results cannot be reproduced;
* scope continues expanding without a stable final release.

Finish the required system before extending it.

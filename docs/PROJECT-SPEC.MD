# Project Specification

## Project

**DiffusionWorks: Stochastic Derivatives Numerics**

**Subtitle:** A C++ Stochastic Derivatives Modeling and Validation Engine

## Objective

Build a reproducible C++ system that implements stochastic derivatives models and numerical methods from first principles, then evaluates pricing accuracy, convergence, numerical stability, and model calibration.

The project must demonstrate mathematical depth and professional C++ engineering through validated code and measurable results.

## Version 1.0 Scope

### Models

#### Black–Scholes–Merton

[
dS_t=(r-q)S_tdt+\sigma S_tdW_t
]

Required uses:

* analytic pricing;
* analytic Greeks;
* exact simulation;
* Euler–Maruyama;
* Milstein;
* Monte Carlo validation;
* PDE validation.

#### Heston

[
dS_t=(r-q)S_tdt+\sqrt{v_t}S_tdW_t^S
]

[
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v
]

[
dW_t^SdW_t^v=\rho dt
]

Required uses:

* characteristic-function pricing;
* full-truncation Monte Carlo;
* correlated simulation;
* variance diagnostics;
* implied-volatility calibration;
* parameter-stability analysis.

No additional models may enter Version 1.0.

## Instruments

Required:

* European call;
* European put;
* arithmetic Asian call and put;
* geometric Asian control instrument;
* down-and-out call;
* up-and-out call.

Optional only after all required acceptance criteria pass:

* knock-in barrier options;
* additional barrier directions.

## Pricing Methods

### Analytic and Semi-Analytic

Implement natively:

* Black–Scholes European prices;
* Black–Scholes Greeks;
* geometric Asian pricing;
* Heston characteristic-function pricing.

### Monte Carlo

Implement:

* exact GBM simulation;
* Euler–Maruyama;
* Milstein;
* Heston full-truncation Euler;
* confidence intervals;
* deterministic seed control;
* multithreaded execution;
* antithetic variates;
* control variates;
* common random numbers.

### Finite Difference

Implement:

* explicit scheme;
* implicit scheme;
* Crank–Nicolson;
* tridiagonal solver;
* configurable asset and time grids;
* vanilla and barrier boundary conditions;
* stability and grid diagnostics.

## Greeks

Required:

* analytic delta, gamma, vega, theta, and rho under Black–Scholes;
* central finite-difference delta and gamma;
* finite-difference vega;
* pathwise delta;
* at least one likelihood-ratio estimator.

Compare methods by:

* bias;
* variance;
* numerical stability;
* runtime;
* sensitivity to bump size.

## Calibration

Implement Heston calibration against:

* a synthetic surface generated from known parameters;
* one static, documented market surface.

Required:

* constrained parameters;
* multiple initial guesses;
* weighted objective function;
* implied-volatility inversion;
* residual reporting;
* convergence status;
* synthetic parameter recovery;
* parameter-stability analysis.

Calibration success cannot be judged only by a reduced objective value.

## Required Experiments

1. Monte Carlo sampling convergence.
2. Euler–Maruyama versus Milstein strong convergence.
3. Euler–Maruyama versus Milstein weak convergence.
4. Sampling error versus discretization error.
5. Variance-reduction efficiency.
6. PDE stability and grid convergence.
7. Barrier monitoring bias and correction.
8. Greek estimator comparison.
9. Heston variance discretization behavior.
10. Heston calibration recovery and stability.
11. Cross-method accuracy and agreement.

Each experiment must define:

* hypothesis;
* controlled variables;
* reference result;
* configuration;
* output metrics;
* acceptance criteria;
* failure conditions.

## Core Architecture

Required modules:

* market data;
* instruments and payoffs;
* stochastic models;
* pricing engines;
* path simulation;
* random-number generation;
* finite-difference numerics;
* Greeks;
* calibration;
* validation;
* statistics;
* experiments;
* CLI.

Core pricing and simulation logic must remain independent from presentation code.

## Interfaces

The primary interface is a configuration-driven CLI.

Required commands:

```bash
diffusionworks price
diffusionworks simulate
diffusionworks validate
diffusionworks experiment
diffusionworks calibrate
```

Every command must support reproducible configuration files and structured JSON or CSV outputs.

## Required Output Metadata

Published runs must record:

* model and instrument parameters;
* numerical-method settings;
* random seed;
* path or grid count;
* thread count;
* compiler and version;
* build type and flags;
* operating system;
* CPU;
* Git commit;
* runtime;
* warnings and failures.

## Language and Dependency Rules

### C++

C++ owns:

* models;
* pricing;
* simulation;
* Greeks;
* calibration;
* validation;
* experiment execution.

### Python

Python may be used only for:

* plotting;
* result aggregation;
* report generation;
* exploratory cross-checks.

Python must not contain the production pricing engine.

### QuantLib

QuantLib may be used only as an external reference.

It must not perform any central implementation.

### Permitted Supporting Libraries

* CMake;
* Eigen;
* GoogleTest or Catch2;
* `fmt`;
* `nlohmann/json`;
* Doxygen.

Dependencies must not replace the core mathematical work.

## Engineering Requirements

The project must include:

* enforced modern C++ standard;
* CMake build system;
* deterministic randomness;
* clear ownership and RAII;
* contiguous data in hot loops;
* no avoidable allocation inside path loops;
* thread-local accumulation;
* explicit error handling;
* unit and regression tests;
* sanitizer builds;
* static analysis;
* continuous integration.

Do not add custom allocators, SIMD, GPU support, or lock-free structures.

## Validation Requirements

Required validation includes:

* analytic Black–Scholes comparisons;
* put-call parity;
* option bounds;
* monotonicity checks;
* barrier parity where applicable;
* Monte Carlo confidence coverage;
* PDE grid convergence;
* Heston reference comparisons;
* synthetic calibration recovery;
* cross-checks against QuantLib;
* deterministic regression outputs.

Numerical tolerances must be method-specific and justified.

## Public Deliverables

Version 1.0 must include:

* complete C++ source;
* tests and CI;
* experiment configurations;
* validation artifacts;
* generated charts and tables;
* API documentation;
* technical paper;
* public results site;
* tagged stable version.

`README.md` must be written only after the implementation and results are complete.

## Non-Goals

Version 1.0 excludes:

* live market data;
* trading execution;
* portfolio management;
* American options;
* local volatility;
* SABR;
* jump diffusion;
* rough volatility;
* interest-rate models;
* credit derivatives;
* GPU acceleration;
* production deployment claims;
* profit or alpha claims.

## Phase Gate Rule

Do not progress because code compiles or produces plausible output.

A phase is complete only when:

* required functionality works;
* tests pass;
* results are validated;
* failures are documented;
* outputs are reproducible;
* acceptance criteria are met.

If a gate fails, continue debugging, redesigning, or reimplementing until it passes.

## Definition of Project Success

DiffusionWorks is complete only when a technical reviewer can verify that:

1. the models and methods were implemented natively in C++;
2. numerical outputs converge toward defensible references;
3. uncertainty, bias, and instability are measured;
4. Heston calibration is validated rather than merely demonstrated;
5. published results can be regenerated from stored configurations;
6. failure cases and limitations remain visible;
7. the project proves both stochastic-finance knowledge and serious C++ engineering.

## Completion Blockers

The project is not done if:

* any required model, method, instrument, or experiment is missing;
* central calculations rely on QuantLib or Python;
* convergence is assumed rather than measured;
* calibration lacks synthetic recovery testing;
* stochastic conclusions rely on one seed;
* unstable or incorrect cases are hidden;
* tests, sanitizers, or CI fail;
* the website advances ahead of the engine;
* documentation describes features not present in code;
* new scope is added while required work remains incomplete.

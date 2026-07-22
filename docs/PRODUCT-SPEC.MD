# Product Specification

## Product

**DiffusionWorks: Stochastic Derivatives Numerics**

**Subtitle:** A C++ Stochastic Derivatives Modeling and Validation Engine

## Purpose

Build a rigorous C++ system that implements stochastic derivatives models and numerical pricing methods from first principles, then tests their accuracy, convergence, stability, reproducibility, and computational cost.

The product must prove mathematical depth and C++ engineering ability through working code, validated experiments, and measured results.

## Primary Users

* Quantitative developers
* Quantitative researchers
* Derivatives quants
* Model-validation quants
* Risk quants
* Technical recruiters and interviewers

## Core User Questions

DiffusionWorks must answer:

1. What price or Greek does each method produce?
2. How close is it to a trusted reference?
3. Does error decay at the expected rate?
4. Where do bias, variance, instability, or model risk enter?
5. Which method achieves the best accuracy for its computational cost?
6. Can every published result be reproduced?

## Required Capabilities

The final product must provide:

* Black–Scholes and Heston models;
* European, Asian, and barrier options;
* analytic, Monte Carlo, finite-difference, and Heston integration methods;
* Euler–Maruyama, Milstein, and full-truncation simulation;
* analytic, finite-difference, pathwise, and likelihood-ratio Greeks;
* variance reduction;
* Heston calibration;
* convergence experiments;
* validation suites;
* CLI-driven execution;
* machine-readable experiment outputs;
* technical documentation and a public results presentation.

## Primary Workflows

### Price

Select an instrument, model, and method; produce price, uncertainty, runtime, and diagnostics.

### Validate

Compare outputs against analytic values, trusted references, financial invariants, and statistical expectations.

### Experiment

Run controlled convergence, stability, calibration, and estimator-comparison studies.

### Reproduce

Rerun any published result using stored configuration, seed, environment, and build metadata.

## Product Outputs

Every run must produce only applicable fields:

* price or Greek estimate;
* confidence interval;
* absolute and relative error;
* convergence statistics;
* runtime;
* model and method parameters;
* random seed;
* compiler and build metadata;
* thread count;
* warnings and failure diagnostics;
* JSON or CSV result artifact.

## Product Principles

1. Correctness before convenience.
2. Validation before presentation.
3. Measured evidence before claims.
4. Reproducibility before publication.
5. Depth before model count.
6. C++ owns all core numerical work.
7. External libraries cannot replace core implementations.
8. Failed or unstable cases must be documented.
9. The interface must expose uncertainty and limitations.
10. No phase is complete until its acceptance criteria pass.

## Non-Goals

DiffusionWorks is not:

* a trading platform;
* a portfolio analytics dashboard;
* a live market-data system;
* a production pricing library;
* a low-latency execution system;
* a QuantLib wrapper;
* a comprehensive derivatives library;
* an alpha-generation project;
* a GPU-computing project;
* a substitute for mathematical documentation.

## User Experience

The primary interface is a command-line application with configuration-driven runs.

The interface must be:

* deterministic when given a fixed seed;
* explicit about model assumptions;
* strict about invalid inputs;
* readable for humans;
* exportable for analysis;
* consistent across pricing, validation, and experiment commands.

The public website is secondary. It may present results but must not contain separate pricing logic.

## Success Criteria

The product succeeds only when:

* required models and methods are implemented natively in C++;
* numerical results match defensible references within justified tolerances;
* theoretical convergence behavior is empirically tested;
* stochastic outputs include valid uncertainty estimates;
* calibration is tested using synthetic parameter recovery;
* all major claims are traceable to generated evidence;
* a clean environment can build, test, and run the project;
* technical reviewers can inspect assumptions, code, and failure cases;
* the project visibly demonstrates both quantitative mathematics and C++ engineering.

## Completion Blockers

The product is not done if:

* any headline result lacks validation;
* external libraries perform the central calculations;
* stochastic results depend on one favorable seed;
* unstable cases are hidden;
* the website is more complete than the engine;
* core tests, sanitizers, or reproducibility checks fail;
* scope expansion leaves required work unfinished;
* results cannot be regenerated from stored configurations;
* the final system communicates breadth without proving correctness.

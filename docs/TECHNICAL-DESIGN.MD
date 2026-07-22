# Technical Design

## Purpose

Define the C++ architecture, interfaces, data flow, ownership, concurrency, and configuration for DiffusionWorks.

The design must remain simple enough to audit. Abstraction is justified only when it improves correctness, reuse, or testability.

---

## 1. System Architecture

```text
Configuration
    ↓
CLI Command
    ↓
Instrument + Market + Model
    ↓
Pricing / Simulation / Calibration Engine
    ↓
Statistics + Validation + Diagnostics
    ↓
JSON / CSV / Console Output
```

Core numerical code must remain independent from:

* CLI parsing;
* plotting;
* website code;
* report generation;
* QuantLib validation adapters.

---

## 2. Repository Structure

```text
include/diffusionworks/
├── core/
├── market/
├── instruments/
├── models/
├── numerics/
├── simulation/
├── engines/
├── greeks/
├── calibration/
├── statistics/
├── validation/
└── experiments/

src/
├── corresponding implementations
└── cli/

tests/
experiments/
configs/
data/
python/
results/
```

Public interfaces belong in `include/`. Internal implementation details remain in `src/`.

---

## 3. Core Domain Types

Use explicit domain types rather than loosely related primitive parameters.

Required concepts:

```cpp
struct MarketState;
struct OptionContract;
struct ModelParameters;
struct PricingRequest;
struct PricingResult;
struct SimulationConfig;
struct CalibrationResult;
```

Prefer:

* immutable configuration objects;
* validated construction;
* type-safe enums;
* explicit units and conventions;
* value semantics where practical.

Avoid parameter lists containing many unlabelled `double` values.

---

## 4. Instruments

Required instrument representations:

* `EuropeanOption`;
* `AsianOption`;
* `BarrierOption`.

Each instrument must contain contract terms only:

* option type;
* strike;
* maturity;
* monitoring schedule;
* barrier;
* barrier direction;
* knock type.

Pricing logic must not live inside instrument classes.

---

## 5. Models

Required models:

* `BlackScholesModel`;
* `HestonModel`.

Each model must expose:

* validated parameters;
* drift and diffusion behavior where applicable;
* exact transition support where available;
* compatibility with supported pricing engines.

Do not create a generic model hierarchy so broad that unsupported models appear implementable.

---

## 6. Pricing Engines

Required engines:

* analytic Black–Scholes;
* analytic geometric Asian;
* Heston characteristic-function;
* Monte Carlo;
* explicit finite difference;
* implicit finite difference;
* Crank–Nicolson.

A pricing engine receives:

* instrument;
* market state;
* model;
* numerical configuration.

It returns a structured result containing:

* value;
* uncertainty where applicable;
* diagnostics;
* runtime;
* warnings;
* metadata.

Do not return a bare `double`.

---

## 7. Interface Strategy

Use runtime polymorphism only when engine or model selection must occur at runtime.

Use templates for:

* payoff policies;
* discretization schemes;
* accumulators;
* compile-time simulation composition;
* hot-loop behavior where virtual dispatch would be measurable.

Do not template the entire codebase.

Prefer concrete, readable interfaces over speculative generality.

---

## 8. Monte Carlo Architecture

```text
SimulationConfig
    ↓
Random Stream
    ↓
Discretization Scheme
    ↓
Path Generator
    ↓
Payoff Evaluation
    ↓
Variance Reduction
    ↓
Thread-Local Statistics
    ↓
Deterministic Reduction
```

Required separation:

* random-number generation;
* Brownian shock generation;
* path evolution;
* payoff calculation;
* estimator accumulation;
* confidence-interval calculation.

This separation must allow methods to be tested independently.

---

## 9. Randomness

Use a deterministic, explicitly seeded random-number system.

Required properties:

* user-provided master seed;
* reproducible thread-specific streams;
* no shared mutable RNG across threads;
* stable stream partitioning;
* stored seed metadata;
* common random numbers for controlled comparisons.

Do not seed each thread from wall-clock time.

Published stochastic results must be reproducible.

---

## 10. Concurrency

Use CPU multithreading for independent Monte Carlo paths.

Preferred design:

* fixed worker count;
* deterministic path partitioning;
* thread-local accumulators;
* no locks inside path loops;
* final deterministic reduction;
* `std::jthread` or equivalent RAII-managed threads.

Concurrency must not change numerical results beyond documented floating-point reduction effects.

Do not introduce asynchronous architecture, task graphs, or lock-free containers without demonstrated need.

---

## 11. Memory and Data Layout

Hot numerical loops must:

* preallocate buffers;
* avoid per-path allocation;
* use contiguous storage;
* reuse scratch memory;
* avoid unnecessary copying;
* minimize pointer chasing;
* separate orchestration from numeric kernels.

Start with standard containers. Do not introduce specialized layouts speculatively.

---

## 12. Finite-Difference Architecture

Separate:

* grid construction;
* coefficient generation;
* boundary conditions;
* terminal conditions;
* time stepping;
* tridiagonal solving;
* interpolation;
* diagnostics.

The tridiagonal solver must be independently tested.

Each scheme must expose:

* grid dimensions;
* stability diagnostics;
* boundary assumptions;
* numerical tolerances;
* convergence metadata.

---

## 13. Heston Pricing and Calibration

Separate:

* characteristic function;
* numerical integration;
* implied-volatility solver;
* objective function;
* parameter transformation;
* optimizer;
* calibration diagnostics.

Calibration must support:

* multiple initial guesses;
* bounded parameters;
* reproducible configurations;
* residual export;
* explicit failure status.

An optimizer returning a parameter vector does not imply calibration success.

---

## 14. Greeks

Greek implementations must share a consistent interface.

Required methods:

* analytic;
* bump-and-revalue;
* pathwise;
* likelihood ratio.

Each result must include:

* estimate;
* standard error where applicable;
* bump size where applicable;
* runtime;
* method;
* warnings.

Use common random numbers for Monte Carlo bump comparisons.

---

## 15. Statistics

Required statistical components:

* online mean;
* sample variance;
* covariance;
* standard error;
* confidence interval;
* RMSE;
* convergence-slope regression;
* multi-seed aggregation.

Use numerically stable online algorithms where appropriate.

Statistics must be independently testable.

---

## 16. Validation Layer

Validation must remain separate from production calculations.

Required validation adapters:

* analytic reference;
* financial invariants;
* stored reference datasets;
* optional QuantLib comparison;
* statistical coverage tests;
* calibration recovery tests.

Validation output must clearly distinguish:

* pass;
* fail;
* warning;
* inconclusive.

Never silently widen tolerances until tests pass.

---

## 17. Configuration

Use versioned JSON configuration files.

A configuration must define:

* instrument;
* market;
* model;
* method;
* numerical settings;
* seed;
* thread count;
* output path.

Reject:

* missing required values;
* invalid ranges;
* unsupported combinations;
* ambiguous conventions.

Default values must be documented and conservative.

---

## 18. CLI

Required command structure:

```bash
diffusionworks price
diffusionworks simulate
diffusionworks validate
diffusionworks experiment
diffusionworks calibrate
```

Each command must support:

```bash
--config
--output
--seed
--threads
--format
--help
```

CLI errors must be explicit and return nonzero exit codes.

---

## 19. Output Schema

Structured outputs must include:

```text
status
command
instrument
market
model
method
result
uncertainty
error_metrics
diagnostics
warnings
runtime
seed
threads
build_metadata
git_commit
```

Do not emit misleading fields when a metric is not applicable.

Human-readable output and machine-readable output must report consistent values.

---

## 20. Error Handling

Use exceptions for unrecoverable boundary-level failures and explicit result states for expected numerical failures.

Examples of expected failures:

* invalid implied-volatility root;
* optimizer non-convergence;
* unstable PDE configuration;
* non-finite path state;
* invalid parameter set;
* failed numerical integration.

Never convert a failure into zero, NaN, or a plausible-looking value without status and diagnostics.

---

## 21. Dependencies

Permitted:

* CMake;
* Eigen;
* GoogleTest or Catch2;
* `fmt`;
* `nlohmann/json`;
* Doxygen;
* QuantLib as optional validation-only dependency.

Dependencies must not implement the project’s central models, pricing engines, simulation methods, Greeks, or calibration logic.

---

## 22. Build System

Use CMake with reproducible presets.

Required presets:

* debug;
* release;
* sanitizer.

Required compiler support:

* GCC;
* Clang;
* optional MSVC.

Enable strict warnings. Treat project warnings as errors in CI where practical.

---

## 23. Testing and CI

CI must run:

* configure and build;
* unit tests;
* regression tests;
* integration tests;
* sanitizer tests;
* formatting checks;
* static analysis;
* clean configuration validation.

---

## 24. Python Boundary

Python may consume generated outputs for:

* plotting;
* tables;
* report generation;
* exploratory validation.

Python must not become a second implementation of the engine.

Any Python cross-check must be clearly labeled as external analysis.

---

## 25. Website Boundary

The public website may display precomputed results or call a deliberately limited WebAssembly demonstration.

It must not:

* contain separate pricing formulas;
* become the authoritative engine;
* conceal failed experiments;
* make claims absent from generated artifacts.

The C++ engine remains the source of truth.

---

## 26. Completion Criteria

The technical design is successfully implemented only when:

* modules have clear responsibilities;
* numerical kernels are independently testable;
* stochastic runs are reproducible;
* thread execution is deterministic by configuration;
* all failures produce explicit diagnostics;
* core calculations are native C++;
* clean builds work from documented presets;
* tests and sanitizers pass;
* generated outputs contain complete metadata;
* architecture supports every required Version 1.0 experiment.

---

## 27. Completion Blockers

The project is not done if:

* core logic is duplicated across CLI, Python, or website code;
* pricing engines return values without diagnostics;
* random streams are non-reproducible;
* multithreading introduces hidden shared state;
* allocations occur unnecessarily inside hot loops;
* abstractions obscure mathematical behavior;
* unsupported models shape the architecture;
* numerical failures are silently converted into outputs;
* QuantLib performs central calculations;
* configuration conventions are ambiguous;
* equations and implementation diverge;
* the architecture cannot reproduce published results.

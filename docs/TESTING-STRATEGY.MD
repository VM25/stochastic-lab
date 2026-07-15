# Testing Strategy

## Purpose

Ensure every DiffusionWorks component is correct, reproducible, failure-aware, and safe to modify.

Tests must verify behavior, not merely increase coverage.

---

## 1. Test Layers

Required:

* unit tests;
* integration tests;
* regression tests;
* property tests;
* statistical tests;
* CLI tests;
* sanitizer checks;
* static analysis;
* clean-build CI.

Each defect must receive a permanent regression test.

---

## 2. Unit Tests

Test numerical components independently.

Required coverage:

* payoff functions;
* option validation;
* Black–Scholes prices and Greeks;
* normal distribution functions;
* exact GBM transition;
* Euler–Maruyama;
* Milstein;
* correlated shocks;
* Heston variance update;
* online statistics;
* confidence intervals;
* tridiagonal solver;
* PDE coefficients and boundaries;
* implied-volatility solver;
* characteristic function;
* quadrature;
* calibration objective;
* JSON parsing;
* result serialization.

---

## 3. Integration Tests

Verify complete workflows:

* configuration to price;
* configuration to simulation;
* configuration to validation;
* configuration to experiment;
* configuration to calibration;
* configuration to benchmark output.

Check:

* correct engine selection;
* metadata completeness;
* reproducibility;
* output-schema consistency;
* explicit failure propagation.

---

## 4. Regression Tests

Store trusted outputs for:

* Black–Scholes prices and Greeks;
* GBM moments;
* PDE prices;
* Heston prices;
* implied volatilities;
* selected calibration results;
* fixed-seed Monte Carlo runs.

Regression updates require documented justification. Never overwrite references only because a test fails.

---

## 5. Property Tests

Verify general invariants across generated valid inputs:

* prices remain finite and non-negative;
* call price increases with spot;
* call price decreases with strike;
* option price increases with volatility;
* knock-out value does not exceed vanilla value;
* put–call parity holds;
* fixed seeds reproduce identical streams;
* antithetic shocks are paired correctly;
* invalid inputs are rejected.

---

## 6. Statistical Tests

Stochastic tests must use statistical acceptance rules rather than exact equality.

Required checks:

* simulated GBM moments;
* Monte Carlo confidence coverage;
* sampling convergence;
* variance-reduction effectiveness;
* correlated-normal moments;
* multi-seed estimator behavior.

Tests must use enough samples to avoid unstable pass/fail behavior.

Flaky tests are failures in test design.

---

## 7. Numerical Tolerances

Tolerance must depend on the method.

Use:

* strict scale-aware tolerances for analytic code;
* grid-dependent tolerances for PDEs;
* confidence-based tolerances for Monte Carlo;
* integration-tolerance-aware checks for Heston pricing;
* price, volatility, and parameter criteria for calibration.

Do not use one global epsilon.

Do not widen tolerances without identifying the numerical cause.

---

## 8. Failure-Path Tests

Required failure cases:

* invalid strikes, maturities, barriers, and model parameters;
* unsupported model-instrument-method combinations;
* unstable explicit PDE configuration;
* implied-volatility root failure;
* numerical-integration failure;
* optimizer non-convergence;
* NaN or infinity inputs;
* non-finite path state;
* malformed configuration;
* unwritable output destination.

Failures must return explicit status, diagnostics, and nonzero CLI exit codes where applicable.

---

## 9. Concurrency Tests

Verify:

* no shared mutable RNG;
* deterministic path partitioning;
* thread-local accumulation;
* reproducible fixed-thread results;
* valid results across thread counts;
* no races or deadlocks;
* correct behavior with one thread and excessive requested threads.

Run race-detection tooling where supported.

---

## 10. CLI Tests

Test every command:

```text
price
simulate
validate
experiment
calibrate
benchmark
```

Verify:

* valid configuration succeeds;
* malformed configuration fails;
* help text works;
* overrides are applied;
* output files are created correctly;
* console and structured outputs agree;
* exit codes are meaningful.

---

## 11. Sanitizers and Analysis

Required development checks:

* AddressSanitizer;
* UndefinedBehaviorSanitizer;
* LeakSanitizer where supported;
* ThreadSanitizer for concurrency paths;
* compiler warnings;
* static analysis;
* formatting checks.

No unresolved sanitizer or high-severity static-analysis issue is permitted.

---

## 12. Continuous Integration

CI must run on clean environments using GCC and Clang.

Required jobs:

1. debug build;
2. release build;
3. unit and integration tests;
4. regression and property tests;
5. sanitizer build;
6. formatting check;
7. static analysis;
8. example configuration execution.

Benchmarks must not become CI pass/fail gates on uncontrolled shared hardware.

---

## 13. Test Data

Test fixtures must be:

* versioned;
* minimal;
* documented;
* deterministic;
* independent from live APIs.

Reference values must record their source and conventions.

Do not commit large generated outputs unless required for reproducibility.

---

## 14. Coverage

Coverage is diagnostic, not the goal.

Measure branch and line coverage, but prioritize:

* numerical boundaries;
* failure paths;
* method selection;
* model conventions;
* high-risk kernels.

High coverage does not compensate for weak assertions.

---

## 15. Test Performance

Keep the default suite fast enough for frequent execution.

Separate:

* fast developer tests;
* extended statistical tests;
* full validation tests;
* benchmarks.

Slow tests must remain automated and runnable through documented commands.

---

## Definition of Done

Testing is complete only when:

* every core module is independently tested;
* all required workflows have integration tests;
* trusted results have regression coverage;
* invariants have property tests;
* stochastic tests are statistically stable;
* failure paths are verified;
* concurrency is race-free;
* CI passes from a clean checkout;
* sanitizers report no unresolved defect;
* published claims are supported by automated tests or validation artifacts.

## Completion Blockers

The project is not done if:

* tests pass only on one machine;
* stochastic tests are flaky;
* references are updated without investigation;
* failures return plausible values;
* important code is tested only indirectly;
* concurrency lacks race testing;
* tolerances are arbitrary;
* sanitizer defects remain;
* CI skips required configurations;
* coverage is used as a substitute for correctness.

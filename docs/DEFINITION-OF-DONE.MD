# Definition of Done

## Purpose

Define the final acceptance contract for DiffusionWorks.

The project is complete only when every mandatory requirement below is satisfied with working code, passing tests, validated results, reproducible artifacts, and accurate documentation.

Explanation, intention, partial implementation, or visual polish cannot substitute for evidence.

---

## 1. Final Project Identity

The completed project must clearly present:

**DiffusionWorks: Stochastic Derivatives Numerics**

**A C++ Stochastic Derivatives Modeling, Validation and Performance Engine**

A reviewer must immediately recognize:

* native C++ numerical engineering;
* stochastic derivatives mathematics;
* model and method validation;
* convergence analysis;
* reproducible performance research.

The project must not read primarily as a dashboard, calculator, QuantLib wrapper, or classroom exercise.

---

## 2. Scope Completion

Version 1.0 must contain:

### Models

* Black–Scholes–Merton;
* Heston.

### Instruments

* European call and put;
* arithmetic Asian call and put;
* geometric Asian control instrument;
* down-and-out call;
* up-and-out call.

### Methods

* analytic Black–Scholes pricing and Greeks;
* geometric Asian analytic pricing;
* exact GBM simulation;
* Euler–Maruyama;
* Milstein;
* Heston full-truncation Euler;
* explicit finite difference;
* implicit finite difference;
* Crank–Nicolson;
* Heston characteristic-function pricing;
* finite-difference Greeks;
* pathwise Greek estimation;
* at least one likelihood-ratio Greek;
* antithetic and control variates;
* Heston calibration.

Any missing mandatory item blocks completion.

---

## 3. Mathematical Correctness

Done means:

* equations match implementation;
* assumptions and conventions are explicit;
* Black–Scholes prices and Greeks match trusted references;
* GBM simulation reproduces theoretical moments;
* Monte Carlo uncertainty is correctly estimated;
* Euler–Maruyama and Milstein convergence is measured;
* PDE methods converge toward analytic values;
* explicit instability is detected or demonstrated;
* Heston prices match independent references;
* Heston variance behavior is diagnosed;
* implied volatility fails safely when no root exists;
* calibration passes synthetic recovery testing;
* edge cases behave correctly or fail explicitly.

Plausible-looking output is not sufficient.

---

## 4. Validation

Done means:

* every required method has an independent validation path;
* put–call parity, bounds, monotonicity, and applicable barrier relationships pass;
* Monte Carlo confidence coverage is tested;
* stochastic conclusions use multiple independent seeds;
* PDE grid and boundary sensitivity are measured;
* Heston integration is cross-validated;
* calibration residuals and initialization sensitivity are reported;
* tolerances are method-specific and justified;
* all required failures are resolved;
* warnings and inconclusive results are explained.

No headline claim may depend on an unresolved validation result.

---

## 5. Experiments

All mandatory experiments in `experiment-catalog.md` must:

* run from stored configurations;
* produce machine-readable results;
* include full metadata;
* generate required tables and charts;
* report pass, fail, warning, or inconclusive status;
* include interpretation and limitations;
* provide exact reproduction commands.

Theoretical and empirical results must be reconciled.

Failed or disappointing outcomes must remain visible.

---

## 6. C++ Engineering

Done means:

* all core numerical logic is implemented in C++;
* Python is limited to analysis and presentation;
* QuantLib is limited to external validation;
* architecture has clear module responsibilities;
* ownership is explicit and safe;
* numerical components are independently testable;
* configuration and results use validated structured types;
* pricing methods return status, diagnostics, and metadata;
* stochastic execution is reproducible;
* multithreading uses deterministic path partitioning and thread-local state;
* no unjustified allocation remains inside hot loops;
* expected numerical failures never appear as valid results.

The repository must demonstrate maintainable C++ engineering, not merely C++ syntax.

---

## 7. Build and Tooling

From a clean checkout:

* CMake configuration succeeds;
* debug build succeeds;
* release build succeeds;
* sanitizer build succeeds;
* benchmark build succeeds;
* required dependencies resolve;
* example configurations run;
* generated outputs match their schemas.

Supported GCC and Clang CI jobs must pass.

---

## 8. Testing

Done means:

* unit tests cover core numerical components;
* integration tests cover every CLI workflow;
* regression tests protect trusted outputs;
* property tests verify financial invariants;
* statistical tests are stable and justified;
* failure paths are tested;
* concurrency paths are race-tested where supported;
* formatting and static analysis pass;
* AddressSanitizer and UndefinedBehaviorSanitizer report no unresolved defect;
* ThreadSanitizer reports no unresolved concurrency defect where supported;
* no flaky mandatory test remains.

Coverage alone does not satisfy this requirement.

---

## 9. Performance

Done means:

* benchmarks use validated release builds;
* workloads are equivalent;
* measurements are repeated;
* dispersion is reported;
* hardware, compiler, flags, threads, and commit are disclosed;
* Monte Carlo throughput is measured;
* parallel scaling is measured;
* PDE performance is measured;
* Greek estimator performance is measured;
* calibration cost is measured;
* memory and allocation behavior are profiled;
* accuracy-adjusted metrics are reported;
* every optimization has before-and-after evidence;
* all optimized paths are revalidated.

Unsupported claims such as “low latency,” “production-grade,” or universal superiority must not appear.

---

## 10. Reproducibility

Done means every published result can be regenerated using:

* stored configuration;
* stored or documented input data;
* fixed seed or documented seed set;
* exact command;
* Git commit;
* compiler and build settings;
* thread count;
* environment metadata.

Charts and tables must be generated from stored engine outputs without manual alteration.

A clean reproduction audit must succeed.

---

## 11. Documentation

Required documentation must:

* match implemented behavior;
* contain no aspirational features presented as complete;
* document assumptions, limitations, conventions, and failures;
* describe the final architecture accurately;
* include reproduction instructions;
* trace claims to generated evidence;
* use the fixed DiffusionWorks identity consistently.

API documentation and the technical paper must be complete.

The final `README.md` may be written only after all other acceptance gates pass.

---

## 12. Public Results Site

The completed site must:

* present the project identity clearly;
* foreground mathematics, C++, validation, and convergence;
* use generated results;
* contain no independent pricing implementation;
* expose methodology and limitations;
* remain usable on desktop and mobile;
* link to code, technical paper, validation, and benchmark evidence;
* avoid fake trading-terminal or dashboard-first presentation.

The site cannot compensate for incomplete engine work.

---

## 13. Final Acceptance Audit

Before completion, run:

1. clean checkout;
2. all build presets;
3. complete test suite;
4. sanitizer and static-analysis checks;
5. full validation suite;
6. all mandatory experiments;
7. benchmark reproduction;
8. chart and table regeneration;
9. documentation-code consistency audit;
10. public-claim evidence audit.

Any failure returns the project to the relevant build phase.

---

## Completion Blockers

DiffusionWorks is not done if any of the following remains:

* missing mandatory scope;
* unresolved validation failure;
* unreliable or flaky test;
* sanitizer or race defect;
* irreproducible result;
* undocumented numerical instability;
* external implementation of core logic;
* calibration without synthetic recovery;
* convergence claimed without measurement;
* benchmark without accuracy equivalence;
* unsupported performance claim;
* documentation-code mismatch;
* manually fabricated result or chart;
* website stronger than the underlying engine;
* inability to identify both mathematical rigor and C++ engineering at first review.

---

## Final Completion Standard

DiffusionWorks is complete only when a skeptical technical reviewer can independently verify:

> The models were implemented natively in C++, the numerical methods behave as claimed, errors and limitations are measured, results are reproducible, performance comparisons are fair, and every public statement is supported by working evidence.

Until that standard is met, continue iterating.

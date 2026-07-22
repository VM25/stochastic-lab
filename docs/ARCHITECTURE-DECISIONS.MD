# Architecture Decisions

## Purpose

Record the decisions that define DiffusionWorks and prevent inconsistent implementation. Change a decision only when measured evidence shows the replacement is better.

---

## ADR-001: C++ Owns the Numerical Core

**Decision:** Implement models, pricing, simulation, Greeks, calibration, validation, and benchmarks in C++.

**Why:** The project must visibly prove C++ engineering and quantitative implementation ability.

**Rejected:**

* Python-first engine;
* Python orchestration around external pricing libraries;
* duplicated C++ and Python engines.

---

## ADR-002: Python Is Analysis-Only

**Decision:** Use Python only for plotting, aggregation, report generation, and independent cross-checks.

**Why:** Python is useful for research presentation but must not dilute the C++ signal.

**Failure condition:** A published result depends on pricing logic implemented only in Python.

---

## ADR-003: QuantLib Is Validation-Only

**Decision:** QuantLib may provide external reference values.

**Why:** Independent comparison improves credibility; delegation of core work destroys it.

**Rejected:** Wrapping QuantLib instruments, engines, Greeks, or calibration as the main implementation.

---

## ADR-004: CLI-First Product

**Decision:** The authoritative interface is a configuration-driven CLI.

**Why:** A CLI is reproducible, testable, automatable, and independent from presentation code.

**Rejected:**

* website-first implementation;
* desktop GUI;
* notebook as the primary interface.

---

## ADR-005: JSON Configuration and Structured Results

**Decision:** Use versioned JSON for inputs and JSON/CSV for outputs.

**Why:** Experiments must be reproducible and machine-readable.

Every output must retain model, numerical, seed, build, hardware, and runtime metadata.

---

## ADR-006: Concrete Domain Types

**Decision:** Represent instruments, markets, models, requests, and results with validated domain types.

**Why:** Long primitive parameter lists are error-prone and obscure conventions.

**Rejected:** Passing unlabelled sequences of strings, integers, and doubles through the system.

---

## ADR-007: Pricing Logic Is Separate from Instruments

**Decision:** Instruments contain contract terms; pricing engines contain valuation logic.

**Why:** This prevents instrument classes from becoming coupled to models and methods.

---

## ADR-008: Controlled Polymorphism

**Decision:** Use runtime polymorphism for runtime engine selection and templates for compile-time policy selection.

**Why:** Neither inheritance everywhere nor templates everywhere is justified.

Use templates for:

* discretization;
* payoff evaluation;
* accumulators;
* hot simulation composition.

Do not build abstractions for models outside Version 1.0.

---

## ADR-009: Structured Results, Never Bare Values

**Decision:** Pricing and risk methods return structured results.

Required fields where applicable:

* estimate;
* uncertainty;
* diagnostics;
* runtime;
* status;
* warnings;
* metadata.

**Why:** A plausible number without evidence or status is unsafe.

---

## ADR-010: Deterministic Randomness

**Decision:** Use a documented master seed and reproducible stream partitioning.

**Why:** Published stochastic results must be repeatable.

**Rejected:**

* time-based thread seeds;
* shared mutable RNG;
* nondeterministic path allocation;
* undocumented random libraries or defaults.

---

## ADR-011: Independent Monte Carlo Paths

**Decision:** Partition paths deterministically across fixed workers, use thread-local statistics, then reduce results in a defined order.

**Why:** This minimizes synchronization and preserves reproducibility.

**Rejected:**

* locks inside path loops;
* shared accumulators;
* uncontrolled task scheduling.

---

## ADR-012: Standard Containers Before Specialized Memory

**Decision:** Begin with contiguous standard containers and preallocated buffers.

**Why:** Custom memory systems add complexity without guaranteed benefit.

Do not add specialized layouts, allocators, or SIMD.

---

## ADR-013: No Allocation in Hot Path Loops

**Decision:** Reuse path buffers and thread-local scratch storage.

**Why:** Per-step or per-path allocation is avoidable overhead and complicates
thread-local buffer ownership under the deterministic reduction.

---

## ADR-014: Double Precision by Default

**Decision:** Use IEEE double precision for primary calculations.

**Why:** It is the standard balance of precision and computational cost for this project.

Use higher precision only for reference analysis or diagnosed numerical failure.

---

## ADR-015: Numerical Methods Remain Independently Testable

**Decision:** Separate random generation, discretization, path evolution, payoff evaluation, statistics, grids, boundary conditions, integration, and solvers.

**Why:** Numerical defects must be isolatable.

**Rejected:** Monolithic pricing functions combining every stage.

---

## ADR-016: Exact GBM Is a Reference Method

**Decision:** Retain exact GBM simulation alongside Euler–Maruyama and Milstein.

**Why:** It provides controlled strong-error, weak-error, and pricing references.

Approximate schemes must use common Brownian increments when compared.

---

## ADR-017: Full-Truncation Euler Is the Primary Heston Scheme

**Decision:** Use full-truncation Euler for Version 1.0 Heston simulation.

**Why:** It is implementable, interpretable, and suitable for studying variance discretization.

**Rejected for Version 1.0:**

* silent variance flooring without diagnostics;
* multiple Heston schemes before full truncation is validated;
* Quadratic Exponential as a required feature.

---

## ADR-018: Three Finite-Difference Schemes

**Decision:** Implement explicit, implicit, and Crank–Nicolson methods.

**Why:** Their stability and accuracy differences are central research outputs.

The explicit method remains included even when it fails; instability must be detected and demonstrated.

---

## ADR-019: Native Tridiagonal Solver

**Decision:** Implement and test the tridiagonal solver directly.

**Why:** It is a core numerical component and small enough to own.

External linear algebra may support non-central matrix operations but must not hide the PDE algorithm.

---

## ADR-020: Heston Pricing Uses Characteristic-Function Integration

**Decision:** Implement semi-analytic Heston European pricing through a documented characteristic-function formulation and numerical quadrature.

**Why:** It provides a reference for Heston simulation and calibration.

Branch conventions, bounds, tolerances, and failure status must be explicit.

---

## ADR-021: Calibration Is a Diagnostic Process

**Decision:** Treat optimization output as a candidate calibration, not proof of success.

Calibration must report:

* residuals;
* initial guess;
* constraints;
* convergence status;
* evaluation count;
* synthetic recovery;
* sensitivity across initializations.

**Rejected:** Reporting only the final objective value and parameter vector.

---

## ADR-022: Implied Volatility Uses a Bracketed Solver

**Decision:** Use a bracketed or safeguarded hybrid root finder.

**Why:** Reliability matters more than nominal speed.

No root, invalid price bounds, or non-convergence must produce explicit failure.

---

## ADR-023: Validation Is Separate from Production Logic

**Decision:** Keep analytic references, invariants, QuantLib adapters, statistical checks, and stored references in a dedicated validation layer.

**Why:** Production methods must not validate themselves using duplicated internal assumptions.

---

## ADR-024: Method-Specific Tolerances

**Decision:** Define tolerances according to method, grid, sample size, and reference scale.

**Why:** Analytic, PDE, Monte Carlo, integration, and calibration errors are fundamentally different.

**Rejected:** One global epsilon or widening tolerances until tests pass.

---

## ADR-025: Benchmarks Require Correctness Equivalence — Withdrawn

**Status:** Withdrawn. The benchmarking and optimization scope was removed from the
project; there is no performance-measurement work to govern. The number is retained so
earlier ADR cross-references stay stable. Multithreading, deterministic reductions, and
race-safety remain in scope under ADR-010/011 and the multithreading methodology, but no
throughput, scaling, or speedup measurement is performed or claimed.

---

## ADR-026: Optimization Follows Profiling — Withdrawn

**Status:** Withdrawn, with ADR-025. No profiling requirement or optimization workflow
exists: correctness, validation, and reproducibility are the engineering contract, and
the code is not tuned against performance measurements.

---

## ADR-027: Website Is a Read-Only Presentation Layer

**Decision:** The website displays generated results and documentation.

**Why:** The C++ engine must remain the single source of truth.

A limited WebAssembly demonstration is optional only after Version 1.0 completion.

---

## ADR-028: Failed Experiments Remain Visible

**Decision:** Preserve unstable, contradictory, or disappointing results with explanations.

**Why:** Failure analysis demonstrates numerical judgment and prevents selective reporting.

**Rejected:** Removing runs because they weaken the intended narrative.

---

## ADR-029: Version 1.0 Scope Is Closed

**Decision:** Version 1.0 includes only:

* Black–Scholes and Heston;
* European, Asian, and barrier options;
* required analytic, Monte Carlo, finite-difference, Greek, calibration, and validation work.

**Rejected until completion:**

* American options;
* SABR;
* local or rough volatility;
* jump diffusion;
* rates or credit models;
* GPU acceleration.

---

## ADR-030: Documentation Must Follow Implemented Reality

**Decision:** Equations, architecture, experiments, and public claims must match working code.

**Why:** Documentation that describes intended but absent behavior is misleading.

The final `README.md` must not be written until implementation and results are complete.

---

## Completion Blockers

The architecture is not complete if:

* Python or QuantLib performs core calculations;
* instruments contain pricing logic;
* pricing methods return bare values;
* stochastic runs are not reproducible;
* multithreading introduces hidden shared state;
* numerical components cannot be tested independently;
* calibration failures appear as successful outputs;
* website logic diverges from the C++ engine;
* abstractions are built for excluded models;
* documentation and implementation disagree.

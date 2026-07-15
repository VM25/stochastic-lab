# Benchmark Plan

## Purpose

Measure DiffusionWorks only after correctness is established.

Benchmarks must compare equivalent work, disclose methodology, and distinguish raw speed from accuracy-adjusted efficiency.

---

## 1. Benchmark Principles

1. Validate before benchmarking.
2. Use release builds.
3. Compare equivalent outputs and accuracy.
4. Report repeated measurements and dispersion.
5. Disclose hardware, compiler, flags, threads, and Git commit.
6. Preserve unfavorable results.
7. Never claim low latency, production performance, or linear scaling without evidence.

---

## 2. Required Environment Metadata

Every benchmark must record:

* CPU and core count;
* memory;
* operating system;
* compiler and version;
* build type and flags;
* C++ standard;
* thread count;
* Git commit;
* configuration;
* random seed;
* benchmark repetitions.

---

## 3. Execution Rules

Each benchmark must use:

* fixed workloads;
* warm-up runs;
* repeated timed runs;
* median runtime;
* dispersion measure;
* isolated output generation;
* disabled debug logging;
* equivalent numerical settings;
* deterministic seeds where applicable.

Avoid unrelated workloads during measurement.

---

## 4. Monte Carlo Throughput

Measure:

* paths per second;
* time steps per second;
* runtime per valuation.

Test:

* exact GBM;
* Euler–Maruyama;
* Milstein;
* Heston full truncation;
* European, Asian, and barrier payoffs.

Vary:

* path count;
* step count;
* thread count.

---

## 5. Parallel Scaling

For thread count (p):

[
Speedup(p)=\frac{T_1}{T_p}
]

[
Efficiency(p)=\frac{Speedup(p)}{p}
]

Report:

* runtime;
* throughput;
* speedup;
* efficiency;
* saturation point;
* result differences caused by reduction order.

Use identical workloads across thread counts.

---

## 6. Pricing Method Comparison

Where methods support the same instrument, compare:

* analytic;
* Monte Carlo;
* finite difference;
* Heston integration.

Primary measures:

* runtime to target error;
* RMSE per second;
* setup cost;
* memory usage;
* scalability.

Do not rank methods using raw runtime when accuracy differs.

---

## 7. Variance-Reduction Benchmarks

Compare:

* crude Monte Carlo;
* antithetic variates;
* control variates;
* combined methods.

Measure:

* variance;
* RMSE;
* runtime;
* variance-reduction ratio;
* efficiency:

[
Efficiency=\frac{1}{Variance\times Runtime}
]

Use common configurations and seed sets.

---

## 8. Finite-Difference Benchmarks

Measure:

* runtime by grid size;
* memory by grid size;
* error by grid size;
* runtime to target tolerance.

Compare:

* explicit;
* implicit;
* Crank–Nicolson.

Unstable explicit configurations must be reported, not benchmarked as valid outputs.

---

## 9. Greek Benchmarks

Compare:

* analytic;
* bump-and-revalue;
* pathwise;
* likelihood ratio.

Measure:

* runtime;
* bias;
* variance;
* RMSE;
* memory;
* number of valuations required.

Use common random numbers where applicable.

---

## 10. Calibration Benchmarks

Measure:

* total runtime;
* objective evaluations;
* pricing calls;
* convergence rate;
* final fit error;
* variation across initial guesses.

A faster failed calibration is not superior.

---

## 11. Memory and Allocation

Profile:

* peak memory;
* allocations per valuation;
* allocations inside simulation loops;
* path-buffer reuse;
* thread-local storage.

Required outcome:

* no avoidable allocation inside hot path loops;
* no unexplained memory growth;
* no performance claim based solely on allocation count.

---

## 12. Optimization Process

For each optimization:

1. establish a validated baseline;
2. profile;
3. identify the bottleneck;
4. change one major factor;
5. rerun validation;
6. rerun benchmarks;
7. record before-and-after results.

Reject optimizations that:

* reduce correctness;
* damage reproducibility;
* complicate ownership without measurable benefit;
* improve only synthetic microbenchmarks.

---

## 13. Required Artifacts

Each benchmark suite must produce:

* versioned configuration;
* raw machine-readable results;
* environment metadata;
* summary table;
* required chart;
* reproduction command;
* interpretation;
* limitations.

Published charts must be generated automatically from stored results.

---

## 14. Permitted Claims

Claims may include:

* measured throughput;
* observed scaling;
* runtime reduction;
* memory reduction;
* RMSE-per-second improvement;
* runtime to target accuracy.

Claims must state the tested environment and workload.

---

## 15. Prohibited Claims

Do not claim:

* low latency;
* production-grade performance;
* real-time trading suitability;
* universal superiority;
* near-linear scaling;
* material optimization gains

unless directly supported by reproducible evidence.

---

## Definition of Done

Benchmarking is complete only when:

* all required suites run in release mode;
* correctness equivalence is established;
* hardware and methodology are disclosed;
* repeated results include dispersion;
* thread scaling is measured;
* memory behavior is profiled;
* accuracy-adjusted comparisons are reported;
* optimization claims have before-and-after evidence;
* all published results are reproducible.

## Completion Blockers

The project is not done if:

* debug builds are benchmarked;
* workloads differ unfairly;
* accuracy is ignored;
* only best runs are reported;
* hardware metadata is missing;
* benchmarks cannot be reproduced;
* performance changes bypass revalidation;
* unstable outputs are treated as fast results;
* unsupported performance language remains.

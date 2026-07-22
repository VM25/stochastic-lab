# Failure Modes

## Purpose

Define conditions that make DiffusionWorks incomplete, misleading, or technically weak.

If any completion-blocking failure remains, continue iterating until it is corrected, validated, and documented.

---

## 1. Dashboard-First Development

**Warning signs:**

* website work begins before the engine is validated;
* visual polish exceeds mathematical substance;
* charts use manually prepared values.

**Why it fails:** The project becomes another interface rather than proof of mathematics and C++.

**Required remediation:**

* stop presentation work;
* complete the engine, validation, and experiments;
* regenerate visuals only from stored outputs.

**Completion blocker:** The website is more complete or credible than the numerical core.

---

## 2. Feature and Model Sprawl

**Warning signs:**

* new models are added before required methods pass;
* unfinished components accumulate;
* breadth is used to avoid difficult validation.

**Why it fails:** More features do not compensate for shallow or incorrect work.

**Required remediation:**

* return to Version 1.0 scope;
* remove or defer non-required features;
* finish current phase gates.

**Completion blocker:** Any required feature remains incomplete while optional scope is active.

---

## 3. External Libraries Perform Core Work

**Warning signs:**

* QuantLib prices instruments used as primary outputs;
* Python contains pricing, calibration, or simulation logic;
* native code only wraps external engines.

**Why it fails:** The project no longer proves C++ or mathematical implementation ability.

**Required remediation:**

* implement the central method natively in C++;
* retain external outputs only for comparison.

**Completion blocker:** Any headline result depends on external core calculations.

---

## 4. Mathematics and Code Diverge

**Warning signs:**

* equations do not match discretization code;
* conventions differ across modules;
* undocumented approximations appear;
* documentation describes intended rather than implemented behavior.

**Why it fails:** Neither the mathematics nor the software can be trusted.

**Required remediation:**

* reconcile equations, assumptions, code, and tests;
* rerun all affected validation.

**Completion blocker:** Any public equation or claim contradicts working code.

---

## 5. Plausible Results Replace Validation

**Warning signs:**

* outputs are accepted because they look reasonable;
* only one reference case is tested;
* errors are not quantified.

**Why it fails:** Numerical defects often produce believable values.

**Required remediation:**

* add independent references, invariants, convergence tests, and edge cases;
* report pass, fail, warning, or inconclusive status.

**Completion blocker:** Any required method lacks independent validation.

---

## 6. Convergence Is Claimed, Not Measured

**Warning signs:**

* only two resolutions are compared;
* slopes are judged visually;
* sampling noise exceeds measured discretization error;
* different Brownian paths are used in strong-error tests.

**Why it fails:** The project cannot prove numerical-method understanding.

**Required remediation:**

* use sufficient grid levels and seeds;
* estimate slopes with uncertainty;
* separate sampling and discretization effects;
* use coupled paths where required.

**Completion blocker:** Required convergence behavior remains unsupported or unexplained.

---

## 7. Stochastic Results Are Not Reproducible

**Warning signs:**

* seeds are missing;
* threads use time-based seeds;
* repeated runs cannot recreate results;
* only favorable seeds are published.

**Why it fails:** Results cannot be audited or trusted.

**Required remediation:**

* store master seed and stream strategy;
* use deterministic partitioning;
* aggregate multiple independent seeds;
* retain reproduction commands.

**Completion blocker:** Any published stochastic result cannot be regenerated.

---

## 8. Statistical Uncertainty Is Mishandled

**Warning signs:**

* Monte Carlo outputs omit standard errors;
* confidence intervals systematically under-cover;
* single-run error is presented as estimator quality;
* variance reduction is judged from one sample.

**Why it fails:** Random error is confused with estimator quality.

**Required remediation:**

* validate estimators across seeds;
* test confidence coverage;
* report RMSE, variance, and uncertainty;
* revise invalid interval methods.

**Completion blocker:** Headline stochastic claims lack statistically defensible evidence.

---

## 9. PDE Stability Is Confused with Accuracy

**Warning signs:**

* implicit or Crank–Nicolson output is trusted because it remains finite;
* explicit instability is hidden;
* boundary and grid sensitivity are omitted.

**Why it fails:** Stable numerical output may still be materially wrong.

**Required remediation:**

* run grid, time-step, and boundary convergence tests;
* detect unstable configurations;
* compare against analytic references.

**Completion blocker:** PDE errors or stability limits remain unmeasured.

---

## 10. Heston Variance Failures Are Hidden

**Warning signs:**

* negative variance is silently replaced;
* Feller-condition failure is treated as invalidity or ignored;
* non-finite paths disappear from results.

**Why it fails:** The simulation conceals its principal numerical weakness.

**Required remediation:**

* report pre-truncation negatives and path failures;
* measure discretization bias;
* test difficult parameter regimes;
* fail explicitly on non-finite states.

**Completion blocker:** Heston simulation can silently produce or conceal invalid states.

---

## 11. Calibration Is Mistaken for Optimization Success

**Warning signs:**

* only final parameters and objective value are reported;
* one initial guess is used;
* synthetic recovery is absent;
* similar fits with different parameters are ignored.

**Why it fails:** A converged optimizer does not prove an identified or reliable model.

**Required remediation:**

* test multiple starts, weights, and bounds;
* perform synthetic recovery;
* report residuals and parameter dispersion;
* expose non-convergence and non-uniqueness.

**Completion blocker:** Calibration quality cannot be independently assessed.

---

## 12. Greek Estimators Are Applied Blindly

**Warning signs:**

* pathwise methods are used for discontinuous payoffs without justification;
* finite-difference bumps are arbitrary;
* Monte Carlo bumps use unrelated random streams.

**Why it fails:** Greek estimates may be biased, unstable, or dominated by noise.

**Required remediation:**

* validate against analytic Greeks;
* run bump-size studies;
* use common random numbers;
* reject unsupported estimator-payoff combinations.

**Completion blocker:** Required Greek outputs lack validated accuracy and limitations.

---

## 13. C++ Looks Superficial

**Warning signs:**

* code is monolithic;
* ownership is unclear;
* hot loops allocate repeatedly;
* tests cannot isolate numerical components;
* modern features are decorative rather than useful.

**Why it fails:** The repository does not prove professional C++ engineering.

**Required remediation:**

* clarify domain types and ownership;
* separate numerical components;
* remove unnecessary abstraction;
* improve testing and build discipline.

**Completion blocker:** Core methods cannot be independently understood, tested, or maintained.

---

## 14. Failures Return Plausible Values

**Warning signs:**

* root failures return zero;
* optimizer failure returns the latest parameters as success;
* NaN is replaced silently;
* unstable PDE output receives normal status.

**Why it fails:** Consumers cannot distinguish valid results from numerical failure.

**Required remediation:**

* use explicit result states;
* attach diagnostics and warnings;
* return nonzero CLI codes;
* add failure-path tests.

**Completion blocker:** Any expected numerical failure can appear successful.

---

## 15. Tests Pass for the Wrong Reasons

**Warning signs:**

* tolerances are widened after failure;
* reference outputs are overwritten without investigation;
* stochastic tests are flaky;
* high coverage substitutes for meaningful assertions.

**Why it fails:** The test suite certifies itself rather than the system.

**Required remediation:**

* diagnose the numerical cause;
* justify tolerances;
* stabilize stochastic tests;
* add targeted assertions and regression cases.

**Completion blocker:** Required tests are unreliable, weak, or manipulated.

---

## 16. Results Cannot Be Reproduced

**Warning signs:**

* configurations are missing;
* charts require manual edits;
* environment metadata is incomplete;
* clean checkout commands fail.

**Why it fails:** Published evidence cannot be independently verified.

**Required remediation:**

* version inputs and outputs;
* automate charts and tables;
* store reproduction commands;
* rerun from a clean environment.

**Completion blocker:** Any headline result lacks a complete reproduction path.

---

## 17. Documentation Becomes Aspirational

**Warning signs:**

* documents claim unfinished features;
* examples no longer run;
* architecture descriptions are outdated;
* limitations disappear from public materials.

**Why it fails:** The project becomes misleading.

**Required remediation:**

* audit documentation against code and artifacts;
* remove unsupported claims;
* update examples and limitations.

**Completion blocker:** Public documentation describes functionality that does not exist.

---

## 18. Final Identity Is Lost

**Warning signs:**

* the project is presented as a generic options calculator;
* C++ is hidden behind the website;
* mathematical evidence is difficult to locate.

**Why it fails:** DiffusionWorks no longer solves its primary profile objective.

**Required remediation:**

* foreground native C++, numerical methods, validation, and convergence;
* simplify presentation;
* remove unrelated features.

**Completion blocker:** A reviewer cannot quickly identify both stochastic-mathematics depth and C++ engineering.

---

## Final Failure Gate

DiffusionWorks is not done while any of the following remains:

* unresolved required validation failure;
* irreproducible result;
* hidden numerical instability;
* unsupported public claim;
* external core implementation;
* unfinished mandatory scope;
* documentation-code mismatch;
* unreliable test;
* presentation stronger than the underlying engine.

Do not explain around a failed requirement. Fix it, rerun it, and produce evidence.

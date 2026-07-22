# DiffusionWorks: Project Identity

## Official Name

**DiffusionWorks: Stochastic Derivatives Numerics**

## Subtitle

**A C++ Stochastic Derivatives Modeling and Validation Engine**

## Repository

`stochastic-lab`

## Core Identity

DiffusionWorks is a C++ computational-finance system that implements stochastic derivatives models and numerical pricing methods from first principles, then evaluates them across:

* pricing accuracy;
* numerical convergence;
* statistical reliability;
* model validation.

It is not primarily an options calculator, dashboard, trading simulator, or QuantLib wrapper.

## Core Thesis

A numerical price is not trustworthy because it was successfully computed.

DiffusionWorks must show:

1. whether the model and algorithm are mathematically correct;
2. whether the numerical result converges toward a defensible reference;
3. where approximation error, estimator variance, instability, or model risk enters;
4. how much computation is required to achieve a target accuracy.

The central project question is:

> How should stochastic derivatives models be translated into correct, validated, reproducible, and performant C++ software?

## Primary Signal

DiffusionWorks must visibly demonstrate:

* stochastic calculus;
* probability and statistics;
* numerical analysis;
* derivatives pricing;
* model calibration;
* modern C++ engineering;
* testing and validation;
* quantitative research judgment.

The project must read as mathematical and computational work first, not visual product work.

## Intended Audience

The public project should be immediately relevant to:

* quantitative developers;
* quantitative researchers;
* derivatives quants;
* model-validation quants;
* financial engineers;
* risk quants;
* research engineers.

A recruiter should understand the project within seconds. A technical reviewer should be able to inspect the mathematics, code, experiments, and validation evidence in depth.

## Final Product Character

DiffusionWorks should feel like:

* a numerical research system;
* a computational-finance codebase;
* a reproducible experiment framework;
* a professional C++ engineering project;
* a technical validation study.

It must not feel like:

* a classroom assignment;
* a collection of disconnected pricing formulas;
* another finance dashboard;
* a feature-heavy but unvalidated platform;
* a thin interface over external libraries;
* a performance demo without correctness evidence.

## Identity Principles

1. **Correctness before breadth.**
2. **Validation before presentation.**
3. **Measured results before claims.**
4. **Mathematical clarity before abstraction.**
5. **C++ owns the computational core.**
6. **Python may support analysis and plotting only.**
7. **QuantLib may validate results only.**
8. **Failures, limitations, and unstable cases must remain visible.**
9. **New scope cannot replace unfinished validation.**
10. **The project is not complete until every required result is reproducible and defensible.**

## Naming Rules

Use the following consistently across all documentation and public materials:

* **DiffusionWorks** — short project name.
* **DiffusionWorks: Stochastic Derivatives Numerics** — full project name.
* **A C++ Stochastic Derivatives Modeling and Validation Engine** — official subtitle.
* **`stochastic-lab`** — repository and local folder name.

Do not rename the project, abbreviate it as `DW`, or substitute generic labels such as “pricing engine,” “quant platform,” or “stochastic lab” as the primary public identity.

## Definition of Identity Success

The identity is successful only when the completed project clearly proves:

> DiffusionWorks is a serious C++ stochastic-derivatives numerics project built to test not only whether a method produces a result, but whether that result is accurate, convergent, reproducible, explainable, and computationally efficient.
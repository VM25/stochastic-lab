# Design Specification

## Purpose

Design DiffusionWorks as a visual extension of its mathematical and computational identity.

The interface must help users understand:

* stochastic models;
* numerical methods;
* convergence;
* pricing error;
* validation;
* calibration.

It must not exist as decoration around the engine.

---

## 1. Design Process Requirement

Do not invent the interface directly.

Before implementation, use every available specialized capability relevant to:

* product design;
* UX research;
* information architecture;
* Figma;
* data visualization;
* accessibility;
* responsive design;
* frontend engineering;
* visual QA.

Use those capabilities to:

1. define the visual system;
2. generate and compare design directions;
3. select the strongest direction;
4. create the interface specification;
5. implement it;
6. compare the implementation against the approved design;
7. iterate until visual and functional QA passes.

Do not substitute generic personal design instincts for available specialist tools -- SKILL/MCPs/Plugins.

---

## 2. Design Objective

The completed site must feel like:

* a computational-finance laboratory;
* a numerical research instrument;
* a stochastic-calculus workspace;
* a quantitative model-validation environment.

It must communicate:

* rigor;
* precision;
* controlled experimentation;
* mathematical depth;
* computational evidence.

It must not look like:

* a SaaS dashboard;
* a trading terminal imitation;
* a fintech landing page;
* a research paper rendered as a website;
* a commercial analytics product;
* a decorative portfolio case study.

---

## 3. Core Visual Concept

The visual system should be built around the transformation:

```text
Model
→ Numerical Method
→ Computation
→ Error
→ Validation
→ Evidence
```

Every major page should make this chain visible.

The user should always understand:

* what was modeled;
* how it was computed;
* what reference was used;
* how much error remained;
* how much computation was required;
* whether the result passed validation.

---

## 4. Required Pages

### Overview

Show:

* project identity;
* core research question;
* implemented models and methods;
* strongest validated results;
* links to code, paper, experiments, and validation.

### Models

Show:

* Black–Scholes and Heston equations;
* parameter definitions;
* assumptions;
* model behavior;
* failure regions.

### Numerical Methods

Show:

* analytic pricing;
* Monte Carlo;
* Euler–Maruyama;
* Milstein;
* finite-difference methods;
* Heston integration;
* calibration.

### Convergence

Show:

* sampling convergence;
* strong convergence;
* weak convergence;
* discretization bias;
* empirical slope estimates;
* uncertainty.

### Validation

Show:

* reference comparisons;
* financial invariants;
* confidence coverage;
* PDE checks;
* Heston validation;
* calibration recovery;
* unresolved limitations.

### Architecture

Show:

* C++ module structure;
* simulation data flow;
* concurrency model;
* validation boundary;
* Python and QuantLib boundaries.

### Experiments

Provide direct access to every mandatory experiment, configuration, output, chart, status, and reproduction command.

---

## 5. Information Hierarchy

Each result view should follow:

1. research question;
2. method;
3. reference;
4. result;
5. error or uncertainty;
6. computational cost;
7. interpretation;
8. limitations;
9. reproduction details.

Do not show charts without an explicit question and conclusion.

Do not hide assumptions or failure states behind tooltips.

---

## 6. Visual Language

Use visual elements derived from:

* discretized paths;
* computational grids;
* parameter surfaces;
* error bands;
* residual structures;
* matrix organization;
* convergence slopes;
* numerical checkpoints.

These elements must represent real project concepts.

Do not use decorative stock charts, random curves, glowing particles, financial meshes, or simulated market movement.

---

## 7. Layout

Use:

* structured grids;
* sharp divisions;
* disciplined spacing;
* aligned numerical columns;
* strong hierarchy;
* dense but readable technical sections;
* clear transitions between model, method, and evidence.

Avoid:

* rounded-card collections;
* floating glass panels;
* large empty marketing sections;
* endless dashboard tiles;
* blog-style single-column reading;
* decorative asymmetry that harms comparison.

Dense information is acceptable when hierarchy remains clear.

---

## 8. Typography

Typography must distinguish:

* equations;
* narrative text;
* numerical outputs;
* labels;
* code;
* experiment metadata.

Choose a compact, technical, non-generic type system using specialist design tools.

### Forbidden Families

Do not use:

* Anthropic;
* Fraunces;
* Archivo;
* IBM Plex;
* Inter;
* Geist;
* Grotesk;
* Grotesque;
* Big Shoulders;
* Saira;
* Public Sans;
* Spline;
* Arial;
* Calibre;
* Times New Roman;
* JetBrains;
* DM Mono;
* DM Sans;
* Newsreader;
* Red Hat;
* Sora;
* Libre Franklin;
* Source Serif 4;
* Terminal;
* Cascadia;
* B612;
* Spectral;
* any variant of these families.

`Source Code Pro` may be used only for fully capitalized text.

Do not use a typeface merely because it appears technical.

---

## 9. Color System

The palette must support:

* mathematical distinction;
* error severity;
* method comparison;
* confidence and uncertainty;
* pass, warning, fail, and inconclusive states;
* readable charts.

Color must have semantic purpose.

### Dark-Theme Restrictions

Avoid dominant use of:

* cyan;
* steel;
* graphite;
* amber;
* gold;
* yellow;
* orange;
* slate;
* dark gray;
* dark blue;
* pink;
* magenta;
* purple;
* violet;
* indigo;
* red;
* green;
* near-black.

### Light-Theme Restrictions

Avoid dominant use of:

* warm tones;
* beige;
* pure white;
* cream;
* off-white;
* gold;
* amber;
* yellow;
* light blue;
* light gray;
* orange;
* green;
* light green.

Do not produce a simple black-and-white theme.

A restricted color may appear only when:

* it has necessary semantic meaning;
* no stronger alternative exists;
* it does not dominate the identity;
* the full system remains distinctive.

Do not select a theme until multiple specialist-generated directions have been compared.

---

## 10. Charts

Charts must be generated from DiffusionWorks outputs.

Required chart types include:

* log-log convergence plots;
* error versus path count;
* error versus time-step size;
* runtime versus accuracy;
* confidence coverage;
* calibration residual surfaces;
* parameter sensitivity;
* PDE grid convergence;
* variance-reduction efficiency.

Every chart must include:

* units;
* method;
* reference;
* sample or grid size;
* uncertainty where relevant;
* configuration access;
* concise interpretation.

Do not animate financial curves or data merely for visual effect.

---

## 11. Equations and Numbers

Equations must be readable and integrated into the interface rather than rendered as decorative wallpaper.

Numerical outputs should support:

* aligned decimals;
* scientific notation;
* confidence intervals;
* error magnitude;
* reference comparison;
* units and conventions.

Do not enlarge arbitrary numbers as marketing statistics.

A headline metric is permitted only when it communicates a validated result.

---

## 12. Interaction

Useful interactions include:

* comparing numerical methods;
* switching parameter regimes;
* viewing reference and error together;
* inspecting experiment metadata;
* filtering experiment results;
* revealing assumptions;
* downloading generated artifacts;
* opening reproduction commands.

Avoid:

* motion without analytical purpose;
* parallax;
* scroll-controlled chart theatrics;
* simulated terminal typing;
* random graph animation;
* hover-only critical information;
* novelty interactions that delay access to results.

---

## 13. Motion

Motion may be used only to explain:

* path generation;
* time stepping;
* grid propagation;
* convergence;
* method comparison;
* state transitions.

Motion must be:

* optional;
* restrained;
* interruptible;
* accessible;
* disabled under reduced-motion preferences.

Do not animate continuously.

---

## 14. Forbidden UI Patterns

Do not use:

* rounded cards;
* SaaS dashboard layouts;
* cybersecurity aesthetics;
* neon visuals;
* terminal or PowerShell aesthetics;
* product-selling brand identity;
* generic trading-terminal interfaces;
* fake live-market elements;
* random graph, mesh, path, or curve effects;
* decorative financial animations;
* excessive glassmorphism;
* standard hero-plus-feature-card landing pages.

---

## 15. Light and Dark Themes

A light or dark theme is permitted only if it serves the project.

The selected theme must:

* remain distinct from generic developer tooling;
* support equations and dense numerical comparison;
* preserve chart readability;
* avoid the forbidden palette patterns;
* strengthen the computational-finance identity.

A light theme must not resemble:

* a paper;
* a newspaper;
* an article;
* a blog;
* academic typesetting transferred directly to the web.

---

## 16. Responsive Design

Desktop is the primary analytical surface, but mobile must remain usable.

Desktop must support:

* side-by-side comparisons;
* dense charts;
* parameter tables;
* experiment navigation.

Mobile must:

* preserve hierarchy;
* stack comparisons deliberately;
* retain equations and metadata;
* avoid horizontal overflow;
* keep charts readable;
* expose no desktop-only critical information.

Shrinking the desktop interface is not an acceptable mobile design.

---

## 17. Accessibility

Required:

* keyboard navigation;
* visible focus states;
* semantic structure;
* chart text alternatives;
* sufficient contrast;
* reduced-motion support;
* non-color status indicators;
* readable mathematical notation;
* accessible tables;
* descriptive control labels.

Accessibility failures block completion.

---

## 18. Performance

The site must:

* load precomputed results efficiently;
* avoid unnecessary client-side computation;
* lazy-load heavy figures;
* preserve responsive interaction;
* avoid excessive animation libraries;
* minimize layout shift;
* remain usable on a standard laptop.

The website must not recreate computations already performed by C++.

---

## 19. Design QA

Before completion:

1. compare the implementation against the approved design;
2. inspect desktop and mobile states;
3. test every interaction;
4. verify all charts against source outputs;
5. check typography and palette restrictions;
6. run accessibility checks;
7. remove generic or decorative elements;
8. repeat until all failures are resolved.

Use specialist design and visual-QA tools for this process.

---

## Definition of Done

The design is complete only when:

* the visual identity clearly belongs to DiffusionWorks;
* mathematics, C++, validation, and convergence dominate the experience;
* every chart contributes to a research question;
* all displayed results come from engine artifacts;
* the interface avoids every forbidden pattern;
* desktop and mobile are fully usable;
* accessibility requirements pass;
* design QA finds no material mismatch;
* no visual element exists without a project-serving purpose.

## Completion Blockers

The project is not done if:

* the site resembles a SaaS dashboard or trading terminal;
* design was created without available specialist tools;
* forbidden fonts or dominant color treatments remain;
* decorative motion or charts distract from evidence;
* the site contains manually fabricated results;
* mobile is a compressed desktop layout;
* critical information depends on hover or color;
* the website hides uncertainty, failures, or limitations;
* the presentation is stronger than the underlying mathematics and C++.

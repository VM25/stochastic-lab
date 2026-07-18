# Version Control

## Purpose

Maintain DiffusionWorks in a GitHub repository with continuous, transparent, and meaningful version history.

Version control is a required project capability, not an administrative afterthought.

---

## Repository Setup

Create a repository on the owner’s GitHub account named:

```text
stochastic-lab
```

Use:

* default branch: `main`;
* repository visibility: public unless instructed otherwise;
* repository owner: the user only.

If GitHub access is unavailable, unauthorized, disconnected, or expired, stop and ask the user to authorize access before continuing.

Do not create the repository under another account or organization.

---

## Attribution Rules

Vatsal (https://github.com/VM25) must be the only visible author and contributor.

Claude should not appear anywhere on GitHub as:

* co-author;
* contributor;
* collaborator;
* bot author;
* commit trailer;
* pull-request author;
* issue author;
* release author.

Do not add:

```text
Co-authored-by:
Generated-by:
Assisted-by:
```

Commit identity must use the user’s configured Git username.

## Branch Strategy

Use only:

```text
main
```

Do not create feature, development, experiment, or release branches unless the user explicitly requests them.

Push completed work directly to `main`.

---

## Commit Frequency

Commit and push after every meaningful:

* phase or milestone completion;
* feature implementation;
* numerical method addition;
* experiment addition;
* test or validation addition;
* bug fix;
* refactor;
* optimization;
* configuration change;
* documentation update;
* file addition, deletion, or relocation;
* resolved failure or completed acceptance gate.

Do not accumulate large unrelated changes into one commit.

Do not leave completed work only on the local machine.

---

## Commit Quality

Commits must be:

* small;
* coherent;
* descriptive;
* buildable where practical;
* traceable to a specific change.

Use imperative commit messages, for example:

```text
Add analytic Black-Scholes pricing
Validate GBM terminal moments
Implement Milstein path simulation
Fix Crank-Nicolson boundary handling
Add Heston calibration recovery test
Document Monte Carlo convergence results
```

Do not use vague messages such as:

```text
update
changes
fix stuff
work
final
```

---

## Contribution History Goal

Maintain a dense, continuous contribution history by committing meaningful progress frequently.

Do not manufacture contribution activity through:

* empty commits;
* whitespace-only changes;
* artificial file churn;
* repeated renaming;
* meaningless formatting changes;
* splitting one indivisible change solely to inflate counts.

The contribution history must reflect real project development. High commit volume is valuable only when the history remains credible under review.

---

## Push Rule

After every commit:

```bash
git push origin main
```

If a push fails:

1. preserve all local work;
2. diagnose authentication, remote, or synchronization issues;
3. resolve the failure;
4. push successfully before treating the task as complete.

A completed local change is not complete until it exists on GitHub.

---

## Safety Rules

Before destructive Git operations:

* inspect repository status;
* verify the active branch;
* confirm the target remote;
* preserve uncommitted work.

Do not use:

```bash
git push --force
git reset --hard
git clean -fd
```

unless strictly necessary and explicitly approved by the user.

Never rewrite public history merely to improve commit structure.

---

## Required Checks Before Push

Where applicable, run:

* build;
* relevant tests;
* formatting;
* static analysis;
* validation affected by the change.

Do not knowingly push broken core functionality without clearly marking it as an incomplete intermediate state.

Milestone-completion commits must pass all milestone exit criteria.

### Formatting runs before the commit, not after CI

Formatting is checked *locally, before each commit* — never discovered as a red CI
job after several commits have accumulated. A committed pre-commit hook enforces this:

```bash
git config core.hooksPath scripts/git-hooks
```

`scripts/git-hooks/pre-commit` runs the pinned `clang-format` over the staged C++ and
rejects a commit that is not formatting-clean. It runs only `clang-format` (sub-second);
the slower `clang-tidy` and include-hygiene checks stay in `scripts/lint.sh` and CI,
because a multi-minute pre-commit hook would only invite `--no-verify`. Set the hooks
path once per clone (the setting is local and cannot be committed). When a fix is
needed, `scripts/lint.sh --fix` reformats in place.

---

## Repository Hygiene

Do not commit:

* credentials;
* API keys;
* tokens;
* private data;
* build directories;
* compiler artifacts;
* temporary files;
* local IDE state;
* large reproducible outputs without justification.

Maintain an accurate `.gitignore`.

Commit experiment results only when required for validation, reproducibility, or public evidence.

---

## Completion Criteria

Version control is correctly implemented only when:

* `stochastic-lab` exists on the user’s GitHub account;
* `main` is the authoritative branch;
* the user is the only visible author;
* every meaningful project increment is committed and pushed;
* commit messages clearly describe the work;
* history is frequent but not artificially inflated;
* no completed milestone exists only locally;
* no sensitive information is committed;
* the GitHub history accurately reconstructs the development of DiffusionWorks.

## Completion Blockers

The project is not done if:

* the repository is missing or owned by another account;
* authorization failure is ignored;
* another author or co-author appears;
* completed work remains unpushed;
* large unrelated changes are hidden in single commits;
* commit history is artificially manufactured;
* destructive history rewriting occurs without approval;
* sensitive or generated junk files are committed;
* the public repository does not reflect the current project state.

# Benchmark Capture Methodology

How the Phase 13 Monte Carlo scaling numbers are captured, why two monolithic capture
attempts on the development laptop were rejected, what controlled protocol replaced them,
and why the headline multi-thread numbers must be produced on a dedicated fixed machine.

This note complements `MULTITHREADING-METHODOLOGY.md`. That note establishes *correctness*
under threading (one-thread equivalence, fixed-count reproducibility, bounded
reassociation); this note is about *measurement* -- the conditions a runtime figure must be
taken under before it can be reported. It records what `BENCHMARK-PLAN.MD` principles 4-6 and
sections 2, 3, 5, 12, and 13 require of the capture, and it preserves the unfavourable result
(principle 6) rather than hiding it. The tools: `scripts/run_benchmarks.sh` (single-session
runner with environment metadata), `scripts/capture_scaling_baseline.py` (the unattended
four-session controlled orchestrator), `python/review_scaling_baseline.py` (an independent
reviewer -- a separate implementation, not an import of the orchestrator), and
`python/plot_benchmarks.py` (chart, summary table, and `--compare` stability check).

## 1. Motivation: the two rejected monolithic captures

The first two attempts ran the whole scaling suite as one monolithic capture on the
development machine, an Apple M1 laptop (4 performance + 4 efficiency cores). Both were
rejected against the `BENCHMARK-PLAN.MD` requirement that a baseline reproduce across
separate sessions (section 3; `plot_benchmarks.py --compare` gate at 5%).

The symptom was specific. *Within* a session the repetitions were tight -- coefficient of
variation below 0.1% -- so a single session looked trustworthy. *Between* sessions the
multi-thread medians drifted 10-20%, while the single-thread medians stayed stable at 1-2%;
the within-session tightness hid the between-session drift entirely.

The cause was thermal accumulation across back-to-back all-core runs, compounded by a
variable background-load floor. Sustained multi-core work heats the die and its sustained
multi-core frequency falls, so every subsequent multi-thread median in that session is
measured against a slower clock. The single-thread cases barely load the chip and did not
drift -- exactly the fingerprint of a thermal, not an algorithmic, effect. Longer workloads
made it worse, giving the die more time to heat. Critically, `pmset` recorded **no** thermal
warning throughout: macOS does not expose the frequency-throttling that was happening, so
absence of a warning was not evidence of a stable clock (see section 7).

Both captures were rejected and are preserved as evidence, not published. They are the reason
the controlled protocol exists.

## 2. The controlled protocol

`scripts/capture_scaling_baseline.py` replaces the monolithic capture with a protocol that
controls conditions during every measurement and removes deterministic thermal bias. Its
properties (constants named for cross-reference; read the script for their exact values):

- **Isolated per-case processes with an excluded warm-up.** Each `(engine, threads)` case is
  a separate benchmark process. Google Benchmark's `min_warmup_time` (`WARMUP_TIME`) runs
  first and is excluded, so process startup, page faults, and cold caches never enter the
  reported repetitions (`REPS`).
- **A true Latin square over thread counts.** Four sessions rotate the thread-count rounds
  one step each (`{1,2,4,8}` -> `{2,4,8,1}` -> `{4,8,1,2}` -> `{8,1,2,4}`), so every thread
  count occupies every round position exactly once across the four sessions -- positional
  balance by construction, no statistical compensation needed. Engine order rotates
  independently.
- **A sustained pre-session soak.** Before a session begins, idle conditions must hold
  continuously for `SOAK_HOLD_S`: AC power, no `pmset` thermal warning, 1-minute load below
  `LOAD_CEIL` (2.0), busiest unrelated process below `CPU_CEIL` (25%), and aggregate
  unrelated CPU below `AGG_SOAK_CEIL` (60%).
- **An in-case aggregate-CPU ceiling calibrated from the soak.** From the session-0 soak
  samples the orchestrator computes `min(AGG_CEIL_CAP, median + max(AGG_CEIL_MARGIN,
  AGG_CEIL_MAD_K * MAD))` -- i.e. `min(70%, median + max(20pp, 3*MAD))` -- and **freezes** it
  for the whole attempt. MAD (median absolute deviation) is a robust dispersion measure. The
  hard cap sits below one fully-occupied logical core (~100% on macOS), so unrelated work
  cannot consume a whole core during 4/8-thread measurement on this 4P+4E chip.
- **Dense, tree-scoped monitoring.** Each case is sampled every `MONITOR_CADENCE_S` (0.25s),
  and a case yielding fewer than `MIN_CPU_SAMPLES` (8) is aborted as too short to trust.
  Intrusion is judged only against processes **outside the active benchmark process tree**,
  excluded by PID -- the launched PID and its descendants. A stale or accidentally concurrent
  benchmark (a *different* PID) is therefore treated as external interference, not exempted.
- **Condition-based cool-downs.** Between cases the orchestrator waits a minimum idle
  duration (longer after the heavier `threads >= 4` cases) and then re-confirms idle
  conditions, so every case starts from a comparable thermal and load state.
- **Acceptance by effect size, non-vacuously.** A baseline is `ACCEPTED` only if, across all
  four sessions: per-case cv below `CV_GATE` (3%); session-to-session median spread below
  `SPREAD_GATE` (5%); no runtime-vs-position dependence; no monotonic session slowdown; no
  material correlation *or* fitted slope between a case's runtime and the aggregate unrelated
  CPU recorded during it (effect size, not p-values, so a small sample cannot pass by being
  inconclusive); each session's soak median drifting from session 0 by no more than
  `min(10pp, max(5pp, 3*(MAD_0 + MAD_j)))`; and explicit guards so no gate passes vacuously
  (a missing case, a zero cv, or absent variation fails rather than passes).
- **Independent review before anything is committed.** On acceptance the four session records
  are written to a *review* directory, never to `results/`. `python/review_scaling_baseline.py`
  is a deliberately separate implementation that re-derives every metric from the raw Google
  Benchmark rows (CVs from the per-repetition timings, not from the benchmark's own cv
  aggregate). The orchestrator commits and publishes nothing on its own.

## 3. Attempt semantics

The waiter distinguishes *waiting* from *measuring*, because a machine that never reaches
controlled conditions has told us nothing about the engine:

- A **waiting-only soak timeout** -- controlled conditions never held, so no session began --
  is **not** an attempt. The waiter re-soaks and tries again.
- A **started session that voids** on interference (a soak that times out mid-attempt, or a
  case whose retries are exhausted) consumes **one of three** attempts (`MAX_REAL_ATTEMPTS`).
- A **completed capture that fails the numeric gates** likewise consumes one attempt.

The waiter stops on acceptance, on three consumed attempts, or on a six-hour horizon.

## 4. Outcome (2026-07-19): environment unsuitable during the observed window

The unattended waiter ran overnight on the M1 laptop under heavy, unambiguous background
interference -- a browser renderer at ~60% CPU, WebKit `WebContent` at 64-100%, and repeated
aggregate-ceiling breaches -- and produced no accepted baseline. That much the evidence
establishes plainly.

**What the evidence does *not* establish is that the formal three-attempt stopping rule
fired.** The waiter's own log reads, at face value, as six waiting-only voids and three
consumed attempts ending `UNSUITABLE_MOVE_TO_DEDICATED`. Forensic reconciliation of the run
directories and file mtimes proved that reading wrong: **two `capture_scaling_baseline.py`
instances ran concurrently** (a `~01:03` `nohup` instance and the `~01:09` instance that owns
the archived log), writing to the same log through a shared descriptor. Their lines
interleaved and clobbered one another -- which is why the void sequence skips `#5`, why five
run directories show a session began against three logged `real attempt` lines, and why the
per-line `real attempt N/3` accounting before 04:27:37 is unreliable. Reconstructed from
disk, only **two** attempts are independently valid. The authoritative per-run ledger, the
validity assessment, and this decision are in
`docs/evidence/benchmark-capture-20260719/FORENSIC-RECONCILIATION.md`.

Applying the stopping rule to the reconciled count (2 < 3): the rule did **not** fire. Per
the decision rule, fewer than three valid attempts means the accounting defect is fixed and
the capture **resumes locally** -- it does not yet move to a dedicated machine. The root
cause was the absence of single-instance protection, now fixed: the orchestrator takes an
exclusive `flock` and refuses to start a second instance, and every attempt-state transition
is written to an append-only ledger from which the counted-attempt total is re-derived and
cross-checked, so a two-process merge can neither happen nor be miscounted again
(`tests/capture_orchestrator_test.py` pins this). Raw evidence from the run -- the void logs,
the per-attempt interference records, the forensic ledger, and the two rejected monolithic
captures -- is archived under `docs/evidence/benchmark-capture-20260719/`.

The dedicated-machine escalation in section 6 still applies if three genuinely uncontaminated
local attempts fail; the headline multi-thread numbers (`BENCHMARK-PLAN.MD` section 5) must
come from a machine that can hold the controlled window. "Unchanged harness" is defined
precisely in section 6: the benchmark workloads and the measurement methodology are
preserved; the operational defects surfaced here (the missing lock, the fragile attempt
accounting, the hard-coded repository path, the inconsistent cool-down thresholds) are
corrected, not carried forward.

## 5. What remains valid: provisional single-thread evidence

The single-thread medians *were* stable across sessions on this machine (1-2% spread),
because a one-thread case barely loads the chip and so escapes the thermal drift that voided
the multi-thread cases. These are recorded as **provisional evidence only** -- not the Phase
13 baseline, which is defined to be the accepted four-session controlled capture from a
dedicated machine:

| Engine (harness id)       | threads | median  | workload                                   |
|---------------------------|--------:|--------:|--------------------------------------------|
| `european`                |       1 | ~298 ms | 1,000,000 paths                            |
| `heston`                  |       1 | ~107 ms | 100,000 paths x 50 steps                   |
| `barrier_bridge`          |       1 | ~259 ms | 200,000 paths                              |
| `asian_control_variate`   |       1 | ~258 ms | 200,000 paths x 12 steps (control variate) |
| `greeks_pathwise_delta`   |       1 | ~56 ms  | 1,000,000 paths                            |

These are useful as a sanity floor and as a rough single-thread reference; they carry no
speedup or efficiency, and they must not be presented as scaling results.

## 6. Dedicated-machine runbook

Run on an otherwise-idle dedicated machine, on AC power. Do **not** use shared CI runners for
headline numbers: their clocks, co-tenancy, and thermal behaviour are exactly the
uncontrolled conditions this methodology exists to exclude.

**What "unchanged harness" means.** Moving to a dedicated machine preserves the two things
that define the measurement: the **benchmark workloads** (the `(engine, threads)` cases, path
counts, step counts, variance-reduction settings, warm-up, and repetition counts) and the
**measurement methodology** (the soak gate, the frozen soak-calibrated ceiling, the Latin
square, the dense tree-scoped monitoring, the effect-size acceptance gates, and independent
review). It does **not** mean carrying operational defects forward: the single-instance lock,
the ledger-based attempt accounting, the repository-root and binary-path resolution, and the
cool-down thresholds below were corrected after the 2026-07-19 run (section 4) and are part of
the harness that runs everywhere from now on. The orchestrator resolves the repository root
from its own file location and reads the benchmark binary from `build/benchmark/...` by
default; override the binary with the `DW_BENCH_BIN` environment variable if the build tree
lives elsewhere.

1. **Clone at the pinned commit** so the capture is reproducible from its own record
   (`BENCHMARK-PLAN.MD` principle 5, section 2):

   ```
   git clone <repo> stochastic-lab && cd stochastic-lab
   git checkout <pinned-commit>
   ```

2. **Configure and build the release benchmark preset:**

   ```
   cmake --preset benchmark
   cmake --build build/benchmark --target dw_bench_monte_carlo_scaling
   ```

3. **Capture.** Either path is valid; the controlled orchestrator is required for a headline
   multi-thread baseline.

   - Single sessions (repeat with distinct labels, then compare). `run_benchmarks.sh`
     cools down before each session by default -- a 240s idle thermal soak
     (`DW_COOLDOWN_S`) followed by a capture-start gate that waits for 1-minute load below
     `DW_COOLDOWN_LOAD` (2.0, the same ceiling as the controlled soak gate). Set
     `DW_COOLDOWN_S=0` only if the machine is already known-quiet:
     ```
     scripts/run_benchmarks.sh session1
     scripts/run_benchmarks.sh session2
     python3 python/plot_benchmarks.py --compare \
         results/benchmarks/monte_carlo_scaling.session1.json \
         results/benchmarks/monte_carlo_scaling.session2.json
     ```
   - Full controlled four-session protocol (unattended; argument order is
     `<review-dir> <scratch-dir>`):
     ```
     python3 scripts/capture_scaling_baseline.py <review-dir> <scratch-dir>
     ```

4. **Independent acceptance review** of the controlled capture:

   ```
   python3 python/review_scaling_baseline.py <review-dir>
   ```

5. **Commit only on `REVIEW PASSED`.** Only then do the four session JSONs and the generated
   chart get committed:

   ```
   python3 python/plot_benchmarks.py <review-dir>/controlled_session0.json --outdir docs/figures
   ```

6. **Disclose the environment** per `BENCHMARK-PLAN.MD` section 2 -- CPU and core count,
   memory, OS, compiler and version, build type and flags, C++ standard, thread count, Git
   commit, seed, repetitions. The capture tools already inject CPU topology, power state, and
   thermal state into the result's `context`; record the dedicated machine's hardware and OS
   alongside it.

## 7. Limitations

- **The thermal probe is one-sided.** `pmset` reporting no thermal warning does **not** prove
  the absence of clock-frequency throttling; macOS does not expose frequency scaling, and the
  rejected captures drifted 10-20% with no warning ever recorded. The protocol never relies on
  the probe alone -- it also enforces minimum idle cool-down durations and the sustained soak.
  Treat the probe as able to confirm throttling, never to rule it out. (`THERMAL_PROBE_NOTE` in
  the orchestrator states the same caveat and is injected into each session's metadata.)
- **The provisional single-thread numbers are not a baseline.** They are a sanity floor from a
  machine that failed the controlled window; they carry no scaling claim.
- **Heterogeneous cores.** On a 4P+4E chip the fall in efficiency past the performance-core
  count is the slower efficiency cores being engaged, not a scaling defect, and it is not
  ordinary linear multicore scaling. `plot_benchmarks.py` annotates the chart accordingly when
  the topology is asymmetric; a homogeneous-CPU dedicated machine is preferable for a clean
  curve, and its topology must be disclosed either way.
- **No unsupported performance language.** Nothing here claims low latency, production-grade
  performance, or near-linear scaling (`BENCHMARK-PLAN.MD` section 15). The only claim is that
  the controlled capture, when it passes independent review on a dedicated machine, produces a
  reproducible measured baseline.

# Forensic reconciliation — overnight run of 2026-07-19

This document supersedes the per-line "attempt" accounting in `waiter.log` and the
`FINAL_VERDICT` file for the purpose of the M1-unsuitable stopping-rule decision. It is
built from disk evidence (run-directory names = run *start* times; `VERDICT` files;
directory and `session0/case_*.json` mtimes and sizes) cross-checked against `waiter.log`
only where the log is trustworthy. Where the log cannot be trusted (concurrent-write
clobbering, see below) the cell is marked **CLOBBERED** rather than guessed.

## What actually happened: two concurrent waiter instances

Two `capture_scaling_baseline.py` processes ran at once:

- **Chain A** — the `nohup` instance launched ~`01:03:13` (first dir `run-20260719-010313`).
  It was misclassified as terminated after a `pgrep` miss but was still alive.
- **Chain B** — launched ~`01:09:27` over the top of A (`run-20260719-010927`), its log fd
  truncated but never detached.

Proof this is two processes, not one, is on disk alone and needs no log: `main()`'s loop
creates each run dir (`datetime.now()`, `scripts/capture_scaling_baseline.py:673`) at the
instant the previous run's `VERDICT` is written (the dir mtime), so within one process the
`[start → verdict-mtime]` intervals can never overlap. Two of them do:

```
run-20260719-010313   start 01:03:13 → verdict 01:33:32
run-20260719-010927   start 01:09:27 → verdict 01:39:45   (starts inside 010313's interval)
```

Both processes redirected stdout to the same `waiter.log` through a shared, non-`O_APPEND`
file descriptor, so their `print(..., flush=True)` lines (`log()`,
`scripts/capture_scaling_baseline.py:110-111`) interleaved and frequently overwrote one
another. The physical smoking gun is **`waiter.log:895`**, where one process's truncated
`03:59:31 soak[cooldown-retry-` is fused with the other's `03:59:52 soak[session0]: need
120s…`. Chain B exited first (three attempts reached at `04:27:37`); Chain A wrote the
surviving log tail and the `FINAL_VERDICT` at `05:25:13`.

## 1. The authoritative ledger

One row per run directory of both chains, in run-start order. `run-20260719-041937` was
created in the same second by both chains (`mkdir(..., exist_ok=True)`) and is a single
shared directory; it is listed once. Log line numbers are in `waiter.log`.

| Run ID (chain) | Soak completed | Session began | First case began | Case completed | Retry-exhaustion cause | Final verdict | Counts as attempt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run-010313 (A) | No — TIMEOUT ~01:33:32 (CLOBBERED) | No | No | No | — | waiting only | No — A waiting-void #1 (CLOBBERED) |
| run-010927 (B) | No — TIMEOUT 01:39:45 (L185) | No | No | No | — | waiting only | No — void #1 (L186) |
| run-013832 (A) | No — TIMEOUT ~02:08:52 (CLOBBERED) | No | No | No | — | waiting only | No — A waiting-void #2 (CLOBBERED) |
| run-014445 (B) | No — TIMEOUT 02:15:05 (L369) | No | No | No | — | waiting only | No — void #2 (L370) |
| run-021352 (A) | No — TIMEOUT ~02:44:12 (CLOBBERED) | No | No | No | — | waiting only | No — A waiting-void #3 (CLOBBERED) |
| run-022005 (B) | No — TIMEOUT 02:50:24 (L553) | No | No | No | — | waiting only | No — void #3 (L554) |
| run-024912 (A) | No — TIMEOUT ~03:19:32 (CLOBBERED) | No | No | No | — | waiting only | No — A waiting-void #4 (CLOBBERED) |
| run-025524 (B) | No — TIMEOUT 03:25:44 (L737) | No | No | No | — | waiting only | No — void #4 (L738) |
| **run-032432 (A)** | No — TIMEOUT ~03:54:52 (CLOBBERED) | No | No | No | — | waiting only | **No — A waiting-void #5 (CLOBBERED) ← the "missing #5"** |
| run-033044 (B) | Yes — held 03:56:03, calib 42% (L873-874) | Yes (L875) | Yes — european/1 abort 03:56:05 (L876) | Yes — european/1 (9881 B, 04:06:32) + asian_cv/1 (10298 B, 04:08:30) | heston/1 exhausted ~04:11-04:12 (heston json 1157 B, 04:11:33); abort text CLOBBERED, **necessarily non-sister** (A soaking, no benchmark, 03:59:52-04:15:50) | session began (04:12:38) | Yes — B attempt 1 (log line CLOBBERED) |
| run-035952 (A) | Yes — held 04:15:50, calib 35% (L943-944) | Yes | Yes — european/1 abort 04:15:51 | No — european/1 0 B (04:18:22) | european/1 exhausted 04:19:37 (L959); aborts on **sister** dw_bench 64/55/53% (L945/949/954) | session began (04:19:37) | Yes — A attempt 1 (real attempt 1/3, L960) |
| run-041238 (B) | Yes — calibration CLOBBERED (session0/ present, dir 04:15) | Yes | Yes — european/1 0 B (04:18:22) | No | european/1 exhausted 04:19:37; cause CLOBBERED; **sister-contaminated** (A035952 running cases 04:15:50-04:19:37) | session began (04:19:37) | Yes — B attempt 2 (log line CLOBBERED) |
| run-041937 (A+B, shared) | Yes — calib 42% 04:24:00 (L971) | Yes | Yes — european/1 abort 04:24:00 | No — european/1 0 B (04:26:11) | european/1 exhausted 04:27:37 (L986); aborts agg 76% (L973) + **sister** dw_bench 65/59% (L977/981) | session began (04:27:37) | Yes — A attempt 2 (real attempt 2/3, L987) **and** B attempt 3 (CLOBBERED) |
| run-042737 (A) | No — TIMEOUT 04:58:00 (L1145) | No | No | No | — | waiting only | No — void #6 (L1146) |
| run-050300 (A) | Yes — held 05:18:06, calib 36% (L1212) | Yes | Yes — european/1 abort 05:18:06 (L1214) | Yes — european/1 (9882 B, 05:19:30; L1219) | asian_cv/1 exhausted 05:25:13 (L1242); aborts agg 40/44% (L1224/1228) + WebKit 100% (L1238); **no sister** (B exited 04:27:37) | session began (05:25:13); wrote FINAL_VERDICT + UNSUITABLE (L1244) | Yes — A attempt 3 (real attempt 3/3, L1243) |

Notes on the ledger:
- Chain A's true waiting-void counter ran `#1…#6` (rows 010313, 013832, 021352, 024912,
  032432, 042737). Only `#6` survived in the log — after `04:27:37` Chain B had exited and
  A was the sole writer, so the tail is clean. Chain B's counter ran `#1…#4` (010927,
  014445, 022005, 025524) and those four survived.
- The log's void line `#5` never appears because Chain B never had a fifth waiting void
  and Chain A's real `#5` (run-032432, `~03:54:52`) was overwritten by a concurrent write.
- `session0/case_*.json` sizes are ground truth per directory: a completed case is a full
  JSON (~9.8-10.3 kB), an aborted-before-first-result case is 0 bytes, `heston/1` in
  run-033044 is a 1157-byte partial.
- Older no-verdict dirs (`run-20260718-*`, `run-20260719-005054`) predate both chains and
  are out of scope here.

## 2. The four explanations

**(a) Void numbering skipped #5.** Not a counter skip — `main()` increments and logs
adjacently (`:689-690`), so `waiting_voids` cannot advance silently. The surviving void
lines are two independent counters spliced by clobbering: Chain B's `#1-#4` (times match
010927/014445/022005/025524 mtimes) then Chain A's `#6` (matches 042737's 04:58:00).
Chain A's genuine `#5` — run-032432, soak timeout `~03:54:52` — had its log line
overwritten by a concurrent write, the same mechanism visible at `waiter.log:895`.

**(b) Five "session began" dirs vs three counted attempts.** Two independent counters, one
contaminated final tally. Each instance ran its own three `void_started` attempts (B:
033044, 041238, 041937; A: 035952, 041937, 050300 — colliding on the shared 041937, so 5
distinct dirs, 6 logical attempts). The `FINAL_VERDICT`'s "3 controlled attempts" is Chain
A's `attempts` counter only, oblivious to Chain B. The `if not sessions` branch
(`:628`) is correct and never misfired: every "session began" verdict came from the
`run_session`-failure path (`:652`), reached only after soak passed and a ceiling was
calibrated, so each such dir *did* calibrate — B's run-041238 calibration line was simply
clobbered, which is why the log shows four calibrations for five dirs. No run reached
session index ≥ 1 (`inter-session idle` count = 0), so no
`…(session began, later soak timed out)` verdict was ever written.

**(c) run-20260719-042737 tagged "(waiting only)".** Benign and correctly tagged. The dir
name is the run's *start* second. At `04:27:37` Chain A logged its attempt 2/3 for the
shared run-041937 and, in the same second, the next loop iteration (`:673`) created
run-042737; that run then soaked 30 min, timed out `04:58:00` (L1145), and became
waiting-only void `#6` (L1146). It is a distinct run from attempt 2, so the tag is right.

**(d) The lingering `dw_bench_monte_carlo_scaling` process.** `active_tree` /
`measurement_exclude` (`:147-183`) worked exactly as designed. They exclude only the
excluding instance's own process tree (`os.getpid()` plus the benchmark PID it launched),
so a benchmark belonging to the *other* instance is a genuinely different PID and is
correctly counted as unrelated interference (the docstring intends precisely this,
`:15-18`). The `unrelated dw_bench…` aborts (L945/949/954/977/981) and soak resets
(L926/928/929) were real contention from a real concurrent process — not a
tree-classification bug. The defect is upstream: nothing stopped a second instance from
existing. **The bug is the missing singleton lock, not the classifier.** (This also
corrects the README's earlier reading of those aborts as one instance's own unreaped
residue.)

## 3. Independent-validity assessment

For each started-and-voided session, the test is: is the **entire** void path — every
abort and any cool-down failure that consumed its retries — attributable to genuinely
external causes (browser/system processes, load, aggregate breaches from non-sister
sources), with **no** sister-benchmark contribution? The discriminator is which chain was
*running cases* (a live `dw_bench` PID) vs merely *soaking/sleeping* (no benchmark) at
each moment.

Chain-A activity windows (for judging Chain B) and vice-versa:
- Chain A ran **no** benchmark from `03:24:32` through `04:15:50` (soak 032432 to timeout
  03:54:52, then 300 s sleep, then soak 035952 to 04:15:50). It ran cases only in
  035952 (04:15:50-04:19:37), 041937 (04:24:00-04:27:37), and 050300 (05:18-05:25).
- Chain B ran cases in 033044 (03:56-04:11), 041238 (~04:15-04:19), 041937 (04:19-04:27),
  then exited. No Chain B benchmark existed after `04:27:37`.

| Session | Verdict-determining exhaustion | Sister running then? | Assessment |
| --- | --- | --- | --- |
| **B5 / run-033044** | heston/1 exhausted ~04:11-04:12 (also european/asian aborts 03:56-04:08) | **No** — A soaking/sleeping 03:24:32-04:15:50; the `dw_bench` at L926/928/929 is 033044's *own* benchmark | **VALID.** Entire path external. Exact heston abort text is CLOBBERED but is necessarily non-sister; corroborated by A's soak only ever seeing 033044's own benchmark. |
| **A9 / run-050300** | asian_cv/1 exhausted 05:25:13 (agg 40/44%, WebKit 64→100%) | **No** — B exited 04:27:37 | **VALID.** WebKit + non-sister aggregate breaches; european/1 even completed within ceiling. |
| A6 / run-035952 | european/1 exhausted 04:19:37 | **Yes** — B041238 running european/1 | CONTAMINATED (sister dw_bench 64/55/53%). |
| B6 / run-041238 | european/1 exhausted 04:19:37 | **Yes** — A035952 running european/1 | CONTAMINATED (mutual). |
| A7 = B7 / run-041937 | european/1 exhausted 04:27:37 | **Yes** — the two chains' benchmarks | CONTAMINATED (sister dw_bench 65/59%; agg 76% partly sister). |

Provisional assessment confirmed with one correction of emphasis: **B5/run-033044 is not
merely "likely" — it is VALID**, because Chain A demonstrably ran no benchmark anywhere in
033044's case window, so the heston exhaustion (clobbered though its text is) cannot have a
sister component.

**Final count of independently valid started attempts: 2** (B5/run-033044 and
A9/run-050300). Three started attempts are sister-contaminated (035952, 041238, 041937).

## 4. Decision-rule application

Owner's rule: **≥ 3** independently valid started attempts failed → the M1-unsuitable
stopping rule stands; **< 3** → the rule has **not** fired; unreconstructable →
indeterminate.

The run is fully reconstructable, and the count is **2**. Therefore:

> **Branch: fewer than 3 → the stopping rule has NOT fired.**

Consequence: the accounting/concurrency defect must be fixed (section 5), and **local
capture resumes** under a corrected, single-instance harness until the rule is genuinely
met (three independently valid, uncontaminated started attempts fail).

Two statements to record plainly:

- **The `FINAL_VERDICT` file's "3 controlled attempts" text is superseded by this
  reconciliation.** It was emitted by Chain A's `main()` at `05:25:13`, whose three
  counted attempts were 035952, 041937, and 050300 — two of which (035952, 041937) are
  sister-contaminated. The literal file is left in place as an artifact; its count is not
  the decision input.
- **Separately, the observed window remains strong evidence of environmental hostility.**
  Across both chains there were **10 waiting-only voids** (A: 6, including the clobbered
  #5; B: 4), i.e. 10 genuine 1800-second soak timeouts, every one driven by browser /
  system-daemon / load activity independent of the concurrency defect (e.g.
  `Brave Browser Helper` 60-107%, `com.apple.WebKit.WebContent` 36-159%, `corespotlightd`
  201%, `biomesyncd`, sustained load 2-6 vs `LOAD_CEIL 2.0`). The machine is clearly
  unsuitable for unattended capture; the formal three-attempt rule simply did not fire on
  this window once contamination is removed, so the *stopping decision* is deferred, not
  the *observation* of hostility.

## 5. Defect and fix

**Primary defect:** no singleton-instance guard in `main()`, allowing a second
orchestrator to run concurrently. Secondary, all consequences of the same gap: a shared
non-`O_APPEND` log fd (line clobbering) and second-granularity run-dir names that collide
when two runs start in the same second (run-041937).

**Minimal fixes** (being applied under the owner's amended policy — workloads and the
measurement methodology preserved unchanged; only operational defects fixed):

- Take a non-blocking exclusive lock at the top of `main()` (e.g. `fcntl.flock` on
  `SCRATCH/.waiter.lock`, `LOCK_EX | LOCK_NB`); on contention, log and exit rather than
  run a second instance. ~5 lines, no change to the single-instance path.
- Open the log with `O_APPEND` per instance (defensive against any future overlap).
- Include `os.getpid()` (or a monotonic counter) in the run-dir name at `:673` to make
  same-second collisions impossible.

None of these touch the soak gates, the ceiling calibration, the Latin-square schedule,
the case set, or any acceptance gate; the frozen methodology is unchanged.

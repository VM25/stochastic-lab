# Benchmark-capture evidence — overnight run of 2026-07-19

**This directory is methodological evidence, not performance results.** Nothing here is
an accepted benchmark number, a latency figure, or a scaling curve. It is the record of
an unattended attempt to capture a multithread Monte Carlo scaling baseline on this
laptop failing under uncontrolled background load, and the decision that followed from
that failure. It is preserved because `docs/BENCHMARK-PLAN.MD` principle 6 requires it:
*"Preserve unfavorable results."* An unaccepted capture is data about the measurement
environment; discarding it would hide the reason the baseline was moved elsewhere.

Do not read any timing value that appears in `waiter.log` or in the rejected JSONs as a
result. Every controlled attempt here was voided; the two monolithic JSONs were
explicitly rejected before this run even began.

## What each file is

| File | What it is |
| --- | --- |
| `waiter.log` | Verbatim copy of the final waiter's full log (1244 lines). The authoritative timeline; the run of the persistent 4-session Latin-square controlled waiter, from `01:09:27` to the final verdict at `05:25:13`. |
| `FINAL_VERDICT` | The one-line machine verdict written at the end of the run. |
| `attempt-verdicts.txt` | One line per `run-*` directory that the orchestrator created, with that directory's `VERDICT` tag (or `(no VERDICT file)` where the directory held no verdict). |
| `rejected-monolithic/rejected.baseline.json` | First rejected monolithic capture (142335 bytes). Validated as parseable JSON. |
| `rejected-monolithic/rejected.session2.json` | Second rejected monolithic capture (142376 bytes). Validated as parseable JSON. |

The two rejected JSONs both carry `"num_cpus": 8`, `"cpu_brand": "Apple M1"`
(`logical=8, performance=4, efficiency=4`), `"build_type": "Release"`, and
`git_commit b716a42`. Each holds 280 benchmark records. They are the two "monolithic"
single-session captures that were rejected before the hardened orchestrator was written:
their own `context` blocks record the environment they were taken in — for
`rejected.baseline.json`, `dw_load_avg_pre 1.65` rising to `dw_load_avg_post 5.29`; for
`rejected.session2.json`, `dw_load_avg_pre 2.93` rising to `dw_load_avg_post 6.10`. That
session-to-session load difference is the drift these captures were rejected for:
multi-thread medians moved 10–20% between sessions from thermal accumulation and
background load, while within-session cv stayed under 0.1%. They are kept as the
"before" that motivated the controlled protocol, not as candidates for acceptance.

## Machine and harness

- Apple M1 laptop, 8 logical cores (4 performance + 4 efficiency), macOS `Darwin 25.5.0`.
- The harness is `scripts/capture_scaling_baseline.py`, moved to this run **unchanged**.
- The waiter's own note on thermal state, quoted verbatim from `waiter.log`:
  > `01:09:27 note: pmset reports no OS-recorded thermal warning; this does NOT prove the absence of clock-frequency throttling, which macOS does not expose. Cool-downs enforce a minimum idle duration in addition to this probe.`
- A "soak" gate must hold before any case runs:
  > `soak[session0]: need 120s of AC + no thermal warning + load<2.0 + max-proc<25.0% + aggregate<60.0%`

## Timeline

Derived from `waiter.log`. Line numbers refer to that file.

> **Superseded in part — read with `FORENSIC-RECONCILIATION.md`.** This timeline reads
> the log at face value, as first archived. The forensic reconciliation later proved the
> log is a two-process merge (two concurrent waiter instances) and supersedes both the
> attempt accounting below and the attempt-1 interpretation: the "lingering
> just-terminated benchmark process" was in fact the **concurrent sister instance's
> live benchmark**, and the authoritative per-run ledger, validity assessment, and
> stopping-rule decision live in `FORENSIC-RECONCILIATION.md`. The narrative below is
> preserved unaltered as the original reading of the evidence.

### Waiting-only voids are uncounted

A soak that never cleared long enough to start a session is a *waiting-only* void and is
explicitly **not** counted as one of the three attempts. The log records these verbatim
(the sequence skips `#5` — see "Surprises" below):

```
186:  01:39:45 waiting-only void #1 (not an attempt); re-soaking in 300s
370:  02:15:05 waiting-only void #2 (not an attempt); re-soaking in 300s
554:  02:50:24 waiting-only void #3 (not an attempt); re-soaking in 300s
738:  03:25:44 waiting-only void #4 (not an attempt); re-soaking in 300s
1146: 04:58:00 waiting-only void #6 (not an attempt); re-soaking in 300s
```

### The three started attempts

An attempt is counted only when a session actually began — soak cleared, the in-case
aggregate-CPU ceiling was calibrated and frozen, the case order was planned, and the
first case ran — and then all of `european/1`'s (or a later case's) retries were
exhausted, yielding `void_started`. That happened exactly three times:

```
960:  04:19:37 real attempt 1/3 outcome: VOID_STARTED
987:  04:27:37 real attempt 2/3 outcome: VOID_STARTED
1243: 05:25:13 real attempt 3/3 outcome: VOID_STARTED
```

**Attempt 1 (04:19:37).** Session began at `04:15:50` with
`calibrated in-case aggregate ceiling: 35% (session-0 soak median 15%, MAD 3.8%, cap 70%)`.
All three retries of the first case aborted on a single lingering process:

```
945: 04:15:51     ABORT european/1: unrelated dw_bench_monte_carlo_scaling at 64% during the case
949: 04:16:56     ABORT european/1: unrelated dw_bench_monte_carlo_scaling at 55% during the case
954: 04:18:22     ABORT european/1: unrelated dw_bench_monte_carlo_scaling at 53% during the case
959: 04:19:37 session0: european/1 retries exhausted -> void_started
```

That `dw_bench_monte_carlo_scaling` process was a **benchmark process left over from a
just-terminated prior retry**, not competing third-party software. It is visible winding
down in the soak resets a few minutes earlier:

```
926: 04:05:14 soak[session0]: reset (dw_bench_monte_carlo_scaling at 95%)
928: 04:06:25 soak[session0]: reset (dw_bench_monte_carlo_scaling at 97%)
929: 04:07:05 soak[session0]: reset (dw_bench_monte_carlo_scaling at 29%)
```

The harness excludes only its **own active process tree** from the "unrelated" CPU total,
so a stray benchmark process that had not yet been reaped counted as unrelated load and
tripped the per-process abort. This is a **conservative abort by design**: the harness
would rather void a session than risk measuring against a process it cannot prove is
idle. Attempt 1 should therefore be read with that caveat — its aborts flag the
harness's own residue, not necessarily external interference.

**Even discounting attempt 1, the other two attempts show unambiguous external
interference.**

**Attempt 2 (04:27:37).** Session began at `04:24:00` with
`calibrated in-case aggregate ceiling: 42% (session-0 soak median 22%, MAD 5.8%, cap 70%)`.
Its first case was aborted by an aggregate-CPU-ceiling breach from unrelated processes,
well above the frozen ceiling:

```
973: 04:24:00     ABORT european/1: aggregate unrelated 76% >= ceiling 42% during the case
977: 04:25:06     ABORT european/1: unrelated dw_bench_monte_carlo_scaling at 65% during the case
981: 04:26:11     ABORT european/1: unrelated dw_bench_monte_carlo_scaling at 59% during the case
986: 04:27:37 session0: european/1 retries exhausted -> void_started
```

The first-case abort (`aggregate unrelated 76% >= ceiling 42%`) is genuine external load;
the two retries then also caught the lingering benchmark process.

**Attempt 3 (05:25:13).** Session began at `05:18:06` with
`calibrated in-case aggregate ceiling: 36% (session-0 soak median 16%, MAD 3.0%, cap 70%)`.
This attempt progressed furthest — `european/1` actually completed within the ceiling —
before the next case was voided by browser activity and further aggregate breaches:

```
1214: 05:18:06     ABORT european/1: unrelated com.apple.WebKit.WebContent at 64% during the case
1219: 05:19:30     european/1  pos  0  median 59.504 ms  agg-cpu mean 8% max 26% n=27 dur=8.1s (ceiling 36%)
1224: 05:20:34     ABORT asian_control_variate/1: aggregate unrelated 40% >= ceiling 36% during the case
1228: 05:21:41     ABORT asian_control_variate/1: aggregate unrelated 44% >= ceiling 36% during the case
1238: 05:24:07     ABORT asian_control_variate/1: unrelated com.apple.WebKit.WebContent at 100% during the case
1242: 05:25:13 session0: asian_control_variate/1 retries exhausted -> void_started
```

`com.apple.WebKit.WebContent` at 64% and then 100% is browser rendering load — external
to the harness. (The single `european/1` line at `05:19:30` is not an accepted result;
its session was voided moments later.) Earlier attempts also saw
`Brave Browser Helper (Renderer)` interference, e.g. `876: 03:56:05 ABORT european/1:
unrelated Brave Browser Helper (Renderer) at 60% during the case`.

### Final verdict

The three-attempt stopping rule fired. The last line of the log is the verdict, which is
also copied to `FINAL_VERDICT`:

```
1244: 05:25:13 UNSUITABLE_MOVE_TO_DEDICATED: 3 controlled attempts did not produce an
      accepted multi-thread baseline; move the harness UNCHANGED to a dedicated fixed machine.
```

The harness worked as intended: it refused to emit a baseline it could not defend, and
recommended moving the measurement — unchanged — to a dedicated machine. No baseline
number was produced or should be inferred from this directory.

## Surprises / notes on the sources

- The literal string `FINAL_VERDICT` does not appear inside `waiter.log`; the verdict is
  written as the log's final line (1244) in its full text form, and separately to the
  `FINAL_VERDICT` file. The two texts match.
- The waiting-only void sequence in the log skips `#5` (it runs `#1, #2, #3, #4, #6`). The
  source does not explain the gap; it is recorded here as-is, not corrected.
- More `run-*` directories are tagged `(session began)` (5 of them) than the log's three
  counted `real attempt` records. The directory `VERDICT` tags are coarser than the log's
  attempt accounting: a session "begins" (ceiling calibrated + order planned + first case
  run) more often than it reaches the retries-exhausted `void_started` state that
  increments the counted-attempt tally. Several early `run-*` directories carry no
  `VERDICT` file at all; those are listed as `(no VERDICT file)` in
  `attempt-verdicts.txt`. Treat `waiter.log`'s `real attempt N/3` lines as authoritative
  for the count of started attempts.
- **Update (fully reconciled):** all three anomalies above are now explained in
  `FORENSIC-RECONCILIATION.md`. Two `capture_scaling_baseline.py` instances ran
  concurrently (a `~01:03` `nohup` instance and the `~01:09` instance that owns this log),
  both writing to this file through a shared descriptor, so their lines interleaved and
  clobbered one another — that is why void `#5` is absent (a Chain-A void whose line was
  overwritten), why there are five `(session began)` dirs against three logged attempts
  (two independent attempt counters), and it is visible directly at line 895. Consequently
  the per-line `real attempt N/3` accounting *before* `04:27:37` is unreliable and does
  **not** supersede the disk evidence; the reconciliation ledger does. The tail after
  `04:27:37` (the second instance having exited) is clean.

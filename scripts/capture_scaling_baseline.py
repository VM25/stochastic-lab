#!/usr/bin/env python3
"""Unattended, condition-controlled Monte Carlo scaling baseline capture.

Monolithic captures drifted 10-20% at multi-thread from thermal accumulation and a
variable background-load floor; within-session repetitions hid it. This orchestrator
controls conditions during every measurement and removes deterministic thermal bias.

Design (numbered to the methodology requirements, latest revision):

1. Four sessions with a true Latin square over the four thread counts {1,2,4,8}: the
   thread-count rounds rotate one step per session (A:1,2,4,8 B:2,4,8,1 C:4,8,1,2
   D:8,1,2,4), so every thread count occupies every round position exactly once across
   the four sessions -- complete positional balance, no statistical compensation needed.
   The engine order rotates independently.
2. In-case intrusion is judged only by processes *outside the active benchmark process
   tree*: the specific launched PID and its descendants are excluded, by PID -- not by
   name. A stale or accidentally concurrent benchmark process (a different PID) is
   therefore external interference and aborts the case.
3. Both the busiest unrelated process and the *aggregate* unrelated CPU are monitored and
   recorded, together with power state and OS thermal state, so several moderate
   background processes that together contend for cores are caught, not just one big one.
4. Every isolated case runs an excluded warm-up (Google Benchmark's min_warmup_time), so
   process startup, page faults, and cold caches do not enter the reported repetitions.
5. A baseline is accepted only if, across all four sessions: per-case cv below threshold;
   session-to-session median spread below 5%; no significant runtime-vs-position
   relationship; no monotonic session slowdown; no material correlation between a case's
   runtime and the aggregate unrelated CPU recorded during it; and consistent workload and
   status metadata -- with explicit guards so no gate can pass vacuously (a missing case,
   a zero cv, or absent variation fails rather than passes).

Persistence and attempts: the waiter re-soaks after a waiting-only void (soak never
reached controlled conditions -- no session began, not an attempt); a completed rejection
or a session that began then voided spends one of three attempts, the distinction
preserved; it stops on acceptance, three attempts, or a six-hour horizon.

On acceptance the four session records are written to a *review* directory, not to
results/ -- the human operator inspects the raw metadata, re-confirms the comparison
independently, checks for vacuous passes, and only then commits. This script commits
nothing and publishes nothing on its own.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path("/Users/vatsal/Documents/stochastic-lab")
BIN = REPO / "build/benchmark/benchmarks/dw_bench_monte_carlo_scaling"
REVIEW_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / ".capture_review"
SCRATCH = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / ".capture_scratch"

ENGINES = ["european", "asian_control_variate", "heston", "greeks_pathwise_delta", "barrier_bridge"]
THREADS = [1, 2, 4, 8]

SESSIONS = 4             # a true Latin square over four thread counts needs four sessions
REPS = 12
MIN_TIME = "0.4s"
WARMUP_TIME = "0.15"     # excluded warm-up per case, in seconds (point 4; a bare double)
LOAD_CEIL = 2.0          # 1-minute load ceiling, used only when the benchmark is idle
CPU_CEIL = 25.0          # busiest unrelated process %CPU
# The in-case aggregate-unrelated-CPU ceiling is calibrated from the pre-run soak, not
# fixed: ceiling = min(AGG_CEIL_CAP, soak_median + max(AGG_CEIL_MARGIN, k*MAD)). On macOS
# ~100% is one fully occupied logical core, so a hard cap well below that keeps unrelated
# work from consuming a whole core during 4/8-thread measurement on this 4P+4E chip.
AGG_SOAK_CEIL = 60.0     # the soak establishes a clean baseline below this aggregate
AGG_CEIL_CAP = 70.0      # the calibrated in-case ceiling is hard-capped here (< one core)
AGG_CEIL_MARGIN = 20.0   # minimum margin above the soak median
AGG_CEIL_MAD_K = 3.0     # MAD multiplier for the margin
SOAK_HOLD_S = 120
SOAK_MAX_WAIT_S = 1800
INTER_SESSION_IDLE_S = 600
COOLDOWN_HIGH_MIN_S = 45
COOLDOWN_LOW_MIN_S = 10
COOLDOWN_RESTORE_MAX_S = 900
MAX_CASE_RETRIES = 3
MAX_REAL_ATTEMPTS = 3
WAITING_RETRY_S = 300
TOTAL_HORIZON_S = 6 * 3600

SPREAD_GATE = 0.05
CV_GATE = 0.03
POSITION_CORR_GATE = 0.30
SESSION_DRIFT_GATE = 0.03
AGG_CORR_GATE = 0.30      # effect-size threshold on |Pearson(agg cpu, residual)|
AGG_SLOPE_EFFECT_GATE = 0.02  # reject if the fitted slope implies >2% runtime over the CPU range

THERMAL_PROBE_NOTE = (
    "pmset reports no OS-recorded thermal warning; this does NOT prove the absence of "
    "clock-frequency throttling, which macOS does not expose. Cool-downs enforce a "
    "minimum idle duration in addition to this probe."
)


def log(message: str) -> None:
    print(f"{datetime.datetime.now():%H:%M:%S} {message}", flush=True)


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def load1() -> float:
    match = re.search(r"load averages?: *([0-9.]+)", _run(["uptime"]))
    return float(match.group(1)) if match else 99.0


def power_is_ac() -> bool:
    return "AC Power" in _run(["pmset", "-g", "ps"])


def thermal_warning() -> bool:
    """True if macOS records an active thermal throttle. Absence is NOT proof of no
    frequency scaling (THERMAL_PROBE_NOTE)."""
    match = re.search(r"CPU_Speed_Limit *= *([0-9]+)", _run(["pmset", "-g", "therm"]))
    return bool(match) and int(match.group(1)) < 100


def process_table() -> list[tuple[int, int, float, str]]:
    rows = []
    for line in _run(["ps", "-Ao", "pid,ppid,pcpu,comm"]).splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2]), parts[3]))
        except ValueError:
            continue
    return rows


def active_tree(root_pid: int, table: list) -> set[int]:
    """The launched benchmark PID and all its descendants (point 2): only *these* are
    excluded from intrusion detection, so a stale/concurrent benchmark (different PID) is
    treated as external interference."""
    children: dict[int, list[int]] = {}
    for pid, ppid, _, _ in table:
        children.setdefault(ppid, []).append(pid)
    tree, stack = {root_pid}, [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree


def unrelated_stats(exclude: set[int], table: list) -> tuple[float, str, float]:
    """(max unrelated %CPU, its comm, aggregate unrelated %CPU) over processes not in the
    excluded set (point 3)."""
    max_cpu, max_comm, aggregate = 0.0, "", 0.0
    for pid, _, pcpu, comm in table:
        if pid in exclude:
            continue
        aggregate += pcpu
        if pcpu > max_cpu:
            max_cpu, max_comm = pcpu, comm
    return max_cpu, max_comm, aggregate


def idle_conditions() -> tuple[bool, str]:
    """Conditions when the benchmark is NOT running."""
    if not power_is_ac():
        return False, "not on AC power"
    if thermal_warning():
        return False, "pmset thermal throttle"
    current = load1()
    if current >= LOAD_CEIL:
        return False, f"load {current:.2f} >= {LOAD_CEIL}"
    max_cpu, comm, aggregate = unrelated_stats(set(), process_table())
    if max_cpu >= CPU_CEIL:
        return False, f"{comm.split('/')[-1]} at {max_cpu:.0f}%"
    if aggregate >= AGG_SOAK_CEIL:
        return False, f"aggregate unrelated {aggregate:.0f}%"
    return True, "ok"


def soak(hold_s: int, label: str, max_wait_s: int) -> tuple[bool, list[float]]:
    """Wait until idle conditions hold continuously for hold_s. Returns (ok, samples),
    where samples are the aggregate-unrelated-CPU readings across the successful hold
    streak -- used to calibrate the in-case aggregate ceiling."""
    log(f"soak[{label}]: need {hold_s}s of AC + no thermal warning + load<{LOAD_CEIL} + "
        f"max-proc<{CPU_CEIL}% + aggregate<{AGG_SOAK_CEIL}%")
    held = waited = 0
    samples: list[float] = []
    while held < hold_s:
        ok, detail = idle_conditions()
        if ok:
            held += 10
            _, _, aggregate = unrelated_stats(set(), process_table())
            samples.append(aggregate)
        else:
            held = 0
            samples.clear()
            log(f"soak[{label}]: reset ({detail})")
        time.sleep(10)
        waited += 10
        if waited > max_wait_s:
            log(f"soak[{label}]: TIMEOUT after {max_wait_s}s")
            return False, []
    log(f"soak[{label}]: held {hold_s}s")
    return True, samples


def cool_down(minimum_s: int, label: str) -> bool:
    time.sleep(minimum_s)
    ok, _ = soak(20, f"cooldown-{label}", COOLDOWN_RESTORE_MAX_S)
    return ok


def _median(xs: list[float]) -> float:
    ordered = sorted(xs)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def derive_ceiling(samples: list[float]) -> tuple[float, float, float]:
    """Calibrate the in-case aggregate ceiling from the soak samples (points 1-4):
    median + max(margin, k * MAD), capped. MAD is the median absolute deviation, a robust
    dispersion measure. Returns (median, mad, ceiling), frozen for the whole attempt."""
    if not samples:
        return 0.0, 0.0, AGG_CEIL_CAP
    median = _median(samples)
    mad = _median([abs(s - median) for s in samples])
    ceiling = min(AGG_CEIL_CAP, median + max(AGG_CEIL_MARGIN, AGG_CEIL_MAD_K * mad))
    return median, mad, ceiling


def run_case(
    engine: str, threads: int, out_path: pathlib.Path, agg_ceiling: float
) -> tuple[dict, float, float, float] | None:
    """Run one isolated, warmed-up case; monitor unrelated interference throughout against
    the frozen aggregate ceiling (points 2-4). Returns (record, max aggregate unrelated
    %CPU, mean aggregate unrelated %CPU, max single unrelated %CPU), or None if aborted."""
    proc = subprocess.Popen(
        [
            str(BIN),
            f"--benchmark_filter=^{engine}/{threads}/real_time$",
            f"--benchmark_repetitions={REPS}",
            f"--benchmark_min_time={MIN_TIME}",
            f"--benchmark_min_warmup_time={WARMUP_TIME}",
            "--benchmark_display_aggregates_only=true",
            f"--benchmark_out={out_path}",
            "--benchmark_out_format=json",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    agg_samples: list[float] = []
    peak_max = 0.0
    while proc.poll() is None:
        table = process_table()
        exclude = active_tree(proc.pid, table)
        max_cpu, comm, aggregate = unrelated_stats(exclude, table)
        agg_samples.append(aggregate)
        peak_max = max(peak_max, max_cpu)
        reason = None
        if max_cpu >= CPU_CEIL:
            reason = f"unrelated {comm.split('/')[-1]} at {max_cpu:.0f}%"
        elif aggregate >= agg_ceiling:
            reason = f"aggregate unrelated {aggregate:.0f}% >= ceiling {agg_ceiling:.0f}%"
        elif thermal_warning():
            reason = "pmset thermal warning"
        elif not power_is_ac():
            reason = "power-state change (battery)"
        if reason:
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: {reason} during the case")
            return None
        time.sleep(1.5)
    if proc.returncode != 0:
        log(f"    ABORT {engine}/{threads}: process exit {proc.returncode}")
        return None
    max_agg = max(agg_samples) if agg_samples else 0.0
    mean_agg = sum(agg_samples) / len(agg_samples) if agg_samples else 0.0
    return json.loads(out_path.read_text()), max_agg, mean_agg, peak_max


def _rotate(seq: list, k: int) -> list:
    k %= len(seq)
    return seq[k:] + seq[:k]


def balanced_schedule(session_index: int) -> list[tuple[str, int]]:
    """True Latin square over four thread counts across four sessions (point 1); engine
    order rotates independently."""
    thread_rounds = _rotate(THREADS, session_index)
    engine_order = _rotate(ENGINES, session_index)
    return [(engine, threads) for threads in thread_rounds for engine in engine_order]


def inject_metadata(context: dict, planned: list, actual: list, unrelated: dict) -> None:
    topo_p = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
    topo_e = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]).strip()
    ncpu = _run(["sysctl", "-n", "hw.ncpu"]).strip()
    context["dw_cpu_topology"] = f"logical={ncpu}, performance={topo_p}, efficiency={topo_e}"
    context["dw_power_state"] = "AC Power" if power_is_ac() else "Battery Power"
    context["dw_thermal_probe_note"] = THERMAL_PROBE_NOTE
    context["dw_capture_timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    context["dw_load_avg_post"] = f"{load1():.2f}"
    context["dw_protocol"] = "latin-square-4session, active-tree-exclusion, agg-cpu-monitored, warmed"
    context["dw_planned_order"] = [f"{e}/{t}" for e, t in planned]
    context["dw_actual_order"] = [f"{e}/{t}" for e, t in actual]
    context["dw_case_unrelated_cpu"] = unrelated  # per case: {"agg": peak, "max": peak}


def run_session(
    index: int, session_dir: pathlib.Path, ceiling: float, ceiling_info: tuple[float, float]
) -> tuple[str, dict | None]:
    """Run one session's cases against the frozen aggregate ceiling. The caller has
    already soaked; this does not soak. Returns ("ok", record) or ("void_started", None)."""
    session_dir.mkdir(parents=True, exist_ok=True)
    planned = balanced_schedule(index)
    actual: list[tuple[str, int]] = []
    log(f"session{index} planned order: " + ", ".join(f"{e}/{t}" for e, t in planned))

    merged_benchmarks: list[dict] = []
    context: dict | None = None
    positions: dict[str, int] = {}
    unrelated: dict[str, dict] = {}

    for position, (engine, threads) in enumerate(planned):
        out_path = session_dir / f"case_{engine}_{threads}.json"
        result = None
        for attempt in range(1, MAX_CASE_RETRIES + 1):
            result = run_case(engine, threads, out_path, ceiling)
            if result is not None:
                break
            log(f"    retry {engine}/{threads} ({attempt}/{MAX_CASE_RETRIES}) after cool-down")
            if not cool_down(COOLDOWN_HIGH_MIN_S, f"retry-{engine}-{threads}"):
                return "void_started", None
        if result is None:
            log(f"session{index}: {engine}/{threads} retries exhausted -> void_started")
            return "void_started", None

        record, max_agg, mean_agg, max_proc = result
        if context is None:
            context = dict(record["context"])
        merged_benchmarks.extend(record["benchmarks"])
        positions[f"{engine}/{threads}"] = position
        unrelated[f"{engine}/{threads}"] = {
            "max_agg": round(max_agg, 1),
            "mean_agg": round(mean_agg, 1),
            "max_proc": round(max_proc, 1),
        }
        actual.append((engine, threads))
        median_ms = next(
            (b["real_time"] for b in record["benchmarks"] if b.get("aggregate_name") == "median"),
            float("nan"),
        )
        log(f"    {engine}/{threads}  pos {position:2d}  median {median_ms:.3f} ms  "
            f"agg-cpu mean {mean_agg:.0f}% max {max_agg:.0f}% (ceiling {ceiling:.0f}%)")
        if not cool_down(
            COOLDOWN_HIGH_MIN_S if threads >= 4 else COOLDOWN_LOW_MIN_S, f"{engine}-{threads}"
        ):
            return "void_started", None

    assert context is not None
    inject_metadata(context, planned, actual, unrelated)
    context["dw_case_positions"] = positions
    context["dw_agg_soak_median"] = round(ceiling_info[0], 1)
    context["dw_agg_soak_mad"] = round(ceiling_info[1], 1)
    context["dw_agg_incase_ceiling"] = round(ceiling, 1)
    merged = {"context": context, "benchmarks": merged_benchmarks}
    (session_dir / "session.json").write_text(json.dumps(merged, indent=2) + "\n")
    log(f"session{index}: complete")
    return "ok", merged


def parse(record: dict) -> dict[tuple[str, int], dict]:
    cells: dict[tuple[str, int], dict] = {}
    for row in record["benchmarks"]:
        base = row["name"].split("/real_time")[0]
        parts = base.split("/")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        key = (parts[0], int(parts[1]))
        unit = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[row.get("time_unit", "ns")]
        if row.get("aggregate_name") == "median":
            cells.setdefault(key, {})["median"] = row["real_time"] * unit
        elif row.get("aggregate_name") == "cv":
            cells.setdefault(key, {})["cv"] = row["real_time"]
    return cells


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _slope_effect(xs: list[float], ys: list[float]) -> float:
    """The predicted change in y across the observed x range from an ordinary-least-
    squares fit (effect size), so a material slope is rejected even when a small sample
    leaves the correlation's significance inconclusive."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope * (max(xs) - min(xs))


def evaluate(sessions: list[dict]) -> str:
    """ACCEPTED only if every gate passes non-vacuously across all sessions (point 5)."""
    parsed = [parse(s) for s in sessions]
    positions = [s["context"].get("dw_case_positions", {}) for s in sessions]
    unrelated = [s["context"].get("dw_case_unrelated_cpu", {}) for s in sessions]
    n = len(sessions)
    keys = sorted(set().union(*[set(p) for p in parsed]))
    expected = {(e, t) for e in ENGINES for t in THREADS}

    log("")
    # Guard against vacuous acceptance: every case present in every session, and metadata
    # consistent across sessions.
    vacuous = False
    if n != SESSIONS:
        log(f"VACUOUS: {n} sessions, expected {SESSIONS}")
        vacuous = True
    if set(keys) != expected:
        log(f"VACUOUS: case set {set(keys)} != expected {expected}")
        vacuous = True
    commits = {s["context"].get("git_commit") for s in sessions}
    topos = {s["context"].get("dw_cpu_topology") for s in sessions}
    if len(commits) != 1 or len(topos) != 1:
        log(f"VACUOUS: inconsistent metadata (commits={commits}, topologies={topos})")
        vacuous = True

    log("acceptance evaluation (medians in ms):")
    log(f"{'engine':<24}{'thr':>4}" + "".join(f"{f's{i}':>10}" for i in range(n)) + f"{'spread':>8}{'max cv':>8}")
    numeric_ok = True
    residual_points: list[tuple[float, float]] = []       # (position, residual)
    agg_points: list[tuple[float, float]] = []            # (aggregate unrelated cpu, residual)
    session_residuals: list[list[float]] = [[] for _ in range(n)]

    for engine, threads in sorted(expected):
        key = (engine, threads)
        medians = [parsed[i].get(key, {}).get("median") for i in range(n)]
        cvs = [parsed[i].get(key, {}).get("cv") for i in range(n)]
        present = [m for m in medians if m]
        flags = []
        if len(present) < n or min(present) <= 0:
            numeric_ok = False
            spread = 1.0
            flags.append("MISSING")
        else:
            spread = (max(present) - min(present)) / min(present)
            if spread > SPREAD_GATE:
                flags.append("SPREAD")
                numeric_ok = False
        # cv must exist and be non-vacuous (present but not exactly zero).
        if any(c is None for c in cvs):
            flags.append("NO_CV")
            numeric_ok = False
            max_cv = float("nan")
        else:
            max_cv = max(cvs)
            if max_cv <= 0.0:
                flags.append("ZERO_CV")
                numeric_ok = False
            elif max_cv > CV_GATE:
                flags.append("CV")
                numeric_ok = False
        if len(present) == n:
            mean_m = sum(present) / n
            for i in range(n):
                residual = (medians[i] - mean_m) / mean_m
                session_residuals[i].append(residual)
                pos = positions[i].get(f"{engine}/{threads}")
                if pos is not None:
                    residual_points.append((float(pos), residual))
                agg = unrelated[i].get(f"{engine}/{threads}", {}).get("mean_agg")
                if agg is not None:
                    agg_points.append((float(agg), residual))
        cells = "".join(f"{m:>10.3f}" if m else f"{'-':>10}" for m in medians)
        cvtxt = f"{max_cv:>7.1%}" if max_cv == max_cv else f"{'-':>8}"
        log(f"{engine:<24}{threads:>4}{cells}{spread:>7.1%}{cvtxt} {' '.join(flags)}")

    order_ok = True
    if len(residual_points) >= 3:
        corr_pos = _pearson([p for p, _ in residual_points], [r for _, r in residual_points])
        session_means = [sum(rs) / len(rs) if rs else 0.0 for rs in session_residuals]
        drift = max(session_means) - min(session_means)
        if len(agg_points) >= 3:
            corr_agg = _pearson([a for a, _ in agg_points], [r for _, r in agg_points])
            agg_effect = _slope_effect([a for a, _ in agg_points], [r for _, r in agg_points])
        else:
            corr_agg = agg_effect = 0.0
        log("")
        log(f"order:   Pearson(position, residual) = {corr_pos:+.3f} (gate |r|<{POSITION_CORR_GATE})")
        log(f"session: mean residuals {[f'{m:+.2%}' for m in session_means]} "
            f"range {drift:.2%} (gate <{SESSION_DRIFT_GATE:.0%})")
        log(f"cpu:     Pearson(mean agg cpu, residual) = {corr_agg:+.3f} (gate |r|<{AGG_CORR_GATE}); "
            f"slope effect over cpu range = {agg_effect:+.2%} (gate |e|<{AGG_SLOPE_EFFECT_GATE:.0%})")
        if abs(corr_pos) > POSITION_CORR_GATE:
            log("  REJECT: runtime depends on execution position")
            order_ok = False
        if drift > SESSION_DRIFT_GATE:
            log("  REJECT: systematic session-level slowdown")
            order_ok = False
        # Effect size, not only significance: reject a material correlation OR a material
        # fitted slope even when the small sample leaves the p-value inconclusive.
        if abs(corr_agg) > AGG_CORR_GATE or abs(agg_effect) > AGG_SLOPE_EFFECT_GATE:
            log("  REJECT: runtime tracks unrelated aggregate CPU (correlation or slope)")
            order_ok = False
    else:
        log("VACUOUS: too few residual points for the order/CPU checks")
        vacuous = True

    verdict = "ACCEPTED" if (numeric_ok and order_ok and not vacuous) else "REJECT_NUMERICAL"
    log("")
    log(verdict)
    return verdict


def one_controlled_run(run_dir: pathlib.Path) -> tuple[str, list[dict]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sessions: list[dict] = []
    frozen_ceiling: float | None = None
    ceiling_info: tuple[float, float] = (0.0, 0.0)

    for index in range(SESSIONS):
        if index > 0:
            log(f"inter-session idle: {INTER_SESSION_IDLE_S}s (not back-to-back)")
            time.sleep(INTER_SESSION_IDLE_S)

        soak_ok, samples = soak(SOAK_HOLD_S, f"session{index}", SOAK_MAX_WAIT_S)
        if not soak_ok:
            # A soak timeout before any session began is waiting-only (not an attempt);
            # after measurement has begun it voids the started attempt.
            status = "void_waiting" if not sessions else "void_started"
            (run_dir / "VERDICT").write_text(
                "VOID_ENVIRONMENT (waiting only)\n" if status == "void_waiting"
                else "VOID_ENVIRONMENT (session began, later soak timed out)\n"
            )
            return status, sessions

        if frozen_ceiling is None:
            median, mad, ceiling = derive_ceiling(samples)
            frozen_ceiling, ceiling_info = ceiling, (median, mad)
            log(f"calibrated in-case aggregate ceiling: {ceiling:.0f}% "
                f"(soak median {median:.0f}%, MAD {mad:.1f}%, cap {AGG_CEIL_CAP:.0f}%) -- "
                "frozen for all four sessions")

        status, data = run_session(index, run_dir / f"session{index}", frozen_ceiling, ceiling_info)
        if status != "ok":
            (run_dir / "VERDICT").write_text("VOID_ENVIRONMENT (session began)\n")
            return status, sessions
        assert data is not None
        sessions.append(data)

    verdict = evaluate(sessions)
    (run_dir / "VERDICT").write_text(verdict + "\n")
    return ("accepted" if verdict == "ACCEPTED" else "reject"), sessions


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    log("=== persistent controlled scaling waiter (4-session Latin square) ===")
    log(f"note: {THERMAL_PROBE_NOTE}")
    if not BIN.exists():
        log(f"error: benchmark binary missing ({BIN}); build the benchmark preset first")
        return 2

    started = time.monotonic()
    attempts = waiting_voids = 0
    while attempts < MAX_REAL_ATTEMPTS and (time.monotonic() - started) < TOTAL_HORIZON_S:
        run_dir = SCRATCH / f"run-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        outcome, sessions = one_controlled_run(run_dir)

        if outcome == "accepted":
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for i, session in enumerate(sessions):
                (REVIEW_DIR / f"controlled_session{i}.json").write_text(
                    json.dumps(session, indent=2) + "\n"
                )
            log(f"ACCEPTED_PENDING_REVIEW after {attempts} prior attempt(s); records in {REVIEW_DIR}")
            log("NOT published: the operator must inspect metadata, re-confirm the four-session "
                "comparison independently, and check for vacuous passes before committing.")
            (SCRATCH / "FINAL_VERDICT").write_text("ACCEPTED_PENDING_REVIEW\n")
            return 0

        if outcome == "void_waiting":
            waiting_voids += 1
            log(f"waiting-only void #{waiting_voids} (not an attempt); re-soaking in {WAITING_RETRY_S}s")
            time.sleep(WAITING_RETRY_S)
            continue

        attempts += 1
        log(f"real attempt {attempts}/{MAX_REAL_ATTEMPTS} outcome: {outcome.upper()}")

    if attempts >= MAX_REAL_ATTEMPTS:
        msg = (
            f"UNSUITABLE_MOVE_TO_DEDICATED: {MAX_REAL_ATTEMPTS} controlled attempts did not "
            "produce an accepted multi-thread baseline; move the harness UNCHANGED to a "
            "dedicated fixed machine."
        )
    else:
        msg = (
            f"HORIZON_EXCEEDED: {TOTAL_HORIZON_S // 3600}h elapsed with {waiting_voids} "
            "waiting-only voids and no controlled window; re-arm later or use a dedicated machine."
        )
    log(msg)
    (SCRATCH / "FINAL_VERDICT").write_text(msg + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

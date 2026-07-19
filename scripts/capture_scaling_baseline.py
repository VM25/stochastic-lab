#!/usr/bin/env python3
"""Unattended, condition-controlled Monte Carlo scaling baseline capture.

Two monolithic capture attempts on this Apple-Silicon laptop failed the 5%
session-to-session stability gate: within-session repetitions agreed to <0.1%, but
back-to-back all-core sessions drifted 10-20% at multi-thread from thermal accumulation
and a variable background-load floor, and a quiet *start* alone did not fix it. This
orchestrator controls conditions during every measurement and removes deterministic
thermal bias.

Design (numbered to the methodology requirements):

1. In-case intrusion is judged by *unrelated* processes only. The benchmark legitimately
   saturates the cores it is timing, so its process is excluded from the busiest-process
   probe and the 1-minute load average is never used as an in-case rejection signal. A
   case is aborted only for an unrelated CPU-heavy process, a pmset thermal warning, or a
   power-state change during the case.
2. The case order is a deterministic Latin-square-style rotation, not a random shuffle:
   the thread-count rounds and the engine order within a round each rotate by the session
   index, so every thread count and every engine appears in early, middle, and late
   positions across the three sessions. Both the planned and the actual order are recorded.
3. Cool-downs are condition-based, not a fixed delay: a minimum duration *and* then a
   restored state (load below threshold, no unrelated heavy process, AC power unchanged,
   pmset thermal acceptable) before the next case begins.
4. The pre-session soak is bounded. If conditions cannot be held within the limit the
   attempt is marked VOID_ENVIRONMENT, distinct from a numerical rejection.
5. Attempt semantics: waiting without starting a session is not an attempt; external
   interference that exhausts a case's retries voids the session (VOID_ENVIRONMENT); a
   completed three-session capture that fails cv, session spread, or order-independence is
   a REJECT_NUMERICAL. The three-attempt stopping rule applies to completed *and*
   repeatedly-voided controlled runs, with the distinction preserved in the verdict.
6. The thermal probe is honest about its limits: `pmset` reporting no warning means no
   OS-recorded thermal warning; it does NOT prove the absence of clock-frequency
   throttling. This limitation is recorded in every session's metadata.
7. Order-dependence is tested statistically: each case's per-session medians are turned
   into normalized residuals, and the run is rejected if the residuals show a monotonic
   dependence on execution position (Pearson correlation) or a systematic session-level
   slowdown -- not merely "the slowest result was not last".

Accepted per-session records (Google Benchmark JSON, so plot_benchmarks reads them
unchanged, plus injected environment metadata) are written to results/benchmarks/ for
review and commit. Rejected or voided runs stay in the scratch capture directory as
methodological evidence. This script commits nothing.
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
CAPTURE_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/benchmarks"
SCRATCH = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / ".capture_scratch"

ENGINES = ["european", "asian_control_variate", "heston", "greeks_pathwise_delta", "barrier_bridge"]
THREADS = [1, 2, 4, 8]

SESSIONS = 3
REPS = 12
MIN_TIME = "0.4s"
LOAD_CEIL = 2.0          # 1-minute load ceiling, used only when the benchmark is idle
CPU_CEIL = 25.0          # unrelated-process %CPU ceiling (benchmark excluded)
SOAK_HOLD_S = 120        # conditions held this long before a session
SOAK_MAX_WAIT_S = 1800   # bound on waiting for conditions (point 4)
INTER_SESSION_IDLE_S = 600
COOLDOWN_HIGH_MIN_S = 45  # minimum cool-down after a >= 4-thread case
COOLDOWN_LOW_MIN_S = 10
COOLDOWN_RESTORE_MAX_S = 900  # bound on restoring conditions after the minimum cool-down
MAX_CASE_RETRIES = 3
SPREAD_GATE = 0.05       # session-to-session median spread
CV_GATE = 0.03           # within-case coefficient of variation
POSITION_CORR_GATE = 0.30   # |Pearson(position, residual)| above this is order drift
SESSION_DRIFT_GATE = 0.03   # range of per-session mean residual above this is a slowdown

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
    out = _run(["pmset", "-g", "therm"])
    match = re.search(r"CPU_Speed_Limit *= *([0-9]+)", out)
    return bool(match) and int(match.group(1)) < 100


def busiest_unrelated(exclude_pids: set[int]) -> tuple[float, str]:
    """The busiest process that is neither the benchmark nor in exclude_pids -- so the
    benchmark's own (legitimate) core saturation never triggers an abort (point 1)."""
    lines = _run(["ps", "-Ao", "pid,pcpu,comm", "-r"]).splitlines()
    for line in lines[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pcpu, comm = int(parts[0]), float(parts[1]), parts[2]
        if pid in exclude_pids or "dw_bench_monte_carlo_scaling" in comm:
            continue
        return pcpu, comm
    return 0.0, ""


def idle_conditions() -> tuple[bool, str]:
    """Conditions when the benchmark is NOT running: AC power, no thermal warning, low
    load, and no unrelated heavy process."""
    if not power_is_ac():
        return False, "not on AC power"
    if thermal_warning():
        return False, "pmset thermal throttle"
    current = load1()
    if current >= LOAD_CEIL:
        return False, f"load {current:.2f} >= {LOAD_CEIL}"
    cpu, comm = busiest_unrelated(set())
    if cpu >= CPU_CEIL:
        return False, f"{comm.split('/')[-1]} at {cpu:.0f}%"
    return True, "ok"


def soak(hold_s: int, label: str, max_wait_s: int) -> bool:
    """Wait until idle conditions hold continuously for hold_s; False on timeout (point 4)."""
    log(f"soak[{label}]: need {hold_s}s of AC + no thermal warning + load<{LOAD_CEIL} + no proc>{CPU_CEIL}%")
    held = waited = 0
    while held < hold_s:
        ok, detail = idle_conditions()
        held = held + 10 if ok else 0
        if not ok:
            log(f"soak[{label}]: reset ({detail})")
        time.sleep(10)
        waited += 10
        if waited > max_wait_s:
            log(f"soak[{label}]: TIMEOUT after {max_wait_s}s")
            return False
    log(f"soak[{label}]: held {hold_s}s")
    return True


def cool_down(minimum_s: int, label: str) -> bool:
    """Condition-based cool-down (point 3): a minimum idle duration, then a restored state
    before the next case. False if conditions cannot be restored within the bound."""
    time.sleep(minimum_s)
    return soak(20, f"cooldown-{label}", COOLDOWN_RESTORE_MAX_S)


def run_case(engine: str, threads: int, out_path: pathlib.Path) -> dict | None:
    """Run one isolated case, monitoring unrelated interference throughout (point 1).
    Returns the case's benchmark record, or None if it had to be aborted."""
    case_filter = f"^{engine}/{threads}/real_time$"
    proc = subprocess.Popen(
        [
            str(BIN),
            f"--benchmark_filter={case_filter}",
            f"--benchmark_repetitions={REPS}",
            f"--benchmark_min_time={MIN_TIME}",
            "--benchmark_display_aggregates_only=true",
            f"--benchmark_out={out_path}",
            "--benchmark_out_format=json",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    exclude = {proc.pid}
    while proc.poll() is None:
        cpu, comm = busiest_unrelated(exclude)  # benchmark excluded; load1 NOT used here
        if cpu >= CPU_CEIL:
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: unrelated {comm.split('/')[-1]} at {cpu:.0f}%")
            return None
        if thermal_warning():
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: pmset thermal warning during the case")
            return None
        if not power_is_ac():
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: power-state change (battery) during the case")
            return None
        time.sleep(1.5)
    if proc.returncode != 0:
        log(f"    ABORT {engine}/{threads}: process exit {proc.returncode}")
        return None
    return json.loads(out_path.read_text())


def _rotate(seq: list, k: int) -> list:
    k %= len(seq)
    return seq[k:] + seq[:k]


def balanced_schedule(session_index: int) -> list[tuple[str, int]]:
    """Deterministic Latin-square-style order (point 2): thread-count rounds and the
    engine order within a round each rotate by the session index, so every thread count
    and engine visits early, middle, and late positions across the three sessions."""
    thread_rounds = _rotate(THREADS, session_index)
    engine_order = _rotate(ENGINES, session_index)
    return [(engine, threads) for threads in thread_rounds for engine in engine_order]


def inject_metadata(context: dict, planned: list, actual: list) -> None:
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
    context["dw_protocol"] = "isolated-per-case, latin-square order, unrelated-only monitoring, condition-cooled"
    context["dw_planned_order"] = [f"{e}/{t}" for e, t in planned]
    context["dw_actual_order"] = [f"{e}/{t}" for e, t in actual]


def capture_session(index: int, session_dir: pathlib.Path) -> dict | None:
    session_dir.mkdir(parents=True, exist_ok=True)
    if not soak(SOAK_HOLD_S, f"session{index}", SOAK_MAX_WAIT_S):
        return None

    planned = balanced_schedule(index)
    actual: list[tuple[str, int]] = []
    log(f"session{index} planned order: " + ", ".join(f"{e}/{t}" for e, t in planned))

    merged_benchmarks: list[dict] = []
    context: dict | None = None
    positions: dict[str, int] = {}

    for position, (engine, threads) in enumerate(planned):
        out_path = session_dir / f"case_{engine}_{threads}.json"
        record = None
        for attempt in range(1, MAX_CASE_RETRIES + 1):
            record = run_case(engine, threads, out_path)
            if record is not None:
                break
            log(f"    retry {engine}/{threads} ({attempt}/{MAX_CASE_RETRIES}) after cool-down")
            if not cool_down(COOLDOWN_HIGH_MIN_S, f"retry-{engine}-{threads}"):
                log(f"session{index}: conditions unrestorable during retry -> VOID_ENVIRONMENT")
                return None
        if record is None:
            log(f"session{index}: {engine}/{threads} retries exhausted -> VOID_ENVIRONMENT")
            return None

        if context is None:
            context = dict(record["context"])
        merged_benchmarks.extend(record["benchmarks"])
        positions[f"{engine}/{threads}"] = position
        actual.append((engine, threads))
        median_ms = next(
            (b["real_time"] for b in record["benchmarks"] if b.get("aggregate_name") == "median"),
            float("nan"),
        )
        log(f"    {engine}/{threads}  pos {position:2d}  median {median_ms:.3f} ms")
        if not cool_down(
            COOLDOWN_HIGH_MIN_S if threads >= 4 else COOLDOWN_LOW_MIN_S, f"{engine}-{threads}"
        ):
            log(f"session{index}: conditions unrestorable after case -> VOID_ENVIRONMENT")
            return None

    assert context is not None
    inject_metadata(context, planned, actual)
    context["dw_case_positions"] = positions
    merged = {"context": context, "benchmarks": merged_benchmarks}
    (session_dir / "session.json").write_text(json.dumps(merged, indent=2) + "\n")
    log(f"session{index}: complete")
    return merged


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


def evaluate(sessions: list[dict]) -> str:
    """Return ACCEPTED or REJECT_NUMERICAL (point 5, 7)."""
    parsed = [parse(s) for s in sessions]
    positions = [s["context"].get("dw_case_positions", {}) for s in sessions]
    n = len(sessions)
    keys = sorted(set().union(*[set(p) for p in parsed]))

    log("")
    log("acceptance evaluation (medians in ms):")
    log(f"{'engine':<24}{'thr':>4}" + "".join(f"{f's{i}':>11}" for i in range(n)) + f"{'spread':>8}{'max cv':>8}")

    numeric_ok = True
    residual_points: list[tuple[float, float]] = []   # (position, normalized residual)
    session_residuals: list[list[float]] = [[] for _ in range(n)]

    for engine, threads in keys:
        medians = [parsed[i].get((engine, threads), {}).get("median") for i in range(n)]
        cvs = [parsed[i].get((engine, threads), {}).get("cv", 0.0) for i in range(n)]
        present = [m for m in medians if m]
        if len(present) < n or min(present) <= 0:
            numeric_ok = False
            spread = 1.0
        else:
            spread = (max(present) - min(present)) / min(present)
        max_cv = max(cvs)
        flags = []
        if spread > SPREAD_GATE:
            flags.append("SPREAD")
            numeric_ok = False
        if max_cv > CV_GATE:
            flags.append("CV")
            numeric_ok = False
        if len(present) == n:
            mean_m = sum(present) / n
            for i in range(n):
                residual = (medians[i] - mean_m) / mean_m
                pos = positions[i].get(f"{engine}/{threads}", -1)
                if pos >= 0:
                    residual_points.append((float(pos), residual))
                    session_residuals[i].append(residual)
        cells = "".join(f"{m:>11.3f}" if m else f"{'-':>11}" for m in medians)
        log(f"{engine:<24}{threads:>4}{cells}{spread:>7.1%}{max_cv:>7.1%} {' '.join(flags)}")

    # Point 7: monotonic position dependence and systematic session slowdown.
    order_ok = True
    if residual_points:
        xs = [p for p, _ in residual_points]
        ys = [r for _, r in residual_points]
        corr = _pearson(xs, ys)
        session_means = [sum(rs) / len(rs) if rs else 0.0 for rs in session_residuals]
        drift = max(session_means) - min(session_means)
        log("")
        log(f"order-dependence: Pearson(position, residual) = {corr:+.3f} (gate |r|<{POSITION_CORR_GATE})")
        log(f"session mean residuals = {[f'{m:+.2%}' for m in session_means]} "
            f"(range {drift:.2%}, gate <{SESSION_DRIFT_GATE:.0%})")
        if abs(corr) > POSITION_CORR_GATE:
            log("  REJECT: runtime depends monotonically on execution position")
            order_ok = False
        if drift > SESSION_DRIFT_GATE:
            log("  REJECT: systematic session-level slowdown")
            order_ok = False

    verdict = "ACCEPTED" if (numeric_ok and order_ok) else "REJECT_NUMERICAL"
    log("")
    log(verdict)
    return verdict


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    run_dir = SCRATCH / f"run-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== controlled scaling capture, {SESSIONS} sessions -> {run_dir} ===")
    log(f"note: {THERMAL_PROBE_NOTE}")

    if not BIN.exists():
        log(f"error: benchmark binary missing ({BIN}); build the benchmark preset first")
        return 2

    sessions: list[dict] = []
    for index in range(SESSIONS):
        if index > 0:
            log(f"inter-session idle: {INTER_SESSION_IDLE_S}s (not back-to-back)")
            time.sleep(INTER_SESSION_IDLE_S)
        session = capture_session(index, run_dir / f"session{index}")
        if session is None:
            (run_dir / "VERDICT").write_text("VOID_ENVIRONMENT\n")
            log("run VOID_ENVIRONMENT: could not hold controlled conditions")
            return 3
        sessions.append(session)

    verdict = evaluate(sessions)
    (run_dir / "VERDICT").write_text(verdict + "\n")

    if verdict == "ACCEPTED":
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        for i, session in enumerate(sessions):
            (CAPTURE_DIR / f"monte_carlo_scaling.controlled_session{i}.json").write_text(
                json.dumps(session, indent=2) + "\n"
            )
        log(f"wrote {SESSIONS} accepted session records to {CAPTURE_DIR} for review and commit")
        return 0
    log(f"rejected run kept in {run_dir} as methodological evidence; nothing published")
    return 1


if __name__ == "__main__":
    sys.exit(main())

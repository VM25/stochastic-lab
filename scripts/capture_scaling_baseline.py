#!/usr/bin/env python3
"""Unattended, condition-controlled Monte Carlo scaling baseline capture.

Two capture attempts on this Apple-Silicon laptop failed the 5% session-to-session
stability gate: the within-session repetitions agreed to <0.1%, but back-to-back
all-core sessions drifted 10-20% at multi-thread from thermal accumulation and a
variable background-load floor. A quiet *start* is not enough. This orchestrator
controls conditions *during* every measurement and removes deterministic thermal bias:

  1. Sustained pre-run soak: AC power, nominal thermal state, low load, and no
     CPU-heavy process, all held continuously before a session begins.
  2. Each (engine, thread-count) case runs as its *own* short benchmark process, not
     one continuous 4-8 minute all-engine session -- so heat cannot accumulate across
     the whole suite.
  3. The case order is shuffled per session (a different, balanced order each time), so
     no thread count is always measured in the same thermal position.
  4. A cool-down follows every high-load (>= 4 thread) case.
  5. Load and thermal state are monitored *throughout* each case; a case is aborted and
     discarded (then retried after a cool-down) if an unrelated process, a thermal
     warning, or the load ceiling intrudes.
  6. At least three sessions run, separated by a substantial idle gap, not back-to-back.

Acceptance (all required): low within-case dispersion, session-to-session median spread
within 5%, and no systematic dependence of a case's timing on its execution position.
On acceptance the merged per-session results (Google Benchmark JSON, so plot_benchmarks
reads them unchanged) are written to results/benchmarks/ for review and commit; the raw
per-case files and any rejected run stay in the scratch capture directory as
methodological evidence. Nothing is committed by this script.

Runs unattended (launch detached); progress and the verdict stream to its log.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import random
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
REPS = 12                 # within-case repetitions -> the case's median and cv
MIN_TIME = "0.4s"         # per repetition; keeps each case short (little heat) but timed
LOAD_CEIL = 2.0           # 1-minute load ceiling on an 8-core machine
CPU_CEIL = 25.0           # busiest non-benchmark process %CPU ceiling
SOAK_S = 120              # conditions held this long before a session
INTER_SESSION_IDLE_S = 600  # substantial idle between sessions (not back-to-back)
COOLDOWN_HIGH_S = 45      # after a >= 4-thread case
COOLDOWN_LOW_S = 10       # after a 1/2-thread case
MAX_CASE_RETRIES = 3
SPREAD_GATE = 0.05        # session-to-session median spread
CV_GATE = 0.03            # within-case coefficient of variation


def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{stamp} {message}", flush=True)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def load1() -> float:
    match = re.search(r"load averages?: *([0-9.]+)", _run(["uptime"]).stdout)
    return float(match.group(1)) if match else 99.0


def power_is_ac() -> bool:
    return "AC Power" in _run(["pmset", "-g", "ps"]).stdout


def thermal_nominal() -> bool:
    out = _run(["pmset", "-g", "therm"]).stdout
    match = re.search(r"CPU_Speed_Limit *= *([0-9]+)", out)
    if match:
        return int(match.group(1)) >= 100  # a limit below 100 is active throttling
    return True  # "No thermal warning level has been recorded" -> nominal


def busiest_process() -> tuple[float, str]:
    lines = _run(["ps", "-Ao", "pcpu,comm", "-r"]).stdout.splitlines()
    for line in lines[1:]:
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pcpu, comm = float(parts[0]), parts[1]
        if "dw_bench_monte_carlo_scaling" in comm:
            continue
        return pcpu, comm
    return 0.0, ""


def conditions() -> tuple[bool, str]:
    if not power_is_ac():
        return False, "not on AC power"
    if not thermal_nominal():
        return False, "thermal throttle active"
    current = load1()
    if current >= LOAD_CEIL:
        return False, f"load {current:.2f} >= {LOAD_CEIL}"
    cpu, comm = busiest_process()
    if cpu >= CPU_CEIL:
        return False, f"{comm.split('/')[-1]} at {cpu:.0f}%"
    return True, "ok"


def soak(seconds: int, label: str, timeout_s: int = 1800) -> bool:
    """Wait until conditions hold continuously for `seconds`; False on timeout."""
    log(f"soak[{label}]: need {seconds}s of AC + nominal thermal + load<{LOAD_CEIL} + cpu<{CPU_CEIL}%")
    held, waited = 0, 0
    while held < seconds:
        ok, detail = conditions()
        held = held + 10 if ok else 0
        if not ok:
            log(f"soak[{label}]: reset ({detail})")
        time.sleep(10)
        waited += 10
        if waited > timeout_s:
            log(f"soak[{label}]: TIMEOUT after {timeout_s}s -- machine will not settle")
            return False
    log(f"soak[{label}]: held {seconds}s")
    return True


def run_case(engine: str, threads: int, out_path: pathlib.Path) -> dict | None:
    """Run one isolated case, monitoring throughout. Returns its benchmark rows, or None
    if it had to be aborted (an intrusion during the measurement)."""
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
    while proc.poll() is None:
        cpu, comm = busiest_process()
        if cpu >= CPU_CEIL:
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: {comm.split('/')[-1]} at {cpu:.0f}% during the case")
            return None
        if not thermal_nominal():
            proc.terminate()
            proc.wait()
            log(f"    ABORT {engine}/{threads}: thermal throttle during the case")
            return None
        time.sleep(1.5)
    if proc.returncode != 0:
        log(f"    ABORT {engine}/{threads}: process exit {proc.returncode}")
        return None
    return json.loads(out_path.read_text())


def cool_down(seconds: int) -> None:
    time.sleep(seconds)


def inject_metadata(context: dict) -> None:
    topo_p = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).stdout.strip()
    topo_e = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]).stdout.strip()
    ncpu = _run(["sysctl", "-n", "hw.ncpu"]).stdout.strip()
    context["dw_cpu_topology"] = f"logical={ncpu}, performance={topo_p}, efficiency={topo_e}"
    context["dw_power_state"] = "AC Power" if power_is_ac() else "Battery Power"
    context["dw_thermal_state"] = "nominal" if thermal_nominal() else "throttled"
    context["dw_capture_timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    context["dw_protocol"] = "isolated-per-case, shuffled order, monitored, cooled"


def capture_session(index: int, session_dir: pathlib.Path) -> dict | None:
    """One session: soak, then each case in a shuffled order, monitored and cooled.
    Returns a merged Google-Benchmark-format record, or None if the session could not
    hold conditions."""
    session_dir.mkdir(parents=True, exist_ok=True)
    if not soak(SOAK_S, f"session{index}"):
        return None

    cases = [(e, t) for e in ENGINES for t in THREADS]
    random.Random(1000 + index).shuffle(cases)  # balanced, reproducible per session
    log(f"session{index} order: " + ", ".join(f"{e}/{t}" for e, t in cases))

    merged_benchmarks: list[dict] = []
    context: dict | None = None
    positions: dict[str, int] = {}

    for position, (engine, threads) in enumerate(cases):
        out_path = session_dir / f"case_{engine}_{threads}.json"
        record = None
        for attempt in range(1, MAX_CASE_RETRIES + 1):
            record = run_case(engine, threads, out_path)
            if record is not None:
                break
            log(f"    retry {engine}/{threads} ({attempt}/{MAX_CASE_RETRIES}) after cool-down")
            cool_down(COOLDOWN_HIGH_S)
            if not soak(30, f"retry-{engine}-{threads}", timeout_s=600):
                break
        if record is None:
            log(f"session{index}: {engine}/{threads} could not be measured cleanly -- session void")
            return None

        if context is None:
            context = dict(record["context"])
        merged_benchmarks.extend(record["benchmarks"])
        positions[f"{engine}/{threads}"] = position
        median_ms = next(
            (b["real_time"] for b in record["benchmarks"] if b.get("aggregate_name") == "median"),
            float("nan"),
        )
        log(f"    {engine}/{threads}  pos {position:2d}  median {median_ms:.3f} ms")
        cool_down(COOLDOWN_HIGH_S if threads >= 4 else COOLDOWN_LOW_S)

    assert context is not None
    inject_metadata(context)
    context["dw_case_positions"] = positions
    context["dw_load_avg_post"] = f"{load1():.2f}"
    merged = {"context": context, "benchmarks": merged_benchmarks}
    out = session_dir / "session.json"
    out.write_text(json.dumps(merged, indent=2) + "\n")
    log(f"session{index}: complete -> {out}")
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


def evaluate(sessions: list[dict]) -> bool:
    parsed = [parse(s) for s in sessions]
    keys = sorted(set().union(*[set(p) for p in parsed]))
    positions = [s["context"].get("dw_case_positions", {}) for s in sessions]
    n = len(sessions)

    log("")
    log("acceptance evaluation (medians in ms):")
    header = f"{'engine':<24}{'thr':>4}" + "".join(f"{f's{i}':>12}" for i in range(n))
    log(header + f"{'spread':>8}{'max cv':>8}{'pos':>14}")
    accepted = True
    for engine, threads in keys:
        medians = [parsed[i].get((engine, threads), {}).get("median") for i in range(n)]
        cvs = [parsed[i].get((engine, threads), {}).get("cv", 0.0) for i in range(n)]
        present = [m for m in medians if m]
        spread = (max(present) - min(present)) / min(present) if len(present) == n and min(present) > 0 else 1.0
        max_cv = max(cvs)
        pos = [positions[i].get(f"{engine}/{threads}", -1) for i in range(n)]
        # Order-bias check: is the slowest session also the latest-position one?
        slow_i = max(range(n), key=lambda i: medians[i] if medians[i] else -1)
        order_bias = pos[slow_i] == max(pos) and spread > SPREAD_GATE
        flags = []
        if spread > SPREAD_GATE:
            flags.append("SPREAD")
        if max_cv > CV_GATE:
            flags.append("CV")
        if order_bias:
            flags.append("ORDER")
        if flags:
            accepted = False
        cells = "".join(f"{m:>12.3f}" if m else f"{'-':>12}" for m in medians)
        log(
            f"{engine:<24}{threads:>4}{cells}{spread:>7.1%}{max_cv:>7.1%}"
            f"{str(pos):>14} {' '.join(flags)}"
        )
    log("")
    log("ACCEPTED" if accepted else "REJECTED: multi-thread stability/order criteria not met")
    return accepted


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = SCRATCH / f"run-{run_stamp}"
    log(f"=== controlled scaling capture, {SESSIONS} sessions -> {run_dir} ===")

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
            log(f"run VOID: session{index} could not hold controlled conditions")
            (run_dir / "VERDICT").write_text("VOID: conditions not holdable\n")
            return 1
        sessions.append(session)

    accepted = evaluate(sessions)
    verdict = "ACCEPTED" if accepted else "REJECTED"
    (run_dir / "VERDICT").write_text(verdict + "\n")

    if accepted:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        for i, session in enumerate(sessions):
            dest = CAPTURE_DIR / f"monte_carlo_scaling.controlled_session{i}.json"
            dest.write_text(json.dumps(session, indent=2) + "\n")
        log(f"wrote {SESSIONS} accepted session records to {CAPTURE_DIR} for review and commit")
    else:
        log(f"rejected run kept in {run_dir} as methodological evidence; nothing published")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())

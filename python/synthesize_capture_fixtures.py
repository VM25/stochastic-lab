#!/usr/bin/env python3
"""Fabricate SYNTHETIC four-session controlled-capture datasets for reviewer validation.

This script writes four ``controlled_session{0..3}.json`` files in the exact JSON shape
that scripts/capture_scaling_baseline.py emits on acceptance -- the same context metadata
(register_metadata() build/host keys plus the orchestrator's dw_* fields) and the same
Google-Benchmark ``benchmarks`` rows -- so python/review_scaling_baseline.py can be
exercised against known-good and known-bad inputs without a real measurement machine.

The numbers here are INVENTED, not measured. These fixtures exist only to prove the
independent reviewer accepts a clean capture and rejects each defect it is meant to catch.
They must never be committed to results/, published, or cited as a performance baseline.

Each invocation emits exactly one scenario, deterministic for a given --seed:

  clean         a valid capture              -> reviewer REVIEW PASSED (exit 0)
  bad_flags     one session's build_flags differ            -> identity check (exit 1)
  retry3        one case retried in 3 of 4 sessions          -> retry-bias check (exit 1)
  dup_rows      one case's iteration rows duplicated once    -> workload/case-count (exit 1)
  bad_nsamples  one case's n_samples != len(samples)         -> monitoring check (exit 1)

Usage:
    python3 python/synthesize_capture_fixtures.py <outdir> --scenario clean [--seed N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics

ENGINES = ["european", "asian_control_variate", "heston", "greeks_pathwise_delta", "barrier_bridge"]
THREADS = [1, 2, 4, 8]
REPS = 12
BASE_MS = {"european": 5.0, "asian_control_variate": 8.0, "heston": 12.0,
           "greeks_pathwise_delta": 15.0, "barrier_bridge": 20.0}
SCENARIOS = ["clean", "bad_flags", "retry3", "dup_rows", "bad_nsamples"]
JITTER_HALF_WIDTH = 0.004  # +-0.4% -> small non-zero cv, comfortably under the 3% gate


def rotate(seq: list, k: int) -> list:
    k %= len(seq)
    return seq[k:] + seq[:k]


def schedule(index: int) -> list[tuple[str, int]]:
    """The true Latin square balanced_schedule() produces (engine order rotates too)."""
    thread_rounds = rotate(THREADS, index)
    engine_order = rotate(ENGINES, index)
    return [(engine, threads) for threads in thread_rounds for engine in engine_order]


def case_ms(engine: str, threads: int) -> float:
    return BASE_MS[engine] + threads * 0.3  # deterministic, positive, session-independent


def iteration_rows(engine: str, threads: int, jitter: list[float]) -> list[dict]:
    rows = []
    base = case_ms(engine, threads)
    for j in range(REPS):
        val = base * (1.0 + jitter[j])
        rows.append({
            "name": f"{engine}/{threads}/real_time",
            "run_name": f"{engine}/{threads}/real_time",
            "run_type": "iteration",
            "repetitions": REPS,
            "repetition_index": j,
            "threads": threads,
            "iterations": 128,
            "real_time": val,
            "cpu_time": val,
            "time_unit": "ms",
        })
    return rows


def aggregate_rows(engine: str, threads: int, jitter: list[float]) -> list[dict]:
    vals = [case_ms(engine, threads) * (1.0 + jitter[j]) for j in range(REPS)]
    mean = statistics.fmean(vals)
    stats = {"median": statistics.median(vals), "cv": statistics.stdev(vals) / mean}
    out = []
    for name, value in stats.items():
        out.append({
            "name": f"{engine}/{threads}/real_time_{name}",
            "run_name": f"{engine}/{threads}/real_time",
            "run_type": "aggregate",
            "aggregate_name": name,
            "aggregate_unit": "time" if name == "median" else "percentage",
            "threads": threads,
            "real_time": value,
            "cpu_time": value,
            "time_unit": "ms",
        })
    return out


def monitoring() -> dict:
    """A per-case unrelated-CPU monitoring block: 10 samples at 0.25s, all below the
    ceiling, with n_samples within the duration/interval budget."""
    samples = [round(20.0 + (j % 5) * 1.5, 1) for j in range(10)]
    n = len(samples)
    return {
        "samples": samples,
        "n_samples": n,
        "max_agg": round(max(samples), 1),
        "mean_agg": round(sum(samples) / n, 1),
        "max_proc": 12.0,
        "duration_s": round(n * 0.25 + 0.35, 2),
        "interval_s": 0.25,
        "warmup_s": 0.15,
        "covered_warmup_and_reps": True,
    }


def soak_samples() -> list[float]:
    return [round(18.0 + (j % 4) * 1.0, 1) for j in range(12)]


def build_session(index: int, jitters: dict[tuple[str, int], list[float]]) -> dict:
    plan = schedule(index)
    order = [f"{e}/{t}" for e, t in plan]
    positions = {f"{e}/{t}": pos for pos, (e, t) in enumerate(plan)}
    unrelated = {f"{e}/{t}": monitoring() for e, t in plan}

    benchmarks: list[dict] = []
    for engine, threads in plan:
        jitter = jitters[(engine, threads)]
        benchmarks.extend(iteration_rows(engine, threads, jitter))
        benchmarks.extend(aggregate_rows(engine, threads, jitter))

    ss = soak_samples()
    smed = statistics.median(ss)
    smad = statistics.median([abs(s - smed) for s in ss])

    context = {
        # Google Benchmark builtins (a representative subset).
        "date": "2026-07-20T09:00:00+00:00",
        "host_name": "dedicated-bench-01",
        "num_cpus": 8,
        "mhz_per_cpu": 3200,
        "cpu_scaling_enabled": False,
        # register_metadata() custom context -- identical across all four sessions.
        "dw_version": "0.13.0",
        "compiler": "AppleClang 17.0.0",
        "build_type": "Release",
        "build_flags": "-O3 -DNDEBUG -std=c++20",
        "cxx_standard": "20",
        "git_commit": "2414515abcdef0123456789",
        "git_branch": "main",
        "os": "macOS 15.5",
        "cpu_brand": "Apple M3",
        "seed": "2718281828",
        # inject_metadata() + run_session() fields.
        "dw_cpu_topology": "logical=8, performance=4, efficiency=4",
        "dw_power_state": "AC Power",
        "dw_thermal_probe_note": "pmset reports no OS-recorded thermal warning ...",
        "dw_capture_timestamp_utc": f"2026-07-20T0{index}:00:00Z",
        "dw_load_avg_post": "0.80",
        "dw_protocol": "latin-square-4session, active-tree-exclusion, agg-cpu-monitored, warmed",
        "dw_planned_order": order,
        "dw_actual_order": list(order),
        "dw_case_unrelated_cpu": unrelated,
        "dw_case_positions": positions,
        "dw_agg_soak_median": round(smed, 1),
        "dw_agg_soak_mad": round(smad, 1),
        "dw_agg_soak_samples": ss,
        "dw_agg_incase_ceiling": 60.0,
        "dw_case_aborts": 0,
        "dw_case_retries": {},
    }
    return {"context": context, "benchmarks": benchmarks}


def build_sessions(rng: random.Random) -> list[dict]:
    # One jitter pattern per case, drawn in a fixed order for reproducibility and shared
    # across the four sessions so the medians match session-to-session (residuals ~ 0).
    jitters = {
        (engine, threads): [rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH) for _ in range(REPS)]
        for threads in THREADS
        for engine in ENGINES
    }
    return [build_session(i, jitters) for i in range(4)]


def apply_scenario(sessions: list[dict], scenario: str) -> None:
    """Inject exactly one defect (or none, for 'clean') into the built sessions."""
    if scenario == "clean":
        return
    if scenario == "bad_flags":
        # One session built with different optimisation flags -> identity check fails.
        sessions[2]["context"]["build_flags"] = "-O2 -DNDEBUG -std=c++20 -march=native"
        return
    if scenario == "retry3":
        # The same case retries in three of the four sessions (each within the per-session
        # limit) -> selective-environment retry-bias check fails.
        for i in (0, 1, 2):
            sessions[i]["context"]["dw_case_retries"] = {"heston/8": 1}
            sessions[i]["context"]["dw_case_aborts"] = 1
        return
    if scenario == "dup_rows":
        # One case's 12 iteration rows appended twice in one session -> its per-case count
        # becomes 24 (a multiple of the constant); workload + case-count checks fail.
        dup = [dict(r) for r in sessions[1]["benchmarks"]
               if r["run_type"] == "iteration" and r["name"] == "european/2/real_time"]
        sessions[1]["benchmarks"].extend(dup)
        return
    if scenario == "bad_nsamples":
        # Recorded n_samples inflated past the raw sample list -> monitoring-consistency
        # check fails.
        mon = sessions[0]["context"]["dw_case_unrelated_cpu"]["heston/4"]
        mon["n_samples"] = len(mon["samples"]) + 5
        return
    raise ValueError(f"unknown scenario: {scenario}")


def write_sessions(outdir: pathlib.Path, sessions: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, session in enumerate(sessions):
        (outdir / f"controlled_session{i}.json").write_text(json.dumps(session, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fabricate a synthetic four-session controlled-capture fixture for "
                    "validating python/review_scaling_baseline.py (data is SYNTHETIC).")
    parser.add_argument("outdir", help="directory to write controlled_session{0..3}.json into")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="which dataset to fabricate (one per invocation)")
    parser.add_argument("--seed", type=int, default=20260720,
                        help="RNG seed for the per-case timing jitter (deterministic output)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sessions = build_sessions(rng)
    apply_scenario(sessions, args.scenario)
    outdir = pathlib.Path(args.outdir)
    write_sessions(outdir, sessions)
    print(f"wrote SYNTHETIC scenario '{args.scenario}' (seed {args.seed}) -> "
          f"{outdir}/controlled_session0..3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent acceptance review of a four-session controlled scaling capture.

Deliberately a *separate* implementation from scripts/capture_scaling_baseline.py: it
re-derives every acceptance metric from the raw Google Benchmark rows and the recorded
metadata, so a bug in the orchestrator's own verdict cannot pass unnoticed. It does not
import the orchestrator.

Checks (the operator's four-session acceptance review):
  - all 20 engine/thread combinations appear exactly once in each session;
  - every thread count occupies every Latin-square round position across the four sessions;
  - each session's soak median does not drift materially from session 0;
  - every case carried enough monitoring samples, and no accepted case's peak aggregate
    unrelated CPU exceeded the frozen ceiling;
  - CVs, session spreads, position drift, session drift, and the CPU effect (correlation
    and slope) recomputed from scratch -- CVs from the per-repetition timings, not from
    the benchmark's own cv aggregate;
  - the actual per-session runtime sequence is printed for inspection.

Usage:
    python3 python/review_scaling_baseline.py <dir with controlled_session*.json>
    python3 python/review_scaling_baseline.py session0.json session1.json ...
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

ENGINES = ["european", "asian_control_variate", "heston", "greeks_pathwise_delta", "barrier_bridge"]
THREADS = [1, 2, 4, 8]
EXPECTED = {(e, t) for e in ENGINES for t in THREADS}
ENGINES_PER_ROUND = len(ENGINES)

CV_GATE = 0.03
SPREAD_GATE = 0.05
POSITION_CORR_GATE = 0.30
SESSION_DRIFT_GATE = 0.03
AGG_CORR_GATE = 0.30
AGG_SLOPE_EFFECT_GATE = 0.02
SOAK_MEDIAN_DRIFT_GATE = 15.0
MIN_CPU_SAMPLES = 2

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL: {message}")


def unit_scale(unit: str) -> float:
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


def per_rep_seconds(record: dict) -> dict[tuple[str, int], list[float]]:
    """The individual repetition wall-times per case, in seconds -- recomputing CV from
    these rather than trusting the benchmark's cv aggregate."""
    reps: dict[tuple[str, int], list[float]] = {}
    for row in record["benchmarks"]:
        if row.get("run_type") != "iteration":
            continue
        base = row["name"].split("/real_time")[0]
        parts = base.split("/")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        key = (parts[0], int(parts[1]))
        reps.setdefault(key, []).append(row["real_time"] * unit_scale(row.get("time_unit", "ns")))
    return reps


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def slope_effect(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope * (max(xs) - min(xs))


def load_sessions(args: list[str]) -> list[dict]:
    paths: list[pathlib.Path] = []
    for arg in args:
        p = pathlib.Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("controlled_session*.json")))
        else:
            paths.append(p)
    return [json.loads(p.read_text()) for p in paths]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sessions = load_sessions(sys.argv[1:])
    n = len(sessions)
    print(f"=== independent review of {n} session(s) ===\n")
    if n != 4:
        fail(f"expected 4 sessions, got {n}")

    # Per-session: recompute median/cv from raw reps; check completeness and ceiling.
    per_case_median: list[dict[tuple[str, int], float]] = []
    per_case_cv: list[dict[tuple[str, int], float]] = []
    positions: list[dict[str, int]] = []
    mean_agg: list[dict[str, float]] = []

    for i, s in enumerate(sessions):
        ctx = s["context"]
        reps = per_rep_seconds(s)
        cases = set(reps)
        if cases != EXPECTED:
            fail(f"session{i}: case set != the 20 expected (missing {EXPECTED - cases}, "
                 f"extra {cases - EXPECTED})")
        medians, cvs = {}, {}
        for key, values in reps.items():
            medians[key] = statistics.median(values)
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            cvs[key] = sd / mean if mean > 0 else float("inf")
        per_case_median.append(medians)
        per_case_cv.append(cvs)
        positions.append(ctx.get("dw_case_positions", {}))

        ceiling = ctx.get("dw_agg_incase_ceiling")
        unrel = ctx.get("dw_case_unrelated_cpu", {})
        mean_agg.append({k: v.get("mean_agg") for k, v in unrel.items()})
        for case, stats in unrel.items():
            if stats.get("n_samples", 0) < MIN_CPU_SAMPLES:
                fail(f"session{i} {case}: {stats.get('n_samples')} monitoring samples")
            if ceiling is not None and stats.get("max_agg", 0) > ceiling:
                fail(f"session{i} {case}: peak aggregate {stats['max_agg']}% exceeded ceiling {ceiling}%")
        print(f"session{i}: soak median {ctx.get('dw_agg_soak_median')}% MAD {ctx.get('dw_agg_soak_mad')}% "
              f"ceiling {ceiling}% aborts {ctx.get('dw_case_aborts')} "
              f"commit {str(ctx.get('git_commit'))[:9]}")

    # Latin-square: every thread count occupies every round position across sessions.
    print("\n=== Latin-square positional balance ===")
    for t in THREADS:
        rounds = []
        for i in range(n):
            case_pos = [positions[i].get(f"{e}/{t}") for e in ENGINES if positions[i].get(f"{e}/{t}") is not None]
            rounds.append(min(case_pos) // ENGINES_PER_ROUND if case_pos else None)
        if sorted(r for r in rounds if r is not None) != list(range(n)):
            fail(f"thread {t}: round positions {rounds} are not a full permutation")
        else:
            print(f"  thread {t}: rounds {rounds} -> complete")

    # Soak-median drift.
    soak_medians = [s["context"].get("dw_agg_soak_median") for s in sessions]
    if all(m is not None for m in soak_medians):
        drift = [abs(m - soak_medians[0]) for m in soak_medians]
        print(f"\nsoak medians {soak_medians}%  drift-from-s0 {[round(d, 1) for d in drift]}% "
              f"(gate <{SOAK_MEDIAN_DRIFT_GATE}%)")
        if max(drift) > SOAK_MEDIAN_DRIFT_GATE:
            fail(f"soak median drift {max(drift):.1f}% exceeds {SOAK_MEDIAN_DRIFT_GATE}%")

    # CV, spread, and the residual-based effects, all recomputed here.
    print("\n=== recomputed medians (ms), spread, max cv ===")
    residual_points, agg_points = [], []
    session_residuals = [[] for _ in range(n)]
    for key in sorted(EXPECTED):
        medians = [per_case_median[i].get(key) for i in range(n)]
        cvs = [per_case_cv[i].get(key, float("inf")) for i in range(n)]
        if any(m is None for m in medians):
            fail(f"{key}: missing in some session")
            continue
        spread = (max(medians) - min(medians)) / min(medians)
        max_cv = max(cvs)
        if spread > SPREAD_GATE:
            fail(f"{key[0]}/{key[1]}: spread {spread:.1%} > {SPREAD_GATE:.0%}")
        if max_cv > CV_GATE:
            fail(f"{key[0]}/{key[1]}: cv {max_cv:.1%} > {CV_GATE:.0%}")
        mean_m = statistics.fmean(medians)
        for i in range(n):
            r = (medians[i] - mean_m) / mean_m
            session_residuals[i].append(r)
            pos = positions[i].get(f"{key[0]}/{key[1]}")
            if pos is not None:
                residual_points.append((float(pos), r))
            agg = mean_agg[i].get(f"{key[0]}/{key[1]}")
            if agg is not None:
                agg_points.append((float(agg), r))
        print(f"  {key[0]:<24}{key[1]:>2}  " + " ".join(f"{m*1e3:8.3f}" for m in medians)
              + f"  spread {spread:5.1%}  cv {max_cv:4.1%}")

    if len(residual_points) >= 3:
        cp = pearson([p for p, _ in residual_points], [r for _, r in residual_points])
        sm = [statistics.fmean(rs) if rs else 0.0 for rs in session_residuals]
        sd = max(sm) - min(sm)
        ca = pearson([a for a, _ in agg_points], [r for _, r in agg_points]) if len(agg_points) >= 3 else 0.0
        ce = slope_effect([a for a, _ in agg_points], [r for _, r in agg_points]) if len(agg_points) >= 3 else 0.0
        print(f"\nposition corr {cp:+.3f} (gate <{POSITION_CORR_GATE}) | session drift {sd:.2%} "
              f"(gate <{SESSION_DRIFT_GATE:.0%}) | cpu corr {ca:+.3f} slope {ce:+.2%} "
              f"(gates <{AGG_CORR_GATE}, <{AGG_SLOPE_EFFECT_GATE:.0%})")
        if abs(cp) > POSITION_CORR_GATE:
            fail(f"position correlation {cp:+.3f}")
        if sd > SESSION_DRIFT_GATE:
            fail(f"session drift {sd:.2%}")
        if abs(ca) > AGG_CORR_GATE or abs(ce) > AGG_SLOPE_EFFECT_GATE:
            fail(f"cpu effect (corr {ca:+.3f}, slope {ce:+.2%})")
    else:
        fail("too few residual points for the effect checks")

    # The actual runtime sequence, in execution order, for inspection.
    print("\n=== runtime sequence in execution order ===")
    for i in range(n):
        order = sorted(positions[i], key=lambda c: positions[i][c])
        seq = []
        for case in order:
            e, t = case.rsplit("/", 1)
            m = per_case_median[i].get((e, int(t)))
            seq.append(f"{case}:{m*1e3:.1f}" if m else f"{case}:?")
        print(f"  session{i}: " + "  ".join(seq))

    print()
    if failures:
        print(f"REVIEW FAILED: {len(failures)} issue(s) -- do NOT commit this baseline.")
        return 1
    print("REVIEW PASSED: all acceptance criteria independently confirmed. Safe to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Independent acceptance review of a four-session controlled scaling capture.

Deliberately a *separate* implementation from scripts/capture_scaling_baseline.py: it
re-derives every acceptance metric from the raw Google Benchmark rows and the recorded
metadata, so a bug in the orchestrator's own verdict cannot pass unnoticed. It does not
import the orchestrator.

Checks (the operator's four-session acceptance review):
  - all 20 engine/thread combinations appear exactly once in each session;
  - build/host metadata identity across all four sessions (compiler, build_type,
    build_flags, cxx_standard, seed, dw_version, os, cpu_brand, git_commit, cpu topology),
    and dw_planned_order == dw_actual_order within each session;
  - a single per-repetition iteration-row count shared by every case and session, and a
    single monitoring interval_s / warmup_s shared by every case and session;
  - monitoring internal consistency per case: recorded n_samples == len(samples), and the
    sample count fits the duration/interval budget (n_samples <= duration_s/interval_s + 2,
    and n_samples >= 8);
  - retry-frequency bias: no case retried more than twice in a session, and no case retried
    in three or more of the four sessions (a case whose environment systematically differs);
  - exactly 20 cases per session, each present exactly once -- iteration rows equal the
    shared constant, not a multiple of it (catches a duplicated retried block);
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
SOAK_DRIFT_MAX_PP = 10.0
SOAK_DRIFT_MIN_PP = 5.0
SOAK_DRIFT_MAD_K = 3.0
MIN_CPU_SAMPLES = 8
CASES_PER_SESSION = 20
MAX_RETRIES_PER_CASE = 2   # a single case may retry at most twice in one session
MAX_RETRY_SESSIONS = 2     # a case retried in > 2 (i.e. 3+) sessions is selective-environment bias

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


def mad(xs: list[float]) -> float:
    med = statistics.median(xs)
    return statistics.median([abs(x - med) for x in xs])


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


def check_metadata_identity(sessions: list[dict]) -> None:
    """Check 1: build/host metadata identical across all sessions, and each session's
    planned order equals its actual order. A silent build/flag/seed difference between
    sessions would make the four-session comparison meaningless, so it fails loudly."""
    print("\n=== metadata identity across sessions ===")
    identity_keys = [
        "git_commit", "dw_cpu_topology", "compiler", "build_type", "build_flags",
        "cxx_standard", "seed", "dw_version", "os", "cpu_brand",
    ]
    for key in identity_keys:
        values = [s["context"].get(key) for s in sessions]
        distinct = {json.dumps(v, sort_keys=True) for v in values}
        if len(distinct) != 1:
            fail(f"metadata '{key}' differs across sessions: {values}")
        elif values[0] is None:
            fail(f"metadata '{key}' missing in all sessions")
        else:
            print(f"  {key:<16} {str(values[0])[:56]}")
    for i, s in enumerate(sessions):
        planned = s["context"].get("dw_planned_order")
        actual = s["context"].get("dw_actual_order")
        if planned is None or actual is None:
            fail(f"session{i}: missing dw_planned_order/dw_actual_order")
        elif planned != actual:
            fail(f"session{i}: dw_actual_order != dw_planned_order")
        else:
            print(f"  session{i}: planned order == actual order ({len(actual)} cases)")


def check_workload_consistency(
    iter_counts: list[dict[tuple[str, int], int]],
    interval_values: list,
    warmup_values: list,
) -> int | None:
    """Check 2: one iteration-row count shared by every case/session, and one monitoring
    interval_s / warmup_s shared by every case/session. Returns the shared iteration
    constant (the most common value when it varies, so check 5 can pinpoint offenders)."""
    print("\n=== workload / repetition / warm-up consistency ===")
    all_counts = [c for counts in iter_counts for c in counts.values()]
    constant: int | None = None
    if not all_counts:
        fail("no per-repetition iteration rows found in any session")
    else:
        tally: dict[int, int] = {}
        for c in all_counts:
            tally[c] = tally.get(c, 0) + 1
        constant = max(tally, key=lambda k: tally[k])
        if len(tally) != 1:
            fail(f"per-repetition iteration-row count varies across cases/sessions: "
                 f"{sorted(tally)} (most common {constant})")
        else:
            print(f"  per-case iteration rows: {constant} "
                  f"(identical across {len(all_counts)} case blocks)")
    for label, values in (("interval_s", interval_values), ("warmup_s", warmup_values)):
        if not values:
            fail(f"monitoring {label}: none recorded")
            continue
        distinct = {json.dumps(v) for v in values}
        if len(distinct) != 1:
            fail(f"monitoring {label} varies across cases/sessions: "
                 f"{sorted(set(values), key=str)}")
        elif values[0] is None:
            fail(f"monitoring {label} missing on every case")
        else:
            print(f"  monitoring {label}: {values[0]} (identical everywhere)")
    return constant


def check_case_counts(iter_counts: list[dict[tuple[str, int], int]], constant: int | None) -> None:
    """Check 5: exactly CASES_PER_SESSION cases per session, each contributing the shared
    iteration constant exactly once. A multiple of the constant signals a duplicated block
    -- e.g. a retried run whose benchmark rows were appended twice."""
    print("\n=== case-count exactness ===")
    for i, counts in enumerate(iter_counts):
        if len(counts) != CASES_PER_SESSION:
            fail(f"session{i}: {len(counts)} distinct cases (expected {CASES_PER_SESSION})")
        if constant:
            for key in sorted(counts):
                cnt = counts[key]
                if cnt != constant:
                    hint = ""
                    if cnt % constant == 0 and cnt // constant > 1:
                        hint = f" ({cnt // constant}x the {constant}-row block -- duplicated?)"
                    fail(f"session{i} {key[0]}/{key[1]}: {cnt} iteration rows != "
                         f"constant {constant}{hint}")
        print(f"  session{i}: {len(counts)} cases")


def check_retry_bias(retries: list[dict[str, int]]) -> None:
    """Check 4: no case retried more than MAX_RETRIES_PER_CASE times in a session, and no
    case retried in more than MAX_RETRY_SESSIONS of the four sessions -- a case that keeps
    retrying selectively indicts its own environment, not the code under test."""
    print("\n=== retry-frequency bias ===")
    sessions_with_retry: dict[str, int] = {}
    for i, session_retries in enumerate(retries):
        for case, count in session_retries.items():
            if count > MAX_RETRIES_PER_CASE:
                fail(f"session{i} {case}: {count} retries (> {MAX_RETRIES_PER_CASE} allowed)")
            if count >= 1:
                sessions_with_retry[case] = sessions_with_retry.get(case, 0) + 1
    for case in sorted(sessions_with_retry):
        k = sessions_with_retry[case]
        if k > MAX_RETRY_SESSIONS:
            fail(f"{case}: retried in {k} of {len(retries)} sessions "
                 f"(> {MAX_RETRY_SESSIONS}; selective-environment bias)")
        print(f"  {case}: retried in {k} session(s)")
    if not sessions_with_retry:
        print("  no case retried in any session")


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
    iter_counts_per_session: list[dict[tuple[str, int], int]] = []
    interval_values: list = []
    warmup_values: list = []
    retries_per_session: list[dict[str, int]] = []

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
        iter_counts_per_session.append({key: len(vals) for key, vals in reps.items()})
        retries_per_session.append(ctx.get("dw_case_retries", {}))

        ceiling = ctx.get("dw_agg_incase_ceiling")
        unrel = ctx.get("dw_case_unrelated_cpu", {})
        # Recompute count/mean/max from the RAW per-case samples, not the orchestrator's
        # derived fields.
        recomputed_mean = {}
        for case, stats in unrel.items():
            # Collect interval/warmup for the cross-case consistency check (check 2), even
            # for a case whose samples are otherwise broken.
            interval_values.append(stats.get("interval_s"))
            warmup_values.append(stats.get("warmup_s"))
            samples = stats.get("samples")
            if not samples:
                fail(f"session{i} {case}: no raw CPU samples recorded")
                continue
            n_samples = len(samples)
            if n_samples < MIN_CPU_SAMPLES:
                fail(f"session{i} {case}: {n_samples} monitoring samples (< {MIN_CPU_SAMPLES})")
            if ceiling is not None and max(samples) > ceiling:
                fail(f"session{i} {case}: peak aggregate {max(samples)}% exceeded ceiling {ceiling}%")
            if not stats.get("covered_warmup_and_reps"):
                fail(f"session{i} {case}: monitoring did not cover warm-up and repetitions")
            # Check 3: monitoring internal consistency. The recorded n_samples must match the
            # raw list, and the sample count must fit the duration/interval budget: the
            # monitor sleeps interval_s (plus ps-call overhead) per sample, so it can never
            # collect MORE than duration_s/interval_s samples, give or take rounding.
            recorded_n = stats.get("n_samples")
            if recorded_n != n_samples:
                fail(f"session{i} {case}: recorded n_samples {recorded_n} != len(samples) {n_samples}")
            duration_s = stats.get("duration_s")
            interval_s = stats.get("interval_s")
            if duration_s is None or not interval_s:
                fail(f"session{i} {case}: missing duration_s/interval_s for the monitoring-cadence check")
            else:
                budget = duration_s / interval_s + 2
                if n_samples > budget:
                    fail(f"session{i} {case}: n_samples {n_samples} exceeds duration/interval budget "
                         f"{budget:.1f} (duration {duration_s}s / interval {interval_s}s + 2)")
            recomputed_mean[case] = statistics.fmean(samples)
        mean_agg.append(recomputed_mean)
        # Recompute the soak median/MAD from the raw soak samples for the drift check below.
        soak_samples = ctx.get("dw_agg_soak_samples") or []
        rmed = statistics.median(soak_samples) if soak_samples else None
        rmad = mad(soak_samples) if soak_samples else None
        ctx["_recomputed_soak_median"] = rmed
        ctx["_recomputed_soak_mad"] = rmad
        print(f"session{i}: soak median {rmed}% MAD {rmad}% (from {len(soak_samples)} samples) "
              f"ceiling {ceiling}% aborts {ctx.get('dw_case_aborts')} "
              f"commit {str(ctx.get('git_commit'))[:9]}")

    # New acceptance checks: metadata identity, workload/rep/warm-up consistency, retry
    # bias, and case-count exactness -- each fails loudly through the same fail() helper.
    check_metadata_identity(sessions)
    iter_constant = check_workload_consistency(iter_counts_per_session, interval_values, warmup_values)
    check_case_counts(iter_counts_per_session, iter_constant)
    check_retry_bias(retries_per_session)

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

    # Soak-median drift, from the independently recomputed medians/MADs, with the robust
    # per-session bound min(10, max(5, 3*(MAD_0 + MAD_j))) pp.
    rmeds = [s["context"].get("_recomputed_soak_median") for s in sessions]
    rmads = [s["context"].get("_recomputed_soak_mad") for s in sessions]
    print("\n=== soak-median drift (recomputed from raw samples) ===")
    if all(m is not None for m in rmeds) and all(a is not None for a in rmads):
        for j in range(1, n):
            bound = min(SOAK_DRIFT_MAX_PP, max(SOAK_DRIFT_MIN_PP, SOAK_DRIFT_MAD_K * (rmads[0] + rmads[j])))
            drift = abs(rmeds[j] - rmeds[0])
            status = "ok" if drift <= bound else "DRIFT"
            print(f"  s{j}: median {rmeds[j]}% vs s0 {rmeds[0]}% -> drift {drift:.1f}pp "
                  f"(bound {bound:.1f}pp) {status}")
            if drift > bound:
                fail(f"session {j} soak median drift {drift:.1f}pp exceeds bound {bound:.1f}pp")
    else:
        fail("a session is missing recomputable soak samples")

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

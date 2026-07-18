#!/usr/bin/env python3
"""Plot the Monte Carlo scaling benchmark (BENCHMARK-PLAN sections 5 and 13).

Reporting only, not analysis (ADR-002). Every runtime drawn here was measured by
Google Benchmark and read out of its JSON output; this script fits nothing and
decides nothing. It renders the measured median wall-clock time at each thread count
and the two ratios BENCHMARK-PLAN section 5 asks for -- speedup = T_1 / T_p and
efficiency = speedup / p -- which are plain arithmetic on those medians, not a model
fit. The build, CPU, compiler, commit, and load average travel with the figure,
because a benchmark number without its environment is not reproducible.

Usage:
    python3 python/plot_benchmarks.py results/benchmarks/monte_carlo_scaling.baseline.json
    python3 python/plot_benchmarks.py <json> --outdir docs/figures
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # No display in CI.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

ENGINE_STYLE = {
    "european": {"color": "#0969da", "marker": "o"},
    "asian_control_variate": {"color": "#1a7f37", "marker": "s"},
    "heston": {"color": "#cf222e", "marker": "^"},
    "greeks_pathwise_delta": {"color": "#9a6700", "marker": "D"},
    "barrier_bridge": {"color": "#8250df", "marker": "v"},
}


def _style(name: str) -> dict:
    return ENGINE_STYLE.get(name, {"color": "#57606a", "marker": "o"})


def _scale(unit: str) -> float:
    # real_time is reported in the benchmark's own unit; convert to seconds so the axis
    # label and the summary are unambiguous.
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


def _parse(record: dict) -> dict[str, dict[int, dict[str, float]]]:
    """engine -> {threads: {"median": seconds, "cv": fraction}}.

    Google Benchmark names a case `engine/<threads>/real_time`, and with repetitions
    emits per-repetition rows plus `_median`/`_stddev`/`_cv` aggregates. The median is
    the robust central runtime BENCHMARK-PLAN section 3 asks to report, and the cv
    (coefficient of variation, stddev/mean) is its dispersion. Both are read from the
    aggregates, never recomputed here. When the JSON has no repetitions (a smoke run),
    the single `real_time` row is used and the cv is left at zero.
    """
    by_engine: dict[str, dict[int, dict[str, float]]] = collections.defaultdict(dict)
    medians: dict[tuple[str, int], float] = {}
    cvs: dict[tuple[str, int], float] = {}

    for row in record["benchmarks"]:
        name = row["name"]
        base = name.split("/real_time")[0]
        parts = base.split("/")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        key = (parts[0], int(parts[1]))
        run_type = row.get("run_type")
        aggregate = row.get("aggregate_name")

        if run_type == "aggregate" and aggregate == "median":
            medians[key] = row["real_time"] * _scale(row.get("time_unit", "ns"))
        elif run_type == "aggregate" and aggregate == "cv":
            # The cv aggregate is a dimensionless ratio, so no unit scaling.
            cvs[key] = row["real_time"]
        elif run_type != "aggregate" and key not in medians:
            # No-repetitions fallback: a lone timing row is its own "median".
            medians[key] = row["real_time"] * _scale(row.get("time_unit", "ns"))

    for (engine, threads), median in medians.items():
        by_engine[engine][threads] = {"median": median, "cv": cvs.get((engine, threads), 0.0)}
    return by_engine


def _context_caption(context: dict) -> str:
    cpu = context.get("cpu_brand", "unknown CPU")
    cores = context.get("num_cpus", "?")
    topology = context.get("dw_cpu_topology", f"logical={cores}")
    compiler = context.get("compiler", "?")
    build = context.get("build_type", "?")
    commit = context.get("git_commit", "?")[:9]
    power = context.get("dw_power_state", "")
    load_pre = context.get("dw_load_avg_pre") or _one_minute(context.get("load_avg"))
    load_post = context.get("dw_load_avg_post")

    load_note = ""
    if load_pre and load_post:
        load_note = f" · load {load_pre}→{load_post}"
    elif load_pre:
        load_note = f" · load {load_pre}"

    power_note = f" · {power}" if power and power != "unknown" else ""
    return f"{cpu} ({topology}) · {compiler} · {build} · commit {commit}{power_note}{load_note}"


def _one_minute(load_avg) -> str:
    """The 1-minute figure from Google Benchmark's own load_avg list, as a string."""
    if not load_avg:
        return ""
    try:
        return f"{float(str(load_avg).strip('[]').split(',')[0]):g}"
    except (ValueError, TypeError):
        return ""


def plot_scaling(record: dict, outdir: pathlib.Path) -> pathlib.Path:
    by_engine = _parse(record)
    context = record.get("context", {})

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # Panel 1: speedup against thread count.
    ax = axes[0]
    max_p = 1
    for engine in sorted(by_engine):
        series = by_engine[engine]
        threads = sorted(series)
        if 1 not in series:
            continue
        t1 = series[1]["median"]
        speedup = [t1 / series[p]["median"] for p in threads]
        max_p = max(max_p, max(threads))
        ax.plot(threads, speedup, linewidth=1.5, markersize=5, label=engine, **_style(engine))

    ideal = list(range(1, max_p + 1))
    ax.plot(ideal, ideal, "--", color="#57606a", linewidth=1.0, alpha=0.7, label="ideal (linear)")
    ax.set_title("Parallel speedup  T₁ / T_p", fontsize=11, pad=10)
    ax.set_xlabel("threads p", fontsize=9)
    ax.set_ylabel("speedup", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ideal if max_p <= 8 else [1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.95)

    # Panel 2: efficiency = speedup / p.
    ax = axes[1]
    for engine in sorted(by_engine):
        series = by_engine[engine]
        threads = sorted(series)
        if 1 not in series:
            continue
        t1 = series[1]["median"]
        efficiency = [(t1 / series[p]["median"]) / p for p in threads]
        ax.plot(threads, efficiency, linewidth=1.5, markersize=5, label=engine, **_style(engine))
    ax.axhline(1.0, linestyle="--", color="#57606a", linewidth=1.0, alpha=0.7, label="ideal (1.0)")
    ax.set_title("Parallel efficiency  (T₁ / T_p) / p", fontsize=11, pad=10)
    ax.set_xlabel("threads p", fontsize=9)
    ax.set_ylabel("efficiency", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.95)

    topology = context.get("dw_cpu_topology", "")
    heterogeneous = "efficiency=" in topology and "performance=" in topology
    core_note = ""
    if heterogeneous:
        # BENCHMARK-PLAN reading discipline: on an asymmetric CPU the fall in efficiency
        # past the performance-core count is the slower efficiency cores being engaged,
        # not a scaling defect, and it is not linear multicore scaling.
        core_note = (
            "\nThreads beyond the performance-core count engage the slower efficiency "
            "cores; the efficiency fall there is expected, not a defect, and this is NOT "
            "ordinary linear multicore scaling."
        )
    fig.suptitle(
        "Monte Carlo parallel scaling (measured median wall-clock time; speedup and "
        "efficiency are arithmetic on those medians)\n"
        + _context_caption(context)
        + core_note,
        fontsize=9,
        y=1.05,
    )
    fig.tight_layout()
    path = outdir / "bench_monte_carlo_scaling.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(record: dict) -> None:
    """A plain-text summary table (BENCHMARK-PLAN section 13), from the measured medians.

    The cv column is the coefficient of variation (stddev/mean) of the repetitions --
    the run-to-run dispersion. A large cv means the median is not a stable estimate and
    the capture should be repeated on a quieter machine.
    """
    by_engine = _parse(record)
    header = f"{'engine':<24} {'threads':>7} {'median (ms)':>12} {'cv':>7} {'speedup':>8} {'efficiency':>10}"
    print(header)
    for engine in sorted(by_engine):
        series = by_engine[engine]
        t1 = series.get(1, {}).get("median")
        for p in sorted(series):
            median_ms = series[p]["median"] * 1e3
            cv = series[p]["cv"]
            if t1:
                speedup = t1 / series[p]["median"]
                print(
                    f"{engine:<24} {p:>7} {median_ms:>12.3f} {cv:>7.1%} "
                    f"{speedup:>8.2f} {speedup / p:>10.2f}"
                )
            else:
                print(f"{engine:<24} {p:>7} {median_ms:>12.3f} {cv:>7.1%} {'-':>8} {'-':>10}")


def compare_sessions(records: list[tuple[str, dict]]) -> bool:
    """Compare medians across capture sessions; return True if they are stable.

    BENCHMARK-PLAN section 3 / the acceptance rule: a baseline is only trustworthy if it
    reproduces across separate sessions. For each (engine, threads) this prints the
    per-session medians and their relative spread (max-min)/min, and flags any case whose
    spread exceeds 5% -- the threshold above which a before/after optimization delta could
    be session noise rather than a real change.
    """
    parsed = [(label, _parse(rec)) for label, rec in records]
    engines = sorted({e for _, be in parsed for e in be})
    labels = [label for label, _ in parsed]

    print(f"{'engine':<24} {'threads':>7} " + " ".join(f"{label + ' (ms)':>16}" for label in labels) + f"{'spread':>9}")
    stable = True
    for engine in engines:
        thread_set = sorted({t for _, be in parsed for t in be.get(engine, {})})
        for p in thread_set:
            medians_ms = []
            for _, be in parsed:
                cell = be.get(engine, {}).get(p)
                medians_ms.append(cell["median"] * 1e3 if cell else None)
            present = [m for m in medians_ms if m is not None]
            spread = (max(present) - min(present)) / min(present) if len(present) >= 2 and min(present) > 0 else 0.0
            if spread > 0.05:
                stable = False
            cells = " ".join(f"{m:>16.3f}" if m is not None else f"{'-':>16}" for m in medians_ms)
            flag = "  UNSTABLE" if spread > 0.05 else ""
            print(f"{engine:<24} {p:>7} {cells} {spread:>8.1%}{flag}")

    print()
    if stable:
        print("STABLE: all cases agree within 5% across sessions; the baseline is trustworthy.")
    else:
        print("UNSTABLE: session-to-session variation exceeds 5% in the flagged cases.")
        print("Reject this baseline and recapture on a quieter, thermally stable machine.")
    return stable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", nargs="+", type=pathlib.Path, help="Google Benchmark JSON output(s)"
    )
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare medians across the given session JSONs and report stability "
        "(no chart is written)",
    )
    args = parser.parse_args()

    if args.compare:
        if len(args.records) < 2:
            print("--compare needs at least two session JSONs", file=sys.stderr)
            return 2
        loaded = [(path.stem.split(".")[-1], json.loads(path.read_text())) for path in args.records]
        return 0 if compare_sessions(loaded) else 1

    record = json.loads(args.records[0].read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    written = plot_scaling(record, args.outdir)
    print(f"wrote {written}")
    print()
    print_summary(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())

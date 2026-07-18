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


def _parse(record: dict) -> dict[str, dict[int, float]]:
    """engine -> {threads: median wall-clock seconds}, from the median aggregates.

    Google Benchmark names a case `engine/<threads>/real_time`, and with repetitions
    emits per-repetition rows plus `_mean`/`_median`/`_stddev` aggregates. The median
    is the robust central runtime BENCHMARK-PLAN section 3 asks to report; it is read,
    never recomputed here.
    """
    by_engine: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for row in record["benchmarks"]:
        if row.get("run_type") == "aggregate" and row.get("aggregate_name") != "median":
            continue
        name = row["name"]
        # e.g. "european/4/real_time_median" or, without repetitions, "european/4/real_time".
        base = name.split("/real_time")[0]
        parts = base.split("/")
        if len(parts) != 2:
            continue
        engine, threads = parts[0], parts[1]
        if not threads.isdigit():
            continue
        # real_time is reported in the benchmark's unit (ms); convert to seconds via the
        # time_unit field so the axis label is unambiguous.
        unit = row.get("time_unit", "ns")
        scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
        by_engine[engine][int(threads)] = row["real_time"] * scale
    return by_engine


def _context_caption(context: dict) -> str:
    cpu = context.get("cpu_brand", "unknown CPU")
    cores = context.get("num_cpus", "?")
    compiler = context.get("compiler", "?")
    build = context.get("build_type", "?")
    commit = context.get("git_commit", "?")[:9]
    load = context.get("load_avg", None)
    load_note = ""
    if load:
        try:
            one_minute = float(str(load).strip("[]").split(",")[0])
            if one_minute > float(cores):
                load_note = f"  ⚠ load {one_minute:g} > {cores} cores at capture"
        except (ValueError, TypeError):
            pass
    return (
        f"{cpu} ({cores} cores) · {compiler} · {build} · commit {commit}{load_note}"
    )


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
        t1 = series[1]
        speedup = [t1 / series[p] for p in threads]
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
        t1 = series[1]
        efficiency = [(t1 / series[p]) / p for p in threads]
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

    fig.suptitle(
        "Monte Carlo parallel scaling (measured median wall-clock time; speedup and "
        "efficiency are arithmetic on those medians)\n" + _context_caption(context),
        fontsize=9,
        y=1.04,
    )
    fig.tight_layout()
    path = outdir / "bench_monte_carlo_scaling.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(record: dict) -> None:
    """A plain-text summary table (BENCHMARK-PLAN section 13), from the measured medians."""
    by_engine = _parse(record)
    print(f"{'engine':<24} {'threads':>7} {'median (ms)':>12} {'speedup':>8} {'efficiency':>10}")
    for engine in sorted(by_engine):
        series = by_engine[engine]
        t1 = series.get(1)
        for p in sorted(series):
            median_ms = series[p] * 1e3
            if t1:
                speedup = t1 / series[p]
                print(
                    f"{engine:<24} {p:>7} {median_ms:>12.3f} {speedup:>8.2f} {speedup / p:>10.2f}"
                )
            else:
                print(f"{engine:<24} {p:>7} {median_ms:>12.3f} {'-':>8} {'-':>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=pathlib.Path, help="Google Benchmark JSON output")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    written = plot_scaling(record, args.outdir)
    print(f"wrote {written}")
    print()
    print_summary(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())

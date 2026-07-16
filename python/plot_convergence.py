#!/usr/bin/env python3
"""Plot the convergence experiment records (EXP-01 through EXP-04).

Reporting only, not analysis (ADR-002). Every number drawn here was computed by
the C++ engine and read out of the experiment's JSON record; this script fits
nothing, decides nothing, and would not change a verdict if it had a bug. The
fitted slopes, their intervals, and the pass/fail verdicts are read from the
record and rendered, never recomputed -- a plot that re-derived its own slope
could disagree with the one the experiment published, and then a reader would have
no way to tell which was the result.

Usage:
    python3 python/plot_convergence.py results/EXP-02.json --outdir docs/figures

    # or all at once
    python3 python/plot_convergence.py results/*.json --outdir docs/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # No display in CI.
import matplotlib.pyplot as plt  # noqa: E402

# Verdicts, and how they read on a chart. The colours carry meaning: a
# pre-asymptotic study is not a failure, and must not be drawn as one.
VERDICT_COLOR = {
    "consistent": "#1a7f37",
    "consistent_asymptotically": "#9a6700",
    "inconsistent": "#cf222e",
    "noise_dominated": "#8250df",
}

SCHEME_STYLE = {
    "euler_maruyama": {"color": "#0969da", "marker": "o"},
    "milstein": {"color": "#cf222e", "marker": "s"},
    "exact": {"color": "#1a7f37", "marker": "^"},
}


def _style(name: str) -> dict:
    return SCHEME_STYLE.get(name, {"color": "#57606a", "marker": "D"})


def _finish(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.95)


def _reference_line(ax, xs, ys, order: float, label: str) -> None:
    """Draw a slope-`order` guide anchored at the finest point.

    A guide, not a fit. Anchored rather than fitted so it cannot be mistaken for
    the measurement: it shows what the theoretical order looks like on these axes,
    and the reader compares the data against it.
    """
    x0, y0 = xs[-1], ys[-1]
    guide = [y0 * (x / x0) ** order for x in xs]
    ax.plot(xs, guide, "--", color="#57606a", linewidth=1.0, alpha=0.7, label=label, zorder=1)


def plot_strong(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    """EXP-02: strong error against step size, one panel per parameter variation."""
    studies = record["results"]["studies"]
    variations = sorted({s["variation"] for s in studies})

    fig, axes = plt.subplots(1, len(variations), figsize=(5.2 * len(variations), 4.2), squeeze=False)
    written = []

    for ax, variation in zip(axes[0], variations):
        for study in (s for s in studies if s["variation"] == variation):
            xs = [lvl["step_size"] for lvl in study["levels"]]
            ys = [lvl["error"] for lvl in study["levels"]]
            # Error bars from the engine's own standard errors. Absent on analytic
            # levels, which have none -- so they are omitted rather than drawn as
            # zero-height bars implying a measured certainty.
            errs = [lvl.get("error_standard_error", 0.0) for lvl in study["levels"]]

            style = _style(study["scheme"])
            fit = study["asymptotic_fit"]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                label=(
                    f"{study['scheme']}: order {fit['slope']:.3f} "
                    f"[{fit['slope_ci_lower']:.3f}, {fit['slope_ci_upper']:.3f}]"
                ),
                linewidth=1.4,
                markersize=4.5,
                capsize=2,
                **style,
            )
            _reference_line(
                ax, xs, ys, study["theoretical_order"], f"order {study['theoretical_order']:g}"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        _finish(ax, f"Strong convergence — {variation}", "step size Δt", "E|S_T^Δ − S_T^exact|")

    fig.suptitle(
        "EXP-02: strong convergence on common Brownian paths\n"
        "orders and intervals fitted over the asymptotic window; error bars are the engine's "
        "standard errors",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp02_strong_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # The local orders are the evidence that a pre-asymptotic full-range slope is
    # arithmetic rather than a defect, so they get their own panel rather than a
    # footnote.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for study in studies:
        if study["variation"] != variations[0]:
            continue
        orders = study["local_orders"]
        xs = [o["fine_steps"] for o in orders]
        ys = [o["order"] for o in orders]
        style = _style(study["scheme"])
        ax.plot(xs, ys, linewidth=1.4, markersize=4.5, label=study["scheme"], **style)
        ax.axhline(
            study["theoretical_order"],
            linestyle="--",
            color=style["color"],
            alpha=0.45,
            linewidth=1.0,
        )

    ax.set_xscale("log", base=2)
    _finish(
        ax,
        "Local order between adjacent grid levels",
        "steps M (finer →)",
        "local order  log(e_coarse/e_fine) / log(Δ_coarse/Δ_fine)",
    )
    fig.suptitle(
        "EXP-02: the local order climbs toward theory as the grid refines.\n"
        "This is why a full-range fit understates the order: the coarse levels are "
        "pre-asymptotic, not wrong.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp02_local_orders.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def plot_weak(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    """EXP-03: weak error against step size, one panel per test function."""
    studies = record["results"]["studies"]
    functions = ["identity", "square", "call_payoff"]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2), squeeze=False)
    for ax, test_function in zip(axes[0], functions):
        for study in (s for s in studies if s["test_function"] == test_function):
            xs = [lvl["step_size"] for lvl in study["levels"]]
            ys = [lvl["error"] for lvl in study["levels"]]
            style = _style(study["scheme"])
            fit = study["asymptotic_fit"]

            exact = study["method"] == "closed_form_scheme_moment"
            label = f"{study['scheme']}: order {fit['slope']:.3f}"
            if not exact:
                label += f" [{fit['slope_ci_lower']:.2f}, {fit['slope_ci_upper']:.2f}]"

            errs = [lvl.get("error_standard_error", 0.0) for lvl in study["levels"]]
            ax.errorbar(
                xs, ys, yerr=errs, label=label, linewidth=1.4, markersize=4.5, capsize=2, **style
            )
            _reference_line(ax, xs, ys, 1.0, "order 1")

        ax.set_xscale("log")
        ax.set_yscale("log")
        source = "exact (closed-form moment)" if test_function != "call_payoff" else "paired simulation"
        _finish(
            ax,
            f"Weak convergence — f(S) = {test_function}\n{source}",
            "step size Δt",
            "|E[f(S_T^Δ)] − E[f(S_T)]|",
        )

    fig.suptitle(
        "EXP-03: weak convergence. For f(S)=S and f(S)=S² the error is computed in closed form, "
        "not estimated —\nthe two schemes' curves coincide exactly for f(S)=S, because the "
        "Milstein correction has mean zero and cannot move E[S].",
        fontsize=9,
        y=1.04,
    )
    fig.tight_layout()
    path = outdir / "exp03_weak_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_sampling(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    """EXP-01: RMSE against path count, plus confidence coverage."""
    scenarios = record["results"]["scenarios"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    for scenario in scenarios:
        xs = [p["paths"] for p in scenario["points"]]
        ys = [p["rmse"] for p in scenario["points"]]
        slope = scenario["study"]["slope_versus_paths"]
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.2, label=f"{scenario['scenario']}: {slope:.3f}")
    _reference_line(ax, xs, ys, -0.5, "slope −1/2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish(ax, "RMSE across seeds against path count", "paths N", "RMSE vs analytic")

    ax = axes[1]
    for scenario in scenarios:
        xs = [p["paths"] for p in scenario["points"]]
        ys = [p["coverage"] for p in scenario["points"]]
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.2, label=scenario["scenario"])
    ax.axhline(0.95, linestyle="--", color="#24292f", linewidth=1.0, label="nominal 0.95")
    ax.set_xscale("log")
    ax.set_ylim(0.7, 1.02)
    _finish(ax, "Confidence coverage of the analytic value", "paths N", "fraction of 95% intervals covering")

    fig.suptitle(
        "EXP-01: Monte Carlo sampling convergence. RMSE is taken across independent seeds, "
        "not from a single run —\na single run's error is one draw from a distribution centred "
        "near zero and can be small by luck.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp01_sampling_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_bias_variance(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    """EXP-04: RMSE against paths at each step count, showing the bias floor."""
    cells = record["results"]["cells"]
    schemes = sorted({c["scheme"] for c in cells})

    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 4.4), squeeze=False)
    for ax, scheme in zip(axes[0], schemes):
        rows = [c for c in cells if c["scheme"] == scheme]
        for steps in sorted({c["steps"] for c in rows}):
            series = sorted((c for c in rows if c["steps"] == steps), key=lambda c: c["paths"])
            xs = [c["paths"] for c in series]
            ys = [c["rmse"] for c in series]
            ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.3, label=f"M = {steps}")

            # The floor each curve is flattening onto: |bias| at that step count.
            # Drawing it makes the plateau legible as a bias rather than as noise.
            bias = abs(series[-1]["bias"])
            if bias > 0:
                ax.axhline(bias, linestyle=":", linewidth=0.8, alpha=0.4, color="#57606a")

        ax.set_xscale("log")
        ax.set_yscale("log")
        _finish(ax, f"RMSE against path count — {scheme}", "paths N", "RMSE vs analytic")

    fig.suptitle(
        "EXP-04: sampling error versus discretisation bias. At coarse M the RMSE flattens onto a "
        "floor set by the bias (dotted).\nPast that point more paths buy a narrower interval "
        "around the wrong number — and the interval keeps looking healthy.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp04_bias_variance.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


PLOTTERS = {
    "EXP-01": plot_sampling,
    "EXP-02": plot_strong,
    "EXP-03": plot_weak,
    "EXP-04": plot_bias_variance,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=pathlib.Path, help="experiment JSON records")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.records:
        record = json.loads(path.read_text())
        experiment_id = record["id"]
        plotter = PLOTTERS.get(experiment_id)
        if plotter is None:
            print(f"{path}: no plotter for {experiment_id}, skipping", file=sys.stderr)
            continue

        for written in plotter(record, args.outdir):
            print(f"wrote {written}")

        # The status travels with the figure. A chart of a failed experiment that
        # does not say so is how a contradictory result quietly becomes a headline.
        status = record["status"]
        if status != "pass":
            print(f"  note: {experiment_id} status is {status.upper()}", file=sys.stderr)
            for limitation in record["limitations"]:
                print(f"    - {limitation}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

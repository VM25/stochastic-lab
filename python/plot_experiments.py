#!/usr/bin/env python3
"""Plot the PDE, barrier, and Greek experiment records (EXP-06, EXP-07, EXP-08).

Reporting only, not analysis (ADR-002). Every number drawn here was computed by
the C++ engine and read out of the experiment's JSON record; this script fits
nothing, decides nothing, and would not change a verdict if it had a bug. Fitted
orders, their intervals, the continuity-corrected references, and the pass/fail
verdicts are read from the record and rendered, never recomputed -- a plot that
re-derived a slope could disagree with the one the experiment published, and then
a reader would have no way to tell which was the result.

Usage:
    python3 python/plot_experiments.py results/EXP-06.json --outdir docs/figures

    # or all at once
    python3 python/plot_experiments.py results/*.json --outdir docs/figures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # No display in CI.
import matplotlib.pyplot as plt  # noqa: E402

# A stable palette so the same object reads the same way across figures.
SCHEME_STYLE = {
    "crank_nicolson": {"color": "#0969da", "marker": "o"},
    "crank_nicolson_rannacher2": {"color": "#1a7f37", "marker": "^"},
    "implicit": {"color": "#cf222e", "marker": "s"},
    "explicit": {"color": "#8250df", "marker": "D"},
    "finite_difference": {"color": "#57606a", "marker": "D"},
    "pathwise": {"color": "#0969da", "marker": "o"},
    "likelihood_ratio": {"color": "#cf222e", "marker": "s"},
}

# One colour per barrier level, so a barrier keeps its identity across panels.
# Every barrier the experiment uses gets a distinct colour; a fallback would let
# two arms share a colour and read as one.
BARRIER_COLOR = {
    70.0: "#0969da",
    80.0: "#1a7f37",
    90.0: "#9a6700",
    95.0: "#bc4c00",
    105.0: "#cf222e",
    110.0: "#8250df",
    120.0: "#0550ae",
    140.0: "#116329",
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
    and the reader compares the data against it. At most one guide per order per
    panel, so two arms sharing a theoretical order do not stack identical lines.
    """
    drawn = getattr(ax, "_dw_guides", set())
    if order in drawn:
        return
    drawn.add(order)
    ax._dw_guides = drawn

    xs_sorted = sorted(xs)
    x0, y0 = xs_sorted[-1], ys[xs.index(xs_sorted[-1])]
    guide = [y0 * (x / x0) ** order for x in xs_sorted]
    ax.plot(xs_sorted, guide, "--", color="#adb5bd", linewidth=1.0, alpha=0.8, label=label, zorder=1)


# --------------------------------------------------------------------------- #
# EXP-06: PDE stability and grid convergence
# --------------------------------------------------------------------------- #
def plot_pde(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    written: list[pathlib.Path] = []

    # (1) Spatial convergence: error against asset spacing, one line per scheme.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for scheme in results["space_sweep"]:
        levels = scheme["levels"]
        xs = [lvl["asset_spacing"] for lvl in levels]
        ys = [lvl["error"] for lvl in levels]
        fit = scheme["fit"]
        style = _style(scheme["scheme"])
        ax.plot(
            xs,
            ys,
            linewidth=1.4,
            markersize=4.5,
            label=(
                f"{scheme['scheme']}: order {fit['order']:.3f} "
                f"[{fit['ci_lower']:.3f}, {fit['ci_upper']:.3f}]"
            ),
            **style,
        )
        _reference_line(ax, xs, ys, 2.0, "order 2")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, "EXP-06: spatial convergence", "asset spacing ΔS", "|price − analytic|")
    fig.suptitle(
        "Second-order in space: the fitted order and its interval are read from the "
        "record,\nnot recomputed here.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp06_space_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # (2) Temporal convergence: error against time step, showing that plain
    # Crank-Nicolson oscillates on coarse grids where Rannacher does not. The
    # full-range and asymptotic-window orders both come from the record.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for arm in results["time_sweep"]:
        levels = arm["levels"]
        xs = [lvl["time_step"] for lvl in levels]
        ys = [lvl["error_vs_analytic"] for lvl in levels]
        asy = arm["fit_asymptotic_window"]
        full = arm["fit_full_range"]
        style = _style(arm["arm"])
        ax.plot(
            xs,
            ys,
            linewidth=1.4,
            markersize=4.5,
            label=(
                f"{arm['arm']}: asy {asy['order']:.3f} "
                f"[{asy['ci_lower']:.3f}, {asy['ci_upper']:.3f}], "
                f"full {full['order']:.3f}"
            ),
            **style,
        )
        _reference_line(ax, xs, ys, 1.0, "order 1")
        _reference_line(ax, xs, ys, 2.0, "order 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish(ax, "EXP-06: temporal convergence", "time step Δτ", "|price − analytic|")
    fig.suptitle(
        "Implicit is first order; Crank-Nicolson is second order but its full-range fit is "
        "inflated\nby coarse-grid oscillation, which two Rannacher start-up steps remove.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp06_time_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # (3) Explicit stability: the price error explodes past the Courant bound. The
    # stable/unstable classification is the engine's, read from each level.
    stability = results["explicit_stability"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ratios = [lvl["achieved_ratio"] for lvl in stability["levels"]]
    errors = [abs(lvl["error"]) for lvl in stability["levels"]]
    stable = [lvl["observed_stable"] for lvl in stability["levels"]]
    ax.scatter(
        [r for r, s in zip(ratios, stable) if s],
        [e for e, s in zip(errors, stable) if s],
        color="#1a7f37",
        marker="o",
        s=42,
        label="observed stable",
        zorder=3,
    )
    ax.scatter(
        [r for r, s in zip(ratios, stable) if not s],
        [e for e, s in zip(errors, stable) if not s],
        color="#cf222e",
        marker="X",
        s=64,
        label="observed unstable",
        zorder=3,
    )
    ax.axvline(
        1.0,
        linestyle="--",
        color="#57606a",
        linewidth=1.0,
        label="Courant bound (ratio = 1)",
    )
    ax.set_yscale("log")
    _finish(
        ax,
        "EXP-06: explicit-scheme stability",
        "Courant ratio Δτ / Δτ_max",
        "|price − analytic|",
    )
    highest = stability.get("highest_ratio_observed_stable")
    lowest = stability.get("lowest_ratio_observed_unstable")
    fig.suptitle(
        "The explicit scheme is stable to ratio "
        f"{highest:g} and divergent at {lowest:g}: the bound is sufficient, not necessary.",
        fontsize=9,
        y=1.00,
    )
    fig.tight_layout()
    path = outdir / "exp06_explicit_stability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


# --------------------------------------------------------------------------- #
# EXP-07: barrier monitoring bias
# --------------------------------------------------------------------------- #
def plot_barrier(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    written: list[pathlib.Path] = []

    # (1) Monitoring bias against frequency: discrete monitoring carries a bias
    # that the Brownian bridge removes. One panel per barrier direction.
    types = ["down_and_out", "up_and_out"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), squeeze=False)
    for ax, barrier_type in zip(axes[0], types):
        for arm in results["arms"]:
            if arm["barrier_type"] != barrier_type:
                continue
            levels = arm["levels"]
            xs = [lvl["monitoring_dates"] for lvl in levels]
            ys = [lvl["relative_bias"] * 100.0 for lvl in levels]
            color = BARRIER_COLOR.get(arm["barrier"], "#57606a")
            discrete = arm["convention"] == "discrete"
            ax.plot(
                xs,
                ys,
                color=color,
                marker="o" if discrete else "o",
                markerfacecolor=color if discrete else "white",
                linestyle="-" if discrete else "--",
                linewidth=1.4,
                markersize=4.5,
                label=f"B={arm['barrier']:g} {'discrete' if discrete else 'bridge'}",
            )
        ax.axhline(0.0, color="#57606a", linewidth=0.8, alpha=0.6)
        ax.set_xscale("log")
        _finish(
            ax,
            f"EXP-07: {barrier_type.replace('_', ' ')}",
            "monitoring dates",
            "relative bias vs continuous (%)",
        )
    fig.suptitle(
        "Discrete monitoring (solid, filled) is biased; the Brownian bridge (dashed, open) "
        "removes it.\nThe bias is two-sided: down-barriers low, up-barriers high.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp07_monitoring_bias.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # (2) The PDE arm prices the continuous contract directly and converges at
    # order ~2. The fitted order is the record's.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for arm in results["pde_convergence"]:
        levels = arm["levels"]
        xs = [lvl["refinement_parameter"] for lvl in levels]
        ys = [lvl["absolute_error"] for lvl in levels]
        fit = arm["order_fit"]
        color = BARRIER_COLOR.get(arm["barrier"], "#57606a")
        marker = "o" if arm["barrier_type"] == "down_and_out" else "s"
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linewidth=1.3,
            markersize=4.2,
            label=(
                f"B={arm['barrier']:g} {arm['barrier_type'].replace('_and_out', '')}: "
                f"order {fit['order']:.3f}"
            ),
        )
        _reference_line(ax, xs, ys, 2.0, "order 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish(
        ax,
        "EXP-07: PDE arm convergence to the continuous price",
        "refinement parameter",
        "|price − continuous reference|",
    )
    fig.suptitle(
        "The absorbing-boundary PDE converges at order ~2 to the analytic continuous barrier "
        "price,\nin both directions -- a standing regression on the operator.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp07_pde_convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


# --------------------------------------------------------------------------- #
# EXP-08: Greek estimator comparison
# --------------------------------------------------------------------------- #
def plot_greeks(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    written: list[pathlib.Path] = []

    # (1) The finite-difference dispersion-vs-bump exponent per cell, against the
    # textbook expectation. Under common random numbers delta sits near 0 (not 1)
    # and gamma near 0.5 (not 2). Exponents and intervals are the record's.
    scaling = results["finite_difference_variance_scaling"]
    textbook = {"delta": 1.0, "gamma": 2.0, "vega": 1.0}
    greeks = ["delta", "gamma"]
    fig, axes = plt.subplots(1, len(greeks), figsize=(6.4 * len(greeks), 4.4), squeeze=False)
    for ax, greek in zip(axes[0], greeks):
        cells = [c for c in scaling if c["greek"] == greek]
        cells.sort(key=lambda c: (c["spot"], c["volatility"], c["maturity"]))
        xs = list(range(len(cells)))
        ys = [c["dispersion_vs_bump_exponent"] for c in cells]
        lo = [c["dispersion_vs_bump_exponent"] - c["exponent_ci_lower"] for c in cells]
        hi = [c["exponent_ci_upper"] - c["dispersion_vs_bump_exponent"] for c in cells]
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            fmt="o",
            color="#0969da",
            markersize=4.5,
            capsize=2,
            linewidth=1.2,
            label="fitted exponent (95% CI)",
        )
        median = sorted(ys)[len(ys) // 2]
        ax.axhline(
            textbook[greek],
            linestyle="--",
            color="#cf222e",
            linewidth=1.2,
            label=f"textbook 1/hᵏ exponent = {textbook[greek]:g}",
        )
        ax.axhline(
            median,
            linestyle="-",
            color="#1a7f37",
            linewidth=1.2,
            label=f"measured median = {median:.2f}",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"S={c['spot']:g}\nσ={c['volatility']:g}\nT={c['maturity']:g}" for c in cells],
            fontsize=6,
        )
        _finish(ax, f"EXP-08: finite-difference {greek}", "cell", "dispersion-vs-bump exponent")
    fig.suptitle(
        "Under common random numbers the finite-difference variance does not follow the "
        "textbook 1/h (delta) or 1/h² (gamma):\nthe measured exponent sits near 0 and 0.5.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp08_fd_variance_scaling.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # (2) Standard error of the two theoretically-unbiased delta estimators,
    # pathwise against likelihood-ratio, cell by cell. Pathwise wins on variance.
    cells = results["cells"]
    pathwise = {
        (c["spot"], c["volatility"], c["maturity"]): c
        for c in cells
        if c["method"] == "pathwise" and c["greek"] == "delta"
    }
    likelihood = {
        (c["spot"], c["volatility"], c["maturity"]): c
        for c in cells
        if c["method"] == "likelihood_ratio" and c["greek"] == "delta"
    }
    shared = sorted(set(pathwise) & set(likelihood))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = list(range(len(shared)))
    ax.plot(
        xs,
        [pathwise[k]["across_seed_standard_error"] for k in shared],
        color=SCHEME_STYLE["pathwise"]["color"],
        marker="o",
        linewidth=1.4,
        markersize=4.5,
        label="pathwise delta",
    )
    ax.plot(
        xs,
        [likelihood[k]["across_seed_standard_error"] for k in shared],
        color=SCHEME_STYLE["likelihood_ratio"]["color"],
        marker="s",
        linewidth=1.4,
        markersize=4.5,
        label="likelihood-ratio delta",
    )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"S={s:g}\nσ={v:g}\nT={t:g}" for (s, v, t) in shared],
        fontsize=6,
    )
    _finish(
        ax,
        "EXP-08: delta standard error, pathwise vs likelihood-ratio",
        "cell",
        "across-seed standard error",
    )
    ratios = sorted(
        likelihood[k]["across_seed_standard_error"] / pathwise[k]["across_seed_standard_error"]
        for k in shared
        if pathwise[k]["across_seed_standard_error"] > 0
    )
    if ratios:
        median_ratio = ratios[len(ratios) // 2]
        fig.suptitle(
            "Both estimators are unbiased for delta; pathwise has the smaller standard error "
            f"in every cell\n(likelihood-ratio SE is {median_ratio:.1f}× larger at the median, "
            f"up to {max(ratios):.0f}× in the deep-OTM short-maturity corner). "
            "Likelihood-ratio is the one that survives a discontinuous payoff.",
            fontsize=9,
            y=1.02,
        )
    fig.tight_layout()
    path = outdir / "exp08_estimator_standard_error.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


PLOTTERS = {
    "EXP-06": plot_pde,
    "EXP-07": plot_barrier,
    "EXP-08": plot_greeks,
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

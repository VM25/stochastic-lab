#!/usr/bin/env python3
"""Plot every experiment record outside the convergence quartet (EXP-05 through EXP-15).

`plot_convergence.py` owns EXP-01 to EXP-04; this owns the rest. Each script
renders the ids it recognises and skips the others, so both are run over the whole
record set.

Reporting only, not analysis (ADR-002). Every number drawn here was computed by
the C++ engine and read out of the experiment's JSON record; this script fits
nothing, decides nothing, and would not change a verdict if it had a bug. Fitted
orders, their intervals, the continuity-corrected references, the coverage
deviations, and the pass/fail verdicts are read from the record and rendered, never
recomputed -- a plot that re-derived a slope could disagree with the one the
experiment published, and then a reader would have no way to tell which was the
result.

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


# --------------------------------------------------------------------------- #
# EXP-05: variance-reduction efficiency
# --------------------------------------------------------------------------- #
ESTIMATOR_STYLE = {
    "crude": {"color": "#57606a", "marker": "o"},
    "antithetic": {"color": "#0969da", "marker": "s"},
    "control_variate": {"color": "#1a7f37", "marker": "^"},
    "combined": {"color": "#8250df", "marker": "D"},
}


def plot_variance_reduction(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    cells = record["results"]["cells"]
    instruments = sorted({c["instrument"] for c in cells})

    fig, axes = plt.subplots(1, len(instruments), figsize=(6.6 * len(instruments), 4.6), squeeze=False)
    for ax, instrument in zip(axes[0], instruments):
        subset = [c for c in cells if c["instrument"] == instrument]
        keys = sorted({(c["spot"], c["volatility"]) for c in subset})
        for estimator in ["crude", "antithetic", "control_variate", "combined"]:
            by_key = {
                (c["spot"], c["volatility"]): c for c in subset if c["estimator"] == estimator
            }
            if not by_key:
                continue
            xs = [i for i, k in enumerate(keys) if k in by_key]
            ys = [by_key[keys[i]]["work_normalised_efficiency_gain_over_crude"] for i in xs]
            ax.plot(
                xs,
                ys,
                linewidth=1.4,
                markersize=5,
                label=estimator,
                **ESTIMATOR_STYLE.get(estimator, {"color": "#57606a", "marker": "x"}),
            )
        ax.axhline(1.0, linestyle="--", color="#57606a", linewidth=1.0, label="crude baseline")
        ax.set_yscale("log")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"S={s:g}\nσ={v:g}" for (s, v) in keys], fontsize=7)
        _finish(
            ax,
            f"EXP-05: {instrument.replace('_', ' ')}",
            "cell",
            "work-normalised efficiency gain over crude",
        )
    fig.suptitle(
        "Efficiency is 1/(variance × deterministic work units), not 1/(variance × runtime):\n"
        "the ranking is a property of the estimators, not of the machine the run happened on.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp05_work_normalised_efficiency.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-09: Heston characteristic-function validation
# --------------------------------------------------------------------------- #
def plot_heston_cf(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), squeeze=False)

    # (a) Quadrature convergence: an internal numerical check, not an oracle.
    ax = axes[0][0]
    levels = results["internal_numerical_checks"]["integration_convergence"]
    xs = [lvl["quadrature_nodes"] for lvl in levels]
    ys = [max(lvl["integration_error"], 1e-18) for lvl in levels]
    ax.plot(xs, ys, color="#0969da", marker="o", linewidth=1.4, markersize=4.5, label="integration error")
    ax.axhline(2.2e-16, linestyle="--", color="#57606a", linewidth=1.0, label="double precision ~2.2e-16")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(ax, "EXP-09: quadrature convergence (internal check)", "quadrature nodes", "|price − converged price|")

    # (b) External references: relative error against each case's own tolerance.
    ax = axes[0][1]
    refs = results["external_references"]
    provenance_color = {"published": "#1a7f37", "independently_generated": "#0969da"}
    xs = list(range(len(refs)))
    for i, ref in enumerate(refs):
        ax.scatter(
            [i],
            [max(ref["relative_error"], 1e-18)],
            color=provenance_color.get(ref["provenance_category"], "#57606a"),
            marker="o" if ref["passes"] else "X",
            s=54,
            zorder=3,
            label=ref["provenance_category"] if i == 0 or ref["provenance_category"] not in
            [r["provenance_category"] for r in refs[:i]] else None,
        )
        ax.plot([i - 0.3, i + 0.3], [ref["tolerance"]] * 2, color="#cf222e", linewidth=1.4,
                label="tolerance" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["case"] for r in refs], fontsize=6.5, rotation=20, ha="right")
    _finish(ax, "EXP-09: agreement with external references", "reference case", "relative error")

    fig.suptitle(
        "The evidence is kept in separate categories: quadrature convergence is an internal "
        "numerical check,\nwhile the reference cases are published or independently generated "
        "values. They are not all independent oracles.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp09_cf_validation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-10: Heston variance discretization
# --------------------------------------------------------------------------- #
def plot_heston_discretization(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    regimes = results["regimes"]
    fits = {f["regime"]: f for f in results["bias_decay_order_fits"]}

    fig, axes = plt.subplots(1, len(regimes), figsize=(6.6 * len(regimes), 4.6), squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        cells = regime["cells"]
        for scheme in sorted({c["scheme"] for c in cells}):
            priced = [c for c in cells if c["scheme"] == scheme and c.get("priced")]
            if not priced:
                # A scheme that produced no price anywhere is the finding, not a
                # gap: it is recorded in the caption rather than drawn as absent.
                continue
            priced.sort(key=lambda c: c["steps"])
            xs = [c["steps"] for c in priced]
            ys = [abs(c["bias"]) for c in priced]
            errs = [c["across_seed_standard_error"] for c in priced]
            ax.errorbar(
                xs, ys, yerr=errs, linewidth=1.4, markersize=4.5, capsize=2,
                label=scheme, **_style(scheme if scheme in SCHEME_STYLE else "implicit"),
            )
            # Unresolved levels are marked so a reader does not read noise as bias.
            unresolved = [(c["steps"], abs(c["bias"])) for c in priced if not c["bias_is_resolved"]]
            if unresolved:
                ax.scatter(
                    [u[0] for u in unresolved], [u[1] for u in unresolved],
                    facecolors="white", edgecolors="#57606a", s=64, zorder=4,
                    label="bias not resolved above noise",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        fit = fits.get(regime["regime"], {})
        order = fit.get("bias_decay_order")
        order_text = f"decay order {order:.2f}" if order is not None else "decay order unresolved"
        _finish(
            ax,
            f"EXP-10: {regime['regime'].replace('_', ' ')} (ξ={regime['xi']:g}) — {order_text}",
            "time steps",
            "|bias vs semi-analytic reference|",
        )

    failed = [
        scheme
        for regime in regimes
        for scheme in sorted({c["scheme"] for c in regime["cells"]})
        if not any(c["scheme"] == scheme and c.get("priced") for c in regime["cells"])
    ]
    note = (
        f"The {sorted(set(failed))[0].replace('_', ' ')} scheme produced no price at any step "
        "count tested, in either regime, and so has no curve to draw."
        if failed
        else "Every scheme tested produced a price at every step count."
    )
    fig.suptitle(
        f"Full truncation prices every regime without a non-finite path. {note}",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp10_variance_discretization.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-11: Heston calibration recovery
# --------------------------------------------------------------------------- #
def plot_calibration_recovery(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    starts = record["results"]["starts"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    xs = list(range(len(starts)))
    for i, start in enumerate(starts):
        blind = start.get("blind", False)
        ax.bar(
            i,
            max(start["distance_to_truth"], 1e-18),
            color="#0969da" if blind else "#9a6700",
            width=0.6,
            label=("blind start" if blind else "start at the truth")
            if i == 0 or (blind != starts[i - 1].get("blind", False))
            else None,
        )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"start {s['index']}\n{'blind' if s.get('blind') else 'seeded'}" for s in starts],
        fontsize=7,
    )
    _finish(
        ax,
        "EXP-11: parameter recovery from a synthetic surface",
        "optimiser start",
        "normalised distance to the generating parameters",
    )
    fig.suptitle(
        "The headline recovery is read from a blind start -- one that did not begin at the "
        "answer.\nA seeded start recovering the truth would prove nothing about calibration.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp11_recovery.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-12: market-surface calibration stability
# --------------------------------------------------------------------------- #
def plot_calibration_stability(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    scenarios = results["scenarios"]
    dispersion = results["parameter_dispersion"]
    written: list[pathlib.Path] = []

    # (1) The dispersion is the finding: every scenario fits, and the parameters
    # still disagree by more than an order of magnitude on two of five axes.
    names = list(dispersion.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        values = [s["calibrated"][name] for s in scenarios]
        ax.plot(range(len(values)), values, "o", color="#0969da", markersize=5)
        stats = dispersion[name]
        ax.axhline(stats["mean"], linestyle="--", color="#cf222e", linewidth=1.1, label="mean")
        relative = stats["stddev"] / abs(stats["mean"]) if stats["mean"] else float("nan")
        ax.set_title(f"{name.replace('_', ' ')}\nrel. sd {relative:.0%}", fontsize=9, pad=8)
        ax.set_xlabel("scenario", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(values)))
    fig.suptitle(
        "EXP-12: the same real surface, seven scenarios, all converged and all fitting well.\n"
        "The short-end parameters are determined; mean reversion and long-run variance are not.",
        fontsize=9,
        y=1.04,
    )
    fig.tight_layout()
    path = outdir / "exp12_parameter_dispersion.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # (2) Where the model tracks the smile and where it cannot. This surface is not
    # generated by Heston, so the residual is genuine model error.
    base = scenarios[0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    maturities = sorted({q["maturity"] for q in base["residual_surface"]})
    palette = ["#0969da", "#1a7f37", "#cf222e", "#8250df", "#9a6700"]
    for i, maturity in enumerate(maturities):
        quotes = sorted(
            (q for q in base["residual_surface"] if q["maturity"] == maturity),
            key=lambda q: q["strike"],
        )
        ax.plot(
            [q["strike"] for q in quotes],
            [q["iv_residual"] for q in quotes],
            marker="o",
            linewidth=1.4,
            markersize=4.5,
            color=palette[i % len(palette)],
            label=f"T = {maturity:.3g}y",
        )
    ax.axhline(0.0, color="#57606a", linewidth=0.9, alpha=0.7)
    _finish(
        ax,
        "EXP-12: implied-volatility residual by strike (base scenario)",
        "strike",
        "model − market implied volatility",
    )
    fig.suptitle(
        "A real surface is not a Heston surface: the residual is model error, and it is "
        "reported\nby strike and maturity rather than summarised into a single RMSE.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp12_residual_surface.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


# --------------------------------------------------------------------------- #
# EXP-13: cross-method accuracy and agreement
# --------------------------------------------------------------------------- #
def plot_cross_method(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    labels: list[str] = []
    sigmas: list[float] = []
    colors: list[str] = []

    for cell in results["black_scholes_european"]:
        for method in cell["methods"]:
            if method.get("is_reference") or "sigmas" not in method:
                continue
            labels.append(f"BS {method['method']}\nT={cell['maturity']:g} σ={cell['volatility']:g}")
            sigmas.append(abs(method["sigmas"]))
            colors.append("#0969da" if method.get("agrees") else "#cf222e")

    for cell in results["heston_european"]:
        labels.append(f"Heston MC vs CF\nT={cell['maturity']:g} ξ={cell['vol_of_variance']:g}")
        sigmas.append(abs(cell["sigmas"]))
        colors.append("#1a7f37" if cell.get("agrees") else "#cf222e")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels)), 4.8))
    ax.bar(range(len(sigmas)), sigmas, color=colors, width=0.65)
    ax.axhline(5.0, linestyle="--", color="#cf222e", linewidth=1.2, label="agreement gate (5σ)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    _finish(
        ax,
        "EXP-13: cross-method agreement",
        "comparison",
        "|difference| in combined standard errors (σ)",
    )
    asian_agree = sum(1 for c in results["arithmetic_asian"] if c.get("all_agree"))
    fig.suptitle(
        "Methods are compared against each other in units of their own combined standard error, "
        "so a\ndisagreement is judged against sampling noise rather than an absolute tolerance. "
        f"All {asian_agree} of {len(results['arithmetic_asian'])} Asian cells agree pairwise.",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / "exp13_agreement.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-14: statistical confidence coverage
# --------------------------------------------------------------------------- #
def plot_coverage(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    results = record["results"]
    cells = sorted(results["cells"], key=lambda c: (c["moneyness"], c["sample_size"]))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = list(range(len(cells)))
    ys = [c["observed_coverage"] for c in cells]
    errs = [c["coverage_standard_error"] for c in cells]
    colors = ["#cf222e" if c["under_covers"] else "#1a7f37" for c in cells]
    for x, y, e, color in zip(xs, ys, errs, colors):
        ax.errorbar([x], [y], yerr=[e], fmt="o", color=color, markersize=6, capsize=3, linewidth=1.3)
    nominal = cells[0]["nominal_coverage"]
    ax.axhline(nominal, linestyle="--", color="#57606a", linewidth=1.2, label=f"nominal {nominal:.0%}")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"N={c['sample_size']:,}\nK/S={c['moneyness']:g}\nskew {c['payoff_skewness']:.1f}" for c in cells],
        fontsize=6.5,
    )
    _finish(
        ax,
        "EXP-14: observed coverage of the nominal 95% interval",
        "cell",
        "observed coverage",
    )
    worst = max(cells, key=lambda c: c["deviation_sigmas"])
    fig.suptitle(
        "Red is a resolved under-coverage. The interval is not automatically valid: at "
        f"K/S={worst['moneyness']:g} with payoff skewness {worst['payoff_skewness']:.1f}, "
        f"N={worst['sample_size']:,} covers only\n{worst['observed_coverage']:.2%} "
        f"({worst['deviation_sigmas']:.1f}σ below nominal). More paths fix it — the central "
        "limit theorem needs enough of them.",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    path = outdir / "exp14_coverage.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# EXP-15: numerical edge cases
# --------------------------------------------------------------------------- #
def plot_edge_cases(record: dict, outdir: pathlib.Path) -> list[pathlib.Path]:
    cases = record["results"]["cases"]
    order = sorted({c["category"] for c in cases})
    grouped = [c for category in order for c in cases if c["category"] == category]

    category_color = {
        "limiting_behavior": "#0969da",
        "already_breached": "#1a7f37",
        "degenerate_refusal": "#8250df",
        "invalid_input_rejected": "#9a6700",
    }

    fig, ax = plt.subplots(figsize=(8.4, 0.30 * len(grouped) + 2.0))
    ys = list(range(len(grouped)))
    ax.barh(
        ys,
        [1.0 if c["passed"] else 0.0 for c in grouped],
        color=[category_color.get(c["category"], "#57606a") for c in grouped],
        height=0.68,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c['case']}" for c in grouped], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not resolved", "behaved as required"], fontsize=8)
    for y, case in zip(ys, grouped):
        ax.text(1.02, y, case["behavior"], va="center", fontsize=6, color="#57606a")
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=6, label=category.replace("_", " "))
        for category, color in category_color.items()
        if any(c["category"] == category for c in grouped)
    ]
    ax.legend(handles=handles, fontsize=7, framealpha=0.95, loc="lower right")
    ax.set_title("EXP-15: every edge case either produces the correct limit or refuses", fontsize=11, pad=10)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
    summary = record["results"]["summary"]
    fig.suptitle(
        f"{summary['passed']} of {summary['total_cases']} cases behaved as required; "
        f"non-finite value escaped: {str(summary['any_non_finite_escaped']).lower()}.\n"
        "Refusing is a correct outcome here -- a plausible number from a degenerate input "
        "would be the failure.",
        fontsize=9,
        y=1.005,
    )
    fig.tight_layout()
    path = outdir / "exp15_edge_cases.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


PLOTTERS = {
    "EXP-05": plot_variance_reduction,
    "EXP-06": plot_pde,
    "EXP-07": plot_barrier,
    "EXP-08": plot_greeks,
    "EXP-09": plot_heston_cf,
    "EXP-10": plot_heston_discretization,
    "EXP-11": plot_calibration_recovery,
    "EXP-12": plot_calibration_stability,
    "EXP-13": plot_cross_method,
    "EXP-14": plot_coverage,
    "EXP-15": plot_edge_cases,
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

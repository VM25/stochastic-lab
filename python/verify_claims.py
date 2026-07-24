#!/usr/bin/env python3
"""Re-derive every headline claim in the prose from the committed records.

Reporting and checking only (ADR-002). Each claim below is recomputed from the JSON
records and then required to appear *verbatim* in the prose. A number that drifts
when the experiments are regenerated stops being a description of the artifact and
becomes a leftover, and prose is where that happens silently -- so this makes it a
build failure instead.

Two documents are checked, with two claim sets:

* `results/README.md` -- the reconciliation. It must carry every **shared** claim: the
  headline number for each experiment.
* `docs/TECHNICAL-PAPER.md` -- the paper. It must carry the shared claims *and* the
  finer-grained ones it alone states (per-scheme weak orders, regime counts, the
  calibration dispersion table, cross-method agreement in sigmas).

The README is deliberately not required to carry the paper's detail; it is a summary,
and forcing the two to state the same things would make one of them redundant.

The check is deliberately literal: it does not parse the prose or try to understand
it, it asserts that the exact string the record implies is present. That catches the
case that matters (the record moved, the sentence did not) and is immune to being
fooled by a rewording, which would simply be reported as a missing claim.

This found a real drift on its first run: EXP-07's PDE arm had six of eight cells
land on order 2.000 in an earlier generation and five of eight in the current one,
while the prose still said six.

Usage:
    python3 python/verify_claims.py
    python3 python/verify_claims.py --results results
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def paper_claims(records: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Claims the technical paper states and the reconciliation summary does not."""
    out: list[tuple[str, str, str]] = []

    def add(exp: str, what: str, text: str) -> None:
        out.append((exp, what, text))

    # EXP-03 -- weak order per test function, including the non-smooth payoff.
    for study in records["EXP-03"]["results"]["studies"]:
        add(
            "EXP-03",
            f"{study['test_function']} / {study['scheme']} weak order",
            f"{study['asymptotic_fit']['slope']:.4f}",
        )

    # EXP-04 -- the regime census that locates the bias floor.
    regimes: dict[str, int] = {}
    for cell in records["EXP-04"]["results"]["cells"]:
        regimes[cell["regime"]] = regimes.get(cell["regime"], 0) + 1
    add("EXP-04", "total cells", f"{sum(regimes.values())} cells")
    add("EXP-04", "bias-dominated cells", f"{regimes['bias_dominated']} are bias-dominated")
    add("EXP-04", "sampling-dominated cells", f"{regimes['sampling_dominated']} sampling-dominated")
    add("EXP-04", "mixed cells", f"{regimes['mixed']} mixed")

    # EXP-06 -- the temporal arms, including the inflated full-range fit.
    for arm in records["EXP-06"]["results"]["time_sweep"]:
        if arm["arm"] == "implicit":
            add("EXP-06", "implicit temporal order", f"{arm['fit_full_range']['order']:.4f}")
        if arm["arm"] == "crank_nicolson":
            add("EXP-06", "plain CN full-range order", f"{arm['fit_full_range']['order']:.4f}")
            add("EXP-06", "plain CN asymptotic order", f"{arm['fit_asymptotic_window']['order']:.4f}")

    # EXP-09 -- the characteristic function's own invariants.
    cf = records["EXP-09"]["results"]["characteristic_function_properties"]
    add("EXP-09", "martingale identity residual", f"{cf['max_martingale_identity_relative_error']:.1e}".replace("e-16", "e−16"))
    add("EXP-09", "regimes swept", f"{cf['regimes']} parameter")

    # EXP-12 -- the dispersion table, parameter by parameter. The *ranges* are asserted
    # rather than only the relative-dispersion percentages: a bare "95%" is satisfied by
    # the phrase "nominal 95% interval" elsewhere in the same document, so it would keep
    # passing after the number it describes had moved. The min-to-max span is specific to
    # this table and cannot be satisfied by accident.
    dispersion = records["EXP-12"]["results"]["parameter_dispersion"]
    spans = {
        "correlation": "{:.2f} to {:.2f}",
        "initial_variance": "{:.3f} to {:.3f}",
        "vol_of_variance": "{:.2f} to {:.2f}",
        "long_run_variance": "{:.3f} to {:.3f}",
        "mean_reversion": "{:.2f} to {:.2f}",
    }
    for name, form in spans.items():
        stats = dispersion[name]
        span = form.format(stats["min"], stats["max"]).replace("-", "−")
        add("EXP-12", f"{name} span across scenarios", span)
    for name in ("correlation", "initial_variance", "vol_of_variance"):
        stats = dispersion[name]
        add(
            "EXP-12",
            f"{name} relative dispersion",
            f"{stats['stddev'] / abs(stats['mean']) * 100:.0f}%",
        )
    surface = records["EXP-12"]["results"]
    add("EXP-12", "included quotes", f"{surface['included_quotes']} quotes")

    # EXP-13 -- agreement measured in combined standard errors.
    european = records["EXP-13"]["results"]["black_scholes_european"]
    worst = max(
        abs(m["sigmas"])
        for cell in european
        for m in cell["methods"]
        if not m.get("is_reference") and "sigmas" in m
    )
    add("EXP-13", "worst European disagreement", f"{worst:.2f}")

    return out


def claims(records: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Every checked claim, as (experiment, description, exact string required)."""
    out: list[tuple[str, str, str]] = []

    def add(exp: str, what: str, text: str) -> None:
        out.append((exp, what, text))

    # EXP-01 -- sampling convergence.
    scenarios = records["EXP-01"]["results"]["scenarios"]
    covering = sum(
        1
        for s in scenarios
        if s["study"]["slope_versus_paths_ci_lower"]
        <= -0.5
        <= s["study"]["slope_versus_paths_ci_upper"]
    )
    add("EXP-01", "scenario count whose interval covers -0.5", f"{WORDS[covering]} scenarios")

    # EXP-02 -- strong convergence, both fits published.
    studies = records["EXP-02"]["results"]["studies"]
    first = sorted({s["variation"] for s in studies})[0]
    for study in (s for s in studies if s["variation"] == first):
        add(
            "EXP-02",
            f"{study['scheme']} full-range slope",
            f"{study['full_fit']['slope']:.4f}",
        )
        add(
            "EXP-02",
            f"{study['scheme']} asymptotic slope",
            f"{study['asymptotic_fit']['slope']:.4f}",
        )

    # EXP-06 -- PDE orders and the stability margin.
    pde = records["EXP-06"]["results"]
    add("EXP-06", "Crank-Nicolson spatial order", f"{pde['space_sweep'][0]['fit']['order']:.4f}")
    stability = pde["explicit_stability"]
    add(
        "EXP-06",
        "highest stable Courant ratio",
        f"{stability['highest_ratio_observed_stable']:g}",
    )
    add(
        "EXP-06",
        "lowest unstable Courant ratio",
        f"{stability['lowest_ratio_observed_unstable']:g}",
    )
    for arm in pde["time_sweep"]:
        if arm["arm"] == "crank_nicolson_rannacher2":
            add("EXP-06", "Rannacher temporal order", f"{arm['fit_full_range']['order']:.4f}")

    # EXP-07 -- monitoring bias, the refused fit, and the correction's breakdown.
    barrier = records["EXP-07"]["results"]
    discrete = [a for a in barrier["arms"] if a["convention"] == "discrete"]
    daily = [
        lvl["relative_bias"]
        for a in discrete
        if a["barrier"] == 105.0
        for lvl in a["levels"]
        if lvl["monitoring_dates"] == 250
    ]
    add("EXP-07", "B=105 daily relative bias", f"{daily[0] * 100:.0f}%")
    worst = max(lvl["relative_bias"] for a in discrete for lvl in a["levels"])
    add("EXP-07", "worst coarse relative bias", f"{worst * 100:.0f}%")
    for geometry, tally in barrier["continuity_correction_by_geometry"].items():
        add(
            "EXP-07",
            f"correction disagreements, {geometry}",
            f"{tally['cells_disagreeing_with_correction']} of {tally['resolved_cells']}",
        )
    orders = [a["order_fit"]["order"] for a in barrier["pde_convergence"]]
    on_two = sum(1 for o in orders if abs(o - 2.0) <= 0.01)
    add(
        "EXP-07",
        "PDE cells landing on order 2.000",
        f"{WORDS[on_two]} of {WORDS[len(orders)]}",
    )

    # EXP-08 -- the finite-difference variance correction.
    greeks = records["EXP-08"]["results"]
    scaling = greeks["finite_difference_variance_scaling"]
    for greek in ("delta", "gamma"):
        exps = [c["dispersion_vs_bump_exponent"] for c in scaling if c["greek"] == greek]
        add("EXP-08", f"{greek} dispersion exponent median", f"{statistics.median(exps):.2f}")
    cells = greeks["cells"]
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
    ratios = sorted(
        likelihood[k]["across_seed_standard_error"] / pathwise[k]["across_seed_standard_error"]
        for k in pathwise
        if pathwise[k]["across_seed_standard_error"] > 0
    )
    add("EXP-08", "median pathwise/LR standard-error ratio", f"{statistics.median(ratios):.1f}")

    # EXP-10 -- one regime resolves an order, the other does not.
    fits = {f["regime"]: f for f in records["EXP-10"]["results"]["bias_decay_order_fits"]}
    violated = fits["feller_violated"]
    add("EXP-10", "Feller-violating decay order", f"{violated['bias_decay_order']:.2f}")

    # EXP-11 -- blind recovery on a synthetic surface.
    recovery = records["EXP-11"]["results"]
    add("EXP-11", "converged starts", f"{recovery['converged_count']} of {recovery['started_count']}")

    # EXP-12 -- a good fit that does not identify the parameters.
    stability12 = records["EXP-12"]["results"]
    dispersion = stability12["parameter_dispersion"]
    for name in ("mean_reversion", "long_run_variance"):
        stats = dispersion[name]
        relative = stats["stddev"] / abs(stats["mean"])
        add("EXP-12", f"{name} relative dispersion", f"{relative * 100:.0f}%")
    rmses = [s["implied_vol_rmse"] for s in stability12["scenarios"]]
    add("EXP-12", "best implied-vol RMSE", f"{min(rmses):.4f}")
    add("EXP-12", "worst implied-vol RMSE", f"{max(rmses):.4f}")
    add("EXP-12", "scenario count", f"{WORDS[len(stability12['scenarios'])]} scenarios")

    # EXP-14 -- the resolved under-coverage.
    under = [c for c in records["EXP-14"]["results"]["cells"] if c["under_covers"]]
    for cell in under:
        add("EXP-14", "observed coverage", f"{cell['observed_coverage']:.2%}")
        add("EXP-14", "deviation in sigmas", f"{cell['deviation_sigmas']:.1f}")
        add("EXP-14", "payoff skewness", f"{cell['payoff_skewness']:.1f}")

    # EXP-15 -- the edge-case envelope.
    summary = records["EXP-15"]["results"]["summary"]
    add("EXP-15", "edge cases passed", f"{summary['passed']} of {summary['total_cases']}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=pathlib.Path, default=pathlib.Path("results/README.md"))
    parser.add_argument("--paper", type=pathlib.Path, default=pathlib.Path("docs/TECHNICAL-PAPER.md"))
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    args = parser.parse_args()

    records = {}
    for path in sorted(args.results.glob("EXP-*.json")):
        records[path.stem] = json.loads(path.read_text())

    shared = claims(records)
    extra = paper_claims(records)

    # (document, claims it must carry). The reconciliation carries the headlines; the
    # paper carries those plus its own finer detail.
    targets: list[tuple[pathlib.Path, list[tuple[str, str, str]]]] = [
        (args.readme, shared),
        (args.paper, shared + extra),
    ]

    total = 0
    missing_total = 0
    for document, required in targets:
        if not document.exists():
            print(f"FAIL: {document} does not exist")
            return 1
        prose = document.read_text()
        missing = [(e, w, t) for e, w, t in required if t not in prose]
        for experiment, what, text in missing:
            print(
                f"FAIL: {document}: {experiment} {what}: the records imply '{text}', "
                "which the prose does not say"
            )
        print(f"{document}: {len(required) - len(missing)} of {len(required)} claims re-derived")
        total += len(required)
        missing_total += len(missing)

    print(f"\n{total - missing_total} of {total} claims re-derived from the committed records")
    if missing_total:
        print("The prose disagrees with the artifacts it describes.")
        return 1
    print("Every checked claim is supported by a committed record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

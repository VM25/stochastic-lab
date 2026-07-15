#!/usr/bin/env python3
"""Generate high-precision Black-Scholes reference values for C++ validation.

This is external analysis, not part of the engine (ADR-002). Python computes
reference numbers that the C++ test suite consumes as a fixed, versioned fixture;
it never prices anything at runtime and the engine does not depend on it.

Why this exists
---------------
The C++ Greeks are hand-derived closed forms. Two checks are needed, and they
answer different questions:

  * Richardson-extrapolated finite differences of the C++ price confirm that the
    Greek formulas are the derivatives of the price formula. That is a real
    check, but it differentiates the same implementation, so it cannot detect an
    error shared by both -- a wrong price formula would produce a matching wrong
    delta and the test would pass.

  * These references are computed at 50 decimal digits with an independent
    implementation (mpmath), and each Greek is obtained by differentiating the
    *price* numerically at that precision rather than by evaluating a
    closed-form Greek. So they depend on neither the C++ price code nor the C++
    Greek algebra. Truncation is controlled by the working precision, leaving
    values accurate far beyond double.

Together with the published Hull and Haug values (a human-checked external
source) and the optional QuantLib comparison (an independent production
implementation), no single oracle is load-bearing.

Conventions match docs/MATHEMATICAL-SPEC.MD:
  d1 = [log(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
  d2 = d1 - sigma sqrt(T)
  C  = S e^{-qT} N(d1) - K e^{-rT} N(d2)
  P  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Greeks are per unit of the underlying variable (not per volatility point or per
basis point), and theta is per year of calendar time, so theta = -dV/dT.

Usage:
    python3 python/generate_bs_references.py > data/references/black_scholes.json
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

try:
    from mpmath import mp, mpf, sqrt, log, exp, erfc
except ImportError:
    sys.exit("mpmath is required: pip install mpmath")

# 50 digits leaves ~34 to spare beyond double's 17, so the reference error is
# far below anything the C++ comparison can resolve.
mp.dps = 50

# Differentiation step for mp.diff. At 50 digits a step of 1e-15 leaves roughly
# 20 digits of headroom against rounding while making the O(h^2) truncation term
# negligible -- neither error source is near double precision.
DIFF_STEP = mpf("1e-15")


def norm_cdf(x):
    """Standard normal CDF via erfc, mirroring the C++ convention."""
    return erfc(-x / sqrt(2)) / 2


def bs_price(option_type, spot, strike, rate, dividend_yield, volatility, maturity):
    """Black-Scholes-Merton price, including the degenerate limits."""
    total_vol = volatility * sqrt(maturity)
    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    if total_vol == 0:
        if option_type == "call":
            return max(discounted_spot - discounted_strike, mpf(0))
        return max(discounted_strike - discounted_spot, mpf(0))

    d1 = (log(spot / strike) + (rate - dividend_yield + volatility**2 / 2) * maturity) / total_vol
    d2 = d1 - total_vol

    if option_type == "call":
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1)


def greeks(option_type, spot, strike, rate, dividend_yield, volatility, maturity):
    """Greeks as high-precision derivatives of the price.

    Deliberately differentiates the price rather than evaluating closed forms:
    the point is to be independent of the very algebra the C++ engine is being
    checked against.
    """

    def by_spot(s):
        return bs_price(option_type, s, strike, rate, dividend_yield, volatility, maturity)

    def by_volatility(v):
        return bs_price(option_type, spot, strike, rate, dividend_yield, v, maturity)

    def by_maturity(t):
        return bs_price(option_type, spot, strike, rate, dividend_yield, volatility, t)

    def by_rate(r):
        return bs_price(option_type, spot, strike, r, dividend_yield, volatility, maturity)

    return {
        "delta": mp.diff(by_spot, spot, 1, h=DIFF_STEP),
        "gamma": mp.diff(by_spot, spot, 2, h=DIFF_STEP),
        "vega": mp.diff(by_volatility, volatility, 1, h=DIFF_STEP),
        # theta is dV/dt in calendar time; maturity runs against the clock.
        "theta": -mp.diff(by_maturity, maturity, 1, h=DIFF_STEP),
        "rho": mp.diff(by_rate, rate, 1, h=DIFF_STEP),
    }


# Scenarios mirror tests/engines/black_scholes_greeks_test.cpp. Coverage follows
# VALIDATION-PLAN section 3: moneyness, maturity, volatility, dividend yield, and
# a negative rate.
SCENARIOS = [
    ("at_the_money", 100.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    ("in_the_money", 130.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    ("out_of_the_money", 70.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    ("with_dividend", 100.0, 100.0, 0.05, 0.03, 0.20, 1.00),
    ("low_volatility", 100.0, 100.0, 0.05, 0.00, 0.05, 1.00),
    ("high_volatility", 100.0, 100.0, 0.05, 0.00, 0.60, 1.00),
    ("short_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 0.08),
    ("long_dated", 100.0, 100.0, 0.05, 0.00, 0.20, 10.0),
    ("negative_rate", 100.0, 100.0, -0.01, 0.00, 0.20, 1.00),
    ("deep_out_of_the_money", 100.0, 200.0, 0.00, 0.00, 0.20, 1.00),
    ("deep_in_the_money", 200.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    ("high_dividend", 100.0, 100.0, 0.02, 0.08, 0.25, 2.00),
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    # 25 significant digits: comfortably more than double can hold, so the C++
    # side reads an exactly-rounded value, while the file stays readable.
    def fmt(x):
        return mp.nstr(x, 25, strip_zeros=False)

    cases = []
    for name, spot, strike, rate, q, vol, maturity in SCENARIOS:
        for option_type in ("call", "put"):
            args = (option_type, mpf(spot), mpf(strike), mpf(rate), mpf(q), mpf(vol), mpf(maturity))
            g = greeks(*args)
            cases.append(
                {
                    "scenario": name,
                    "option_type": option_type,
                    "market": {"spot": spot, "rate": rate, "dividend_yield": q},
                    "instrument": {"strike": strike, "maturity": maturity},
                    "model": {"volatility": vol},
                    "price": fmt(bs_price(*args)),
                    "greeks": {k: fmt(v) for k, v in g.items()},
                }
            )

    document = {
        "schema_version": 1,
        "description": (
            "High-precision Black-Scholes reference prices and Greeks for validating the "
            "DiffusionWorks analytic engine."
        ),
        "provenance": {
            "generator": "python/generate_bs_references.py",
            "method": (
                "mpmath at 50 decimal digits. Greeks are numerical derivatives of the price "
                "at that precision, not evaluations of closed-form Greek formulas, so they are "
                "independent of the C++ engine's Greek algebra."
            ),
            "mpmath_working_digits": mp.dps,
            "differentiation_step": str(DIFF_STEP),
            "reported_significant_digits": 25,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
        },
        "conventions": {
            "d1": "[log(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))",
            "d2": "d1 - sigma sqrt(T)",
            "delta": "dV/dS, per unit of spot",
            "gamma": "d2V/dS2, per unit of spot squared",
            "vega": "dV/dsigma, per unit of volatility (not per volatility point)",
            "theta": "dV/dt, per year of calendar time (= -dV/dT)",
            "rho": "dV/dr, per unit of rate (not per basis point)",
        },
        "cases": cases,
    }

    json.dump(document, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

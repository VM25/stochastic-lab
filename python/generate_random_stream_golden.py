#!/usr/bin/env python3
"""Generate golden values pinning the random stream's reproducibility tiers.

External analysis, not part of the engine (ADR-002).

Why tiers
---------
"Reproducible" is not one property here, and claiming it uniformly would overstate
what is actually guaranteed. Three distinct levels exist, and this fixture pins
each with the strictness it deserves:

  tier 1  Philox counter output.
          Integer arithmetic only -- multiply, xor, add on uint64. Bit-exact on
          every conforming platform, compiler, and optimisation level. No
          floating point is involved at all.

  tier 2  uniform_from_bits.
          Bit-exact too, and provably so rather than by luck: the 52-bit shift is
          integer, converting a 52-bit integer to double is exact, adding 0.5 is
          exact in that binade (spacing 0.5), and scaling by 2^-52 is a power of
          two. Every step is exact, so no rounding mode or libm is involved.

  tier 3  inverse_norm_cdf, and everything downstream of it.
          Split further:
            - central branch (|p - 0.5| <= 0.425, about 85% of draws) evaluates a
              rational polynomial with only + - * /. Each is correctly rounded by
              IEEE-754, and with contraction disabled the evaluation order is
              fixed, so this is bit-exact across conforming platforms.
            - tail branches (about 15% of draws) call std::log, whose accuracy
              IEEE-754 does not specify. glibc, macOS libm and musl may each
              return a different last bit. These draws are reproducible bit-for-bit
              on a fixed platform and toolchain, and equal to within about an ulp
              across them.

The same caveat applies to norm_cdf (std::erfc) and the Student-t functions
(std::lgamma, std::log, std::exp).

So the C++ tests demand bit equality for tiers 1 and 2 and a tolerance for tier
3. Demanding bit equality of tier 3 would produce a test that passes on the
machine it was written on and fails elsewhere for no defensible reason.

Normal references here are the *true* quantiles at 50 digits rather than AS 241's
output, so the fixture validates the algorithm rather than merely recording it.

Usage:
    python3 python/generate_random_stream_golden.py > data/references/random_stream_golden.json
"""

import json
import struct
import subprocess
import sys
from datetime import datetime, timezone

try:
    from mpmath import mp, mpf, erfc, sqrt, log, findroot
except ImportError:
    sys.exit("mpmath is required: pip install mpmath")

mp.dps = 50

M0 = 0xD2E7470EE14C6C93
M1 = 0xCA5A826395121157
W0 = 0x9E3779B97F4A7C15
W1 = 0xBB67AE8584CAA73B
MASK = (1 << 64) - 1
ROUNDS = 10

# Must match include/diffusionworks/random/random_stream.hpp.
PURPOSE_ASSET_SHOCK = 0
PURPOSE_VARIANCE_SHOCK = 1
PURPOSE_DIAGNOSTIC = 1000

WORDS_PER_BLOCK = 4


def mulhilo(a, b):
    product = a * b
    return (product >> 64) & MASK, product & MASK


def philox_round(ctr, key):
    hi0, lo0 = mulhilo(M0, ctr[0])
    hi1, lo1 = mulhilo(M1, ctr[2])
    return [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]


def philox4x64_10(ctr, key):
    ctr, key = list(ctr), list(key)
    for i in range(ROUNDS):
        if i > 0:
            key[0] = (key[0] + W0) & MASK
            key[1] = (key[1] + W1) & MASK
        ctr = philox_round(ctr, key)
    return ctr


def uniform_from_bits(bits):
    """Mirrors the C++ transform. Every step is exact, so this is bit-exact."""
    return (float(bits >> 12) + 0.5) * (2.0**-52)


def norm_cdf(x):
    return erfc(-x / sqrt(2)) / 2


def bisect_monotone(f, lo, hi):
    lo, hi = mpf(lo), mpf(hi)
    for _ in range(200):
        if (f(lo) <= 0) != (f(hi) <= 0):
            break
        lo *= 2
        hi *= 2
    else:
        raise RuntimeError("could not bracket root")
    return findroot(f, [lo, hi], solver="bisect", tol=mpf(10) ** (-(mp.dps - 10)))


def true_normal_quantile(u):
    """The exact quantile of the double u, at 50 digits.

    Solved in log space for the same reason as in
    generate_distribution_references.py: in the tail the CDF is vanishingly small
    and an absolute residual cannot meet any useful tolerance.
    """
    p = mpf(u)
    if p == mpf("0.5"):
        return mpf(0)
    if p < mpf("0.5"):
        target = log(p)
        scale = sqrt(-2 * target) if target < 0 else mpf(1)
        return bisect_monotone(lambda x: log(norm_cdf(x)) - target, -scale - 1, mpf("-1e-9"))
    target = log(1 - p)
    scale = sqrt(-2 * target) if target < 0 else mpf(1)
    return bisect_monotone(lambda x: log(norm_cdf(-x)) - target, mpf("1e-9"), scale + 1)


def draw(master_seed, purpose, path, position):
    block = position // WORDS_PER_BLOCK
    word = position % WORDS_PER_BLOCK
    output = philox4x64_10([block, path, 0, 0], [master_seed, purpose])
    return output[word]


def hex_double(x):
    """Exact hexadecimal rendering, so a bit-exact value survives the round trip.

    Decimal at 17 significant digits also round-trips, but hex makes the intent
    explicit: this value is compared bit-for-bit, not approximately.
    """
    return float(x).hex()


CASES = [
    ("asset_seed0_path0", 0, PURPOSE_ASSET_SHOCK, 0),
    ("asset_seed1_path0", 1, PURPOSE_ASSET_SHOCK, 0),
    ("asset_seed20260715_path0", 20260715, PURPOSE_ASSET_SHOCK, 0),
    ("asset_seed20260715_path1", 20260715, PURPOSE_ASSET_SHOCK, 1),
    ("asset_seed20260715_path1000000", 20260715, PURPOSE_ASSET_SHOCK, 1000000),
    ("variance_seed20260715_path0", 20260715, PURPOSE_VARIANCE_SHOCK, 0),
    ("diagnostic_seed42_path7", 42, PURPOSE_DIAGNOSTIC, 7),
    # Maximum representable coordinates, so the addressing space is pinned at its
    # documented edge rather than only in its comfortable middle.
    ("max_seed_max_path", MASK, PURPOSE_ASSET_SHOCK, MASK),
]

DRAWS_PER_CASE = 8


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    cases = []
    for name, seed, purpose, path in CASES:
        raw_words = []
        uniforms = []
        normals = []
        for position in range(DRAWS_PER_CASE):
            bits = draw(seed, purpose, path, position)
            u = uniform_from_bits(bits)
            raw_words.append(f"{bits:016x}")
            uniforms.append(hex_double(u))
            normals.append(mp.nstr(true_normal_quantile(u), 25, strip_zeros=False))

        cases.append(
            {
                "name": name,
                "master_seed": str(seed),
                "purpose": purpose,
                "path": str(path),
                "raw_words": raw_words,
                "uniforms_hex": uniforms,
                "normals": normals,
            }
        )

    document = {
        "schema_version": 1,
        "description": (
            "Golden values for the DiffusionWorks random stream, separated by reproducibility "
            "tier."
        ),
        "reproducibility_tiers": {
            "raw_words": {
                "guarantee": "bit_exact_everywhere",
                "reason": (
                    "Philox uses integer arithmetic only (multiply, xor, add on uint64). No "
                    "floating point is involved, so no platform, compiler, or optimisation "
                    "level can change it."
                ),
                "test_requirement": "exact equality",
            },
            "uniforms_hex": {
                "guarantee": "bit_exact_everywhere",
                "reason": (
                    "Every step of the transform is exact: the shift is integer, converting a "
                    "52-bit integer to double is exact, adding 0.5 is exact in that binade "
                    "(spacing 0.5), and scaling by 2^-52 is a power of two. Nothing rounds."
                ),
                "test_requirement": "exact equality; stored as hex to make that explicit",
            },
            "normals": {
                "guarantee": "tolerance_only_across_platforms",
                "reason": (
                    "inverse_norm_cdf evaluates a rational polynomial for |p - 0.5| <= 0.425 "
                    "(about 85% of draws), which uses only + - * / and is bit-exact on any "
                    "IEEE-754 platform once contraction is disabled. The remaining ~15% take "
                    "tail branches that call std::log, whose accuracy IEEE-754 does not "
                    "specify: glibc, macOS libm and musl may each return a different last bit. "
                    "Those draws are bit-reproducible on a fixed platform and toolchain, and "
                    "agree to about an ulp across them."
                ),
                "test_requirement": "relative tolerance ~1e-14",
                "note": (
                    "These are the true quantiles at 50 digits, not AS 241's output, so the "
                    "fixture validates the algorithm rather than recording it."
                ),
            },
        },
        "coordinates": {
            "description": (
                "A draw is f(master_seed, purpose, path, position). The map to Philox is "
                "key = (master_seed, purpose), counter = (position / 4, path, 0, 0), and the "
                "draw selects word (position mod 4) of the 256-bit output."
            ),
            "purpose_identifiers": {
                "asset_shock": PURPOSE_ASSET_SHOCK,
                "variance_shock": PURPOSE_VARIANCE_SHOCK,
                "diagnostic": PURPOSE_DIAGNOSTIC,
            },
            "purpose_identifiers_are_frozen": (
                "These values are part of the reproducibility contract. Changing one changes "
                "every stored result that used it."
            ),
            "max_master_seed": str(MASK),
            "max_purpose": str(MASK),
            "max_path": str(MASK),
            "max_position": str(MASK),
            "words_per_block": WORDS_PER_BLOCK,
        },
        "provenance": {
            "generator": "python/generate_random_stream_golden.py",
            "mpmath_working_digits": mp.dps,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
        },
        "cases": cases,
    }

    json.dump(document, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

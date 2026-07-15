#!/usr/bin/env python3
"""Generate high-precision reference values for the distribution functions.

External analysis, not part of the engine (ADR-002).

Covers the inverse normal CDF (Wichura AS 241) and the Student-t CDF and
quantile, which the C++ side implements from scratch. Two independent oracles are
emitted for each:

  * mpmath at 50 decimal digits, computing the quantities from their definitions
    (root-finding on the CDF for the inverse normal, the regularized incomplete
    beta for Student-t). Independent of any library's implementation.
  * scipy, an independent production implementation. Cross-checked against mpmath
    here so a disagreement surfaces when the fixture is built rather than being
    blamed on the C++ later.

The C++ tests compare against the mpmath column; the scipy agreement is recorded
as provenance.

Usage:
    python3 python/generate_distribution_references.py > data/references/distributions.json
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

try:
    from mpmath import mp, mpf, erfc, sqrt, log, findroot, betainc
except ImportError:
    sys.exit("mpmath is required: pip install mpmath")

mp.dps = 50


def norm_cdf(x):
    return erfc(-x / sqrt(2)) / 2


def bisect_monotone(f, lo, hi, increasing=True):
    """Bisect a monotone f to a sign change, expanding the bracket as needed.

    Used in preference to mpmath's secant default because these functions are
    solved out to p = 1e-300, where a derivative-based method has nothing to work
    with. Bisection only needs a sign, which stays meaningful everywhere.
    """
    lo, hi = mpf(lo), mpf(hi)

    for _ in range(200):
        if (f(lo) <= 0) != (f(hi) <= 0):
            break
        lo *= 2
        hi *= 2
    else:
        raise RuntimeError(f"could not bracket root in [{lo}, {hi}]")

    # Tolerance on the residual, not on x. It must stay reachable at the working
    # precision: asking for 1e-100 at 50 digits simply never converges. Ten
    # digits of headroom below mp.dps leaves ~40 correct digits, far more than
    # the 25 reported and than double can consume.
    return findroot(f, [lo, hi], solver="bisect", tol=mpf(10) ** (-(mp.dps - 10)))


def inverse_norm_cdf(p):
    """p-quantile of N(0,1), computed at 50 digits.

    Deliberately not a closed-form rational approximation: the C++ implements
    AS 241, and a reference sharing that approximation would validate the
    transcription while missing an error in the algorithm itself.

    Solved in log space. In the tail the CDF itself is ~1e-300, so an absolute
    residual can never meet a 1e-50 tolerance and a direct solve simply fails --
    but log(N(x)) is a smooth O(1)-scaled function there, and its root is the
    same. The problem is badly scaled, not badly posed.
    """
    # Round the input to double *before* solving. The C++ receives a double, so
    # the reference must be the exact quantile of that double, not of the decimal
    # literal it was written as.
    #
    # This is not pedantry. Near p = 1 the quantile is savagely sensitive:
    # dx/dp = 1/phi(x), which is about 4e9 at x = 6.36. A half-ulp difference in
    # how p is represented then moves the answer by ~1e-8 -- four orders of
    # magnitude above the tolerance the C++ is held to. Solving for the decimal
    # would make the fixture demand an answer to a different question.
    p = mpf(float(p))
    if p == mpf("0.5"):
        return mpf(0)

    if p < mpf("0.5"):
        target = log(p)
        # N(-x) ~ phi(x)/x, so -log(p) ~ x^2/2: a sound starting scale.
        scale = sqrt(-2 * target) if target < 0 else mpf(1)
        return bisect_monotone(lambda x: log(norm_cdf(x)) - target, -scale - 1, mpf("-1e-9"))

    target = log(1 - p)
    scale = sqrt(-2 * target) if target < 0 else mpf(1)
    return bisect_monotone(lambda x: log(norm_cdf(-x)) - target, mpf("1e-9"), scale + 1)


def student_t_cdf(t, nu):
    """CDF of Student's t via the regularized incomplete beta, from the definition."""
    t = mpf(t)
    nu = mpf(nu)
    x = nu / (nu + t * t)
    # betainc(a, b, 0, x, regularized=True) is I_x(a, b).
    tail = betainc(nu / 2, mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def student_t_quantile(p, nu):
    """p-quantile of Student's t at nu degrees of freedom.

    Bracketed rather than secant-solved: at nu = 1 the distribution is Cauchy and
    its 0.005 quantile is near -64, far from any naive starting guess, and the
    tails are heavy enough that a secant step can overshoot wildly.
    """
    # Rounded to double first, for the same reason as inverse_norm_cdf.
    p = mpf(float(p))
    if p == mpf("0.5"):
        return mpf(0)
    return bisect_monotone(lambda t: student_t_cdf(t, nu) - p, -2, 2)


def cross_check_with_scipy(normal_cases, t_cdf_cases, t_quantile_cases):
    """Sanity-check the fixture against an independent implementation.

    scipy is a cross-check, not the oracle, and the thresholds below reflect
    scipy's own accuracy rather than mpmath's. Measured on this fixture:

      * norm.ppf (Cephes ndtri) agrees to ~1e-15 relative -- essentially double
        precision, as expected.
      * t.ppf (Cephes stdtrit) agrees only to ~4e-11. The gap is not a
        representation artifact: at p = 0.975, nu = 100 the quantile's
        sensitivity to p is about 1/0.058, which cannot turn a half-ulp input
        difference into a 7e-11 output difference. It is scipy's inversion
        accuracy. mpmath computes the same quantity at 50 digits from the
        regularized incomplete beta, so where they differ, mpmath is right.

    The thresholds therefore differ per family. A single loose threshold would
    have hidden a real error in the normal cases behind scipy's t weakness.
    """
    try:
        import scipy
        from scipy.stats import norm, t as student_t
    except ImportError:
        return None

    def worst_relative(pairs):
        worst = mpf(0)
        for theirs, ours in pairs:
            if abs(ours) > mpf("1e-300"):
                worst = max(worst, abs((theirs - ours) / ours))
        return worst

    worst_normal = worst_relative(
        (mpf(float(norm.ppf(case["p"]))), mpf(case["expected"])) for case in normal_cases
    )
    worst_t_cdf = worst_relative(
        (mpf(float(student_t.cdf(case["t"], case["degrees_of_freedom"]))), mpf(case["expected"]))
        for case in t_cdf_cases
    )
    worst_t_quantile = worst_relative(
        (mpf(float(student_t.ppf(case["p"], case["degrees_of_freedom"]))), mpf(case["expected"]))
        for case in t_quantile_cases
    )

    for name, worst, threshold in [
        ("norm.ppf", worst_normal, mpf("1e-13")),
        ("t.cdf", worst_t_cdf, mpf("1e-13")),
        ("t.ppf", worst_t_quantile, mpf("1e-9")),
    ]:
        if worst > threshold:
            sys.exit(
                f"FATAL: scipy {name} disagrees with mpmath by {mp.nstr(worst, 5)} relative, "
                f"beyond the {mp.nstr(threshold, 2)} expected from its own accuracy. "
                "One of the two is wrong about the mathematics."
            )

    return (
        f"scipy {scipy.__version__}: norm.ppf to {mp.nstr(worst_normal, 3)}, "
        f"t.cdf to {mp.nstr(worst_t_cdf, 3)}, t.ppf to {mp.nstr(worst_t_quantile, 3)} relative "
        "(t.ppf is limited by scipy's own inversion accuracy, not by mpmath)"
    )


# Probabilities span the central region and both tails. The tails are where a
# rational approximation degrades, so a fixture of central values would pass
# while the far quantiles were wrong -- and the far quantiles are exactly what a
# deep out-of-the-money path needs.
NORMAL_PROBABILITIES = [
    "1e-300", "1e-100", "1e-20", "1e-10", "1e-6", "0.001", "0.01", "0.025",
    "0.05", "0.1", "0.25", "0.4", "0.5", "0.6", "0.75", "0.9", "0.95",
    "0.975", "0.99", "0.999", "0.999999", "0.9999999999",
]

# Degrees of freedom from the pathological (1: Cauchy, no mean) up to where the
# t is indistinguishable from the normal.
T_DEGREES_OF_FREEDOM = [1, 2, 3, 5, 10, 30, 100, 1000, 10000]

T_QUANTILE_PROBABILITIES = ["0.005", "0.025", "0.05", "0.5", "0.95", "0.975", "0.995"]

T_CDF_POINTS = ["-10.0", "-2.5", "-1.0", "0.0", "1.0", "2.5", "10.0"]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    def fmt(x):
        return mp.nstr(x, 25, strip_zeros=False)

    normal_cases = [
        {"p": float(p), "expected": fmt(inverse_norm_cdf(p))} for p in NORMAL_PROBABILITIES
    ]

    t_cdf_cases = [
        {
            "t": float(t),
            "degrees_of_freedom": nu,
            "expected": fmt(student_t_cdf(t, nu)),
        }
        for nu in T_DEGREES_OF_FREEDOM
        for t in T_CDF_POINTS
    ]

    t_quantile_cases = [
        {
            "p": float(p),
            "degrees_of_freedom": nu,
            "expected": fmt(student_t_quantile(p, nu)),
        }
        for nu in T_DEGREES_OF_FREEDOM
        for p in T_QUANTILE_PROBABILITIES
    ]

    scipy_note = cross_check_with_scipy(normal_cases, t_cdf_cases, t_quantile_cases)

    document = {
        "schema_version": 1,
        "description": (
            "High-precision reference values for the inverse normal CDF and the Student-t "
            "distribution, used to validate the DiffusionWorks implementations."
        ),
        "provenance": {
            "generator": "python/generate_distribution_references.py",
            "method": (
                "mpmath at 50 decimal digits. The inverse normal CDF is obtained by "
                "root-finding on the CDF rather than from a rational approximation, so it is "
                "independent of the C++ AS 241 implementation. Student-t is computed from the "
                "regularized incomplete beta."
            ),
            "mpmath_working_digits": mp.dps,
            "reported_significant_digits": 25,
            "cross_checked_against": scipy_note or "scipy unavailable; check skipped",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
        },
        "inverse_norm_cdf": normal_cases,
        "student_t_cdf": t_cdf_cases,
        "student_t_quantile": t_quantile_cases,
    }

    json.dump(document, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

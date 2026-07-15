#pragma once

#include <cmath>
#include <limits>

namespace diffusionworks {

/// 1/sqrt(2*pi), the standard normal density's normalising constant.
inline constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934;

/// 1/sqrt(2), used to map the normal CDF onto erfc.
inline constexpr double kInvSqrt2 = 0.707106781186547524400844362105;

/// Standard normal probability density.
///
/// \f[ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \f]
[[nodiscard]] inline double norm_pdf(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Standard normal cumulative distribution.
///
/// \f[ N(x) = \frac{1}{2}\operatorname{erfc}\!\left(-\frac{x}{\sqrt{2}}\right) \f]
///
/// Implemented through erfc rather than the algebraically equivalent
/// \f$\tfrac{1}{2}(1 + \operatorname{erf}(x/\sqrt{2}))\f$. The two agree in exact
/// arithmetic but not in floating point: for x well below zero, erf(x/sqrt(2))
/// approaches -1, and `1 + (-1 + eps)` cancels away the small quantity that *is*
/// the answer. erfc computes that tail directly and stays accurate in relative
/// terms until it underflows near x = -38.
///
/// This matters for deep out-of-the-money options, where N(d2) is exactly the
/// small tail probability being priced.
[[nodiscard]] inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

/// Inverse standard normal CDF: the p-quantile of N(0,1).
///
/// Wichura's algorithm AS 241 (PPND16), "Algorithm AS 241: The Percentage Points
/// of the Normal Distribution", Applied Statistics 37(3), 1988. Accurate to about
/// 1e-16 relative across the representable range of p.
///
/// Why inversion rather than Box-Muller or a Ziggurat
/// --------------------------------------------------
/// This function defines how a uniform becomes a normal, and that mapping is
/// load-bearing beyond mere sampling:
///
///   - It is monotone, so the antithetic partner of u is 1-u and maps to exactly
///     -z. With Box-Muller the pairing is not the negation, and with a rejection
///     method it is not even well defined.
///   - It consumes exactly one uniform per normal, so a draw's coordinates are
///     fixed. Box-Muller consumes two and Ziggurat a variable number, either of
///     which would desynchronise common random numbers between two runs that
///     differ only in a parameter.
///
/// Speed is the trade: a Ziggurat is faster per draw. Reproducibility and exact
/// variance-reduction identities are worth more here than sampling throughput,
/// and the cost is measured in Phase 13 rather than assumed.
///
/// Returns -infinity at p = 0 and +infinity at p = 1; those are the true
/// quantiles, and callers are expected not to supply them. The random stream
/// never does: its uniforms lie strictly inside (0,1). NaN propagates.
[[nodiscard]] inline double inverse_norm_cdf(double p) noexcept {
    // AS 241 splits the domain into a central rational approximation and two tail
    // approximations in sqrt(-log(r)), which is what keeps it accurate out to
    // p ~ 1e-300 rather than degrading like a single expansion would.
    if (!(p > 0.0)) {
        // Also catches NaN, which fails every comparison.
        return (p == 0.0) ? -std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::quiet_NaN();
    }
    if (p >= 1.0) {
        return (p == 1.0) ? std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::quiet_NaN();
    }

    const double q = p - 0.5;

    if (std::abs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q *
               (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
                     67265.770927008700853) *
                        r +
                    45921.953931549871457) *
                       r +
                   13731.693765509461125) *
                      r +
                  1971.5909503065514427) *
                     r +
                 133.14166789178437745) *
                    r +
                3.387132872796366608) /
               (((((((5226.495278852854561 * r + 28729.085735721942674) * r +
                     39307.89580009271061) *
                        r +
                    21213.794301586595867) *
                       r +
                   5394.1960214247511077) *
                      r +
                  687.1870074920579083) *
                     r +
                 42.313330701600911252) *
                    r +
                1.0);
    }

    double r = (q < 0.0) ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));

    double value = 0.0;
    if (r <= 5.0) {
        r -= 1.6;
        value = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
                      0.24178072517745061177) *
                         r +
                     1.27045825245236838258) *
                        r +
                    3.64784832476320460504) *
                       r +
                   5.7694972214606914055) *
                      r +
                  4.6303378461565452959) *
                     r +
                 1.42343711074968357734) /
                (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
                      0.0151986665636164571966) *
                         r +
                     0.14810397642748007459) *
                        r +
                    0.68976733498510000455) *
                       r +
                   1.6763848301838038494) *
                      r +
                  2.05319162663775882187) *
                     r +
                 1.0);
    } else {
        // Far tail. Splitting here rather than extending the previous branch is
        // what preserves accuracy at extreme p.
        r -= 5.0;
        value = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                      0.0012426609473880784386) *
                         r +
                     0.026532189526576123093) *
                        r +
                    0.29656057182850489123) *
                       r +
                   1.7848265399172913358) *
                      r +
                  5.4637849111641143699) *
                     r +
                 6.6579046435011037772) /
                (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
                      1.8463183175100546818e-5) *
                         r +
                     7.868691311456132591e-4) *
                        r +
                    0.0148753612908506148525) *
                       r +
                   0.13692988092273580531) *
                      r +
                  0.59983220655588793769) *
                     r +
                 1.0);
    }

    return (q < 0.0) ? -value : value;
}

}  // namespace diffusionworks

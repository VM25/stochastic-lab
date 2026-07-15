#pragma once

#include <cmath>

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

}  // namespace diffusionworks

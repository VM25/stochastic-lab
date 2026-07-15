#pragma once

#include <diffusionworks/core/result.hpp>

namespace diffusionworks {

/// Regularized incomplete beta function \f$ I_x(a,b) \f$.
///
/// Evaluated with the standard continued-fraction expansion under modified
/// Lentz iteration, applying the symmetry \f$ I_x(a,b) = 1 - I_{1-x}(b,a) \f$ so
/// that the expansion is only used where it converges quickly.
///
/// Present because the Student-t distribution is defined through it, and
/// MATHEMATICAL-SPEC section 6 requires t critical values rather than a normal
/// approximation at small sample sizes.
///
/// Requires a > 0, b > 0 and x in [0,1]; reports a violation rather than
/// returning a plausible number.
[[nodiscard]] Result<double> regularized_incomplete_beta(double a, double b, double x);

/// CDF of Student's t distribution with `degrees_of_freedom` degrees of freedom.
///
/// Requires degrees_of_freedom > 0. Non-integer degrees of freedom are admitted:
/// the definition does not require an integer, and Welch-style effective degrees
/// of freedom are fractional.
[[nodiscard]] Result<double> student_t_cdf(double t, double degrees_of_freedom);

/// The p-quantile of Student's t distribution.
///
/// Found by bracketed bisection on the CDF, which is monotone, so convergence is
/// unconditional. This is called once per confidence interval rather than inside
/// a path loop, so robustness is worth more than the iteration count.
///
/// Requires 0 < p < 1 and degrees_of_freedom > 0. The infinite quantiles at
/// p = 0 and p = 1 are reported as failures rather than returned: a confidence
/// interval of infinite width is not a useful answer, and silently returning one
/// would let a caller publish it.
[[nodiscard]] Result<double> student_t_quantile(double p, double degrees_of_freedom);

}  // namespace diffusionworks

#include <diffusionworks/numerics/normal.hpp>
#include <diffusionworks/statistics/distributions.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "distributions";

/// Continued-fraction expansion for the incomplete beta function, evaluated with
/// Lentz's modified algorithm.
///
/// Converges rapidly only for x < (a+1)/(a+b+2); the caller applies the
/// symmetry relation to stay on that side.
[[nodiscard]] Result<double> beta_continued_fraction(double a, double b, double x) {
    // Guards against division by zero in Lentz's recurrence, which is the
    // failure mode the algorithm is known for.
    constexpr double kTiny = 1e-300;

    // Convergence is declared when a step changes the result by less than this.
    //
    // Not the tightest value that ever works. At 3e-16 (~1.35 ULP) the iteration
    // is chasing the last representable digit, and whether it can land there
    // depends on the exact rounding of every step -- which the compiler may alter
    // by contracting a*b+c into an FMA. Under contraction the residual stalls
    // near 1e-14 and 3e-16 is never reached, so Student-t at large degrees of
    // freedom fails outright.
    //
    // The build disables contraction (see CMakeLists.txt), but a convergence
    // criterion that depends on a build flag is brittle: this code must remain
    // correct if it is ever compiled elsewhere. 1e-14 is comfortably reachable
    // either way and costs nothing real -- the result's accuracy is limited to
    // ~1e-11 by cancellation in the lgamma prefactor long before the iteration's
    // own residual matters.
    constexpr double kTolerance = 1e-14;

    // Empirically 35 iterations suffice even at a = 5e5 (nu = 1e6); the
    // expansion's convergence is nearly independent of a once the caller has
    // selected the favourable branch. 300 is generous headroom, and exceeding it
    // means something is wrong rather than merely slow.
    constexpr int kMaxIterations = 300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kTiny) {
        d = kTiny;
    }
    d = 1.0 / d;
    double result = d;

    for (int m = 1; m <= kMaxIterations; ++m) {
        const auto even_m = static_cast<double>(m);
        const double two_m = 2.0 * even_m;

        // Even step.
        double numerator = even_m * (b - even_m) * x / ((qam + two_m) * (a + two_m));
        d = 1.0 + numerator * d;
        if (std::abs(d) < kTiny) {
            d = kTiny;
        }
        c = 1.0 + numerator / c;
        if (std::abs(c) < kTiny) {
            c = kTiny;
        }
        d = 1.0 / d;
        result *= d * c;

        // Odd step.
        numerator = -(a + even_m) * (qab + even_m) * x / ((a + two_m) * (qap + two_m));
        d = 1.0 + numerator * d;
        if (std::abs(d) < kTiny) {
            d = kTiny;
        }
        c = 1.0 + numerator / c;
        if (std::abs(c) < kTiny) {
            c = kTiny;
        }
        d = 1.0 / d;
        const double delta = d * c;
        result *= delta;

        if (std::abs(delta - 1.0) < kTolerance) {
            return Result<double>::success(result);
        }
    }

    // Non-convergence is reported rather than returning the last iterate, which
    // would look like an answer.
    return Result<double>::failure(
        ErrorCode::ConvergenceFailure,
        fmt::format("incomplete beta continued fraction did not converge in {} iterations "
                    "(a={}, b={}, x={})",
                    kMaxIterations,
                    a,
                    b,
                    x),
        kContext);
}

}  // namespace

Result<double> regularized_incomplete_beta(double a, double b, double x) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        return Result<double>::failure(ErrorCode::InvalidArgument,
                                       fmt::format("a must be finite and positive but is {}", a),
                                       kContext);
    }
    if (!(b > 0.0) || !std::isfinite(b)) {
        return Result<double>::failure(ErrorCode::InvalidArgument,
                                       fmt::format("b must be finite and positive but is {}", b),
                                       kContext);
    }
    // Negated conjunction, not the DeMorgan form `x < 0.0 || x > 1.0`. The two
    // are equivalent in Boolean logic but not in IEEE arithmetic: every
    // comparison against NaN is false, so the negated form is true for NaN and
    // rejects it, while the DeMorgan form is false and would let NaN through to
    // the continued fraction. clang-tidy's suggestion here is unsound for
    // floating point.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(x >= 0.0 && x <= 1.0)) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument, fmt::format("x must lie in [0, 1] but is {}", x), kContext);
    }

    if (x == 0.0) {
        return Result<double>::success(0.0);
    }
    if (x == 1.0) {
        return Result<double>::success(1.0);
    }

    // The prefactor is formed in logs: the individual gamma functions overflow
    // for moderate a and b even where their ratio is perfectly representable.
    const double log_prefactor =
        std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x);
    const double prefactor = std::exp(log_prefactor);

    // Use the expansion on whichever side converges, then reflect if needed.
    if (x < (a + 1.0) / (a + b + 2.0)) {
        auto fraction = beta_continued_fraction(a, b, x);
        if (!fraction) {
            return fraction;
        }
        return Result<double>::success(prefactor * fraction.value() / a);
    }

    auto fraction = beta_continued_fraction(b, a, 1.0 - x);
    if (!fraction) {
        return fraction;
    }
    return Result<double>::success(1.0 - prefactor * fraction.value() / b);
}

Result<double> student_t_cdf(double t, double degrees_of_freedom) {
    if (!(degrees_of_freedom > 0.0) || !std::isfinite(degrees_of_freedom)) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("degrees_of_freedom must be finite and positive but is {}",
                        degrees_of_freedom),
            kContext);
    }
    if (std::isnan(t)) {
        return Result<double>::failure(ErrorCode::InvalidArgument, "t is NaN", kContext);
    }
    if (std::isinf(t)) {
        return Result<double>::success(t > 0.0 ? 1.0 : 0.0);
    }

    // F(t) = 1 - I_x(nu/2, 1/2)/2 for t > 0, with x = nu/(nu + t^2), and the
    // mirror image below zero. Computing the small side directly rather than as
    // 1 - (large side) keeps the tail accurate, the same reasoning as norm_cdf.
    const double x = degrees_of_freedom / (degrees_of_freedom + t * t);

    auto tail = regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x);
    if (!tail) {
        return tail;
    }

    const double half_tail = 0.5 * tail.value();
    return Result<double>::success(t > 0.0 ? 1.0 - half_tail : half_tail);
}

Result<double> student_t_quantile(double p, double degrees_of_freedom) {
    if (!(degrees_of_freedom > 0.0) || !std::isfinite(degrees_of_freedom)) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("degrees_of_freedom must be finite and positive but is {}",
                        degrees_of_freedom),
            kContext);
    }
    // Negated conjunction rather than the DeMorgan form, so that NaN is rejected:
    // see the note in regularized_incomplete_beta. The quantiles at 0 and 1 are
    // infinite; a confidence interval of infinite width is not an answer, so this
    // fails rather than returning one.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(p > 0.0 && p < 1.0)) {
        return Result<double>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("p must lie strictly inside (0, 1) but is {}", p),
            kContext);
    }

    if (p == 0.5) {
        return Result<double>::success(0.0);
    }

    // Bracket the root. The t distribution has heavier tails than the normal at
    // every finite degrees_of_freedom, so the normal quantile is always inside
    // the true one and makes a sound starting bound. It is then widened until it
    // straddles, rather than assumed adequate: at one degree of freedom the t
    // quantile exceeds the normal's by orders of magnitude.
    const double normal_quantile = inverse_norm_cdf(p);
    double lower = 0.0;
    double upper = 0.0;

    if (p > 0.5) {
        lower = 0.0;
        upper = std::max(1.0, normal_quantile);
        for (int expansion = 0; expansion < 200; ++expansion) {
            auto cdf = student_t_cdf(upper, degrees_of_freedom);
            if (!cdf) {
                return cdf;
            }
            if (cdf.value() >= p) {
                break;
            }
            upper *= 2.0;
            if (!std::isfinite(upper)) {
                return Result<double>::failure(
                    ErrorCode::ConvergenceFailure,
                    fmt::format(
                        "could not bracket the t quantile for p={}, nu={}", p, degrees_of_freedom),
                    kContext);
            }
        }
    } else {
        upper = 0.0;
        lower = std::min(-1.0, normal_quantile);
        for (int expansion = 0; expansion < 200; ++expansion) {
            auto cdf = student_t_cdf(lower, degrees_of_freedom);
            if (!cdf) {
                return cdf;
            }
            if (cdf.value() <= p) {
                break;
            }
            lower *= 2.0;
            if (!std::isfinite(lower)) {
                return Result<double>::failure(
                    ErrorCode::ConvergenceFailure,
                    fmt::format(
                        "could not bracket the t quantile for p={}, nu={}", p, degrees_of_freedom),
                    kContext);
            }
        }
    }

    // Bisection. Unconditionally convergent on a monotone function, and this is
    // called once per interval rather than per path, so the iteration count does
    // not matter.
    constexpr int kMaxIterations = 200;
    constexpr double kTolerance = 1e-13;

    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);

        // Converged when the bracket can no longer be split, measured relatively
        // so the test holds for quantiles of any magnitude.
        if (upper - lower <= kTolerance * std::max(1.0, std::abs(midpoint))) {
            return Result<double>::success(midpoint);
        }

        auto cdf = student_t_cdf(midpoint, degrees_of_freedom);
        if (!cdf) {
            return cdf;
        }

        if (cdf.value() < p) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }

    return Result<double>::failure(
        ErrorCode::ConvergenceFailure,
        fmt::format("t quantile bisection did not converge in {} iterations (p={}, nu={})",
                    kMaxIterations,
                    p,
                    degrees_of_freedom),
        kContext);
}

}  // namespace diffusionworks

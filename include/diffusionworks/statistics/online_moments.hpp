#pragma once

#include <diffusionworks/core/interval.hpp>
#include <diffusionworks/core/result.hpp>

#include <cstdint>

namespace diffusionworks {

/// Online mean and variance by Welford's method, with a merge operation.
///
/// Numerical motivation
/// --------------------
/// The textbook variance, \f$ \frac{1}{n-1}(\sum x^2 - n\bar{x}^2) \f$, subtracts
/// two large nearly-equal quantities and loses most of its significant digits
/// whenever the mean is large relative to the spread. That is the normal case
/// here: a discounted payoff of 10.45 with a standard deviation of 0.01 would see
/// the leading digits cancel, and the resulting variance -- and hence every
/// confidence interval built on it -- would be dominated by rounding. Welford's
/// recurrence accumulates deviations from the running mean instead and never
/// forms that difference.
///
/// The merge is not an optimisation
/// --------------------------------
/// ADR-011 requires thread-local accumulation followed by a reduction in a
/// defined order. merge() is that reduction. Because it is exact and
/// order-defined, combining per-thread accumulators in index order yields the
/// same answer at any thread count, up to the documented floating-point effects
/// of reassociation. Accumulating into one shared object under a lock would be
/// both slower and non-deterministic.
class OnlineMoments {
public:
    /// Incorporates one observation.
    void add(double x) noexcept {
        ++count_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        // Deliberately the deviation from the *updated* mean: pairing it with the
        // pre-update delta is what makes the recurrence exact.
        const double delta_after = x - mean_;
        sum_squared_deviations_ += delta * delta_after;
    }

    /// Absorbs another accumulator, as if its observations had been added here.
    ///
    /// Chan, Golub and LeVeque's parallel update. Exact for any split, so the
    /// result does not depend on how observations were partitioned.
    void merge(const OnlineMoments& other) noexcept {
        if (other.count_ == 0) {
            return;
        }
        if (count_ == 0) {
            *this = other;
            return;
        }

        const auto n_a = static_cast<double>(count_);
        const auto n_b = static_cast<double>(other.count_);
        const double total = n_a + n_b;
        const double delta = other.mean_ - mean_;

        mean_ += delta * (n_b / total);
        sum_squared_deviations_ +=
            other.sum_squared_deviations_ + delta * delta * (n_a * n_b / total);
        count_ += other.count_;
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }

    [[nodiscard]] double mean() const noexcept { return mean_; }

    /// Unbiased sample variance, \f$ s^2 = \frac{1}{n-1}\sum (x_i-\bar{x})^2 \f$.
    ///
    /// Fails below two observations, where it is undefined rather than zero.
    /// Returning 0.0 would claim a perfectly precise estimate from a single
    /// sample, which is exactly the kind of plausible-looking value this project
    /// refuses to emit.
    [[nodiscard]] Result<double> sample_variance() const;

    /// \f$ s/\sqrt{n} \f$, the standard error of the mean.
    [[nodiscard]] Result<double> standard_error() const;

    /// Two-sided confidence interval for the mean at `level` (e.g. 0.95).
    ///
    /// Uses Student-t critical values with n-1 degrees of freedom, per
    /// MATHEMATICAL-SPEC section 6. The t and normal quantiles agree to within
    /// 0.2% by a few hundred observations, so this costs nothing at production
    /// path counts while remaining correct at small ones -- where a normal
    /// approximation would quietly under-cover, which is precisely the defect
    /// EXP-15 exists to detect.
    [[nodiscard]] Result<ConfidenceInterval> confidence_interval(double level = 0.95) const;

private:
    std::uint64_t count_{0};
    double mean_{0.0};

    /// \f$ \sum (x_i - \bar{x})^2 \f$, accumulated without ever forming
    /// \f$ \sum x^2 \f$.
    double sum_squared_deviations_{0.0};
};

/// Online covariance and correlation of a paired sample, with a merge operation.
///
/// Needed for the control-variate coefficient \f$ \beta^* =
/// \operatorname{Cov}(X,Y)/\operatorname{Var}(Y) \f$ (MATHEMATICAL-SPEC section 11) and for
/// verifying that correlated normals actually carry the correlation they claim.
class OnlineCovariance {
public:
    void add(double x, double y) noexcept {
        ++count_;
        const double delta_x = x - mean_x_;
        mean_x_ += delta_x / static_cast<double>(count_);
        mean_y_ += (y - mean_y_) / static_cast<double>(count_);

        // Pairs the pre-update x deviation with the post-update y deviation;
        // using both pre-update or both post-update deviations is a common error
        // and biases the estimate.
        sum_co_deviations_ += delta_x * (y - mean_y_);

        moments_x_.add(x);
        moments_y_.add(y);
    }

    void merge(const OnlineCovariance& other) noexcept {
        if (other.count_ == 0) {
            return;
        }
        if (count_ == 0) {
            *this = other;
            return;
        }

        const auto n_a = static_cast<double>(count_);
        const auto n_b = static_cast<double>(other.count_);
        const double total = n_a + n_b;

        sum_co_deviations_ += other.sum_co_deviations_ + (mean_x_ - other.mean_x_) *
                                                             (mean_y_ - other.mean_y_) *
                                                             (n_a * n_b / total);

        mean_x_ += (other.mean_x_ - mean_x_) * (n_b / total);
        mean_y_ += (other.mean_y_ - mean_y_) * (n_b / total);
        count_ += other.count_;

        moments_x_.merge(other.moments_x_);
        moments_y_.merge(other.moments_y_);
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }

    [[nodiscard]] double mean_x() const noexcept { return mean_x_; }

    [[nodiscard]] double mean_y() const noexcept { return mean_y_; }

    [[nodiscard]] const OnlineMoments& moments_x() const noexcept { return moments_x_; }

    [[nodiscard]] const OnlineMoments& moments_y() const noexcept { return moments_y_; }

    /// Unbiased sample covariance. Fails below two observations.
    [[nodiscard]] Result<double> covariance() const;

    /// Pearson correlation. Fails below two observations, and where either
    /// marginal variance is zero -- there the correlation is genuinely undefined
    /// (a constant series has no direction to correlate with), and returning 0
    /// would assert independence that has not been established.
    [[nodiscard]] Result<double> correlation() const;

private:
    std::uint64_t count_{0};
    double mean_x_{0.0};
    double mean_y_{0.0};
    double sum_co_deviations_{0.0};
    OnlineMoments moments_x_;
    OnlineMoments moments_y_;
};

}  // namespace diffusionworks

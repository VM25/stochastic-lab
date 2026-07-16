#pragma once

#include <diffusionworks/core/result.hpp>

#include <cstdint>

namespace diffusionworks {

/// A uniform discretisation of [0, T].
///
/// Uniform by choice, not by omission. A non-uniform grid would complicate the
/// convergence experiments (EXP-02, EXP-03) without changing what they measure:
/// strong and weak order are properties of the scheme, and a uniform grid makes
/// the step size a single number to halve. Non-uniform grids belong with barrier
/// monitoring, where the dates are dictated by the contract rather than chosen.
///
/// Construction is validated (ADR-006): a grid in hand always has a positive
/// maturity and at least one step.
class TimeGrid {
public:
    /// Validates and constructs.
    ///
    /// Requires maturity > 0 and finite, and steps >= 1.
    ///
    /// Zero maturity is rejected here, unlike in EuropeanOption where it is the
    /// expiry limit: a grid over an instant has no steps to take, and simulating
    /// it is a caller error rather than a limiting case. The analytic engine
    /// handles T = 0 exactly, so nothing is lost.
    [[nodiscard]] static Result<TimeGrid> uniform(double maturity, std::int64_t steps);

    [[nodiscard]] double maturity() const noexcept { return maturity_; }

    [[nodiscard]] std::int64_t steps() const noexcept { return steps_; }

    /// Step size \f$ \Delta t = T/M \f$.
    [[nodiscard]] double step_size() const noexcept { return step_size_; }

    /// \f$ \sqrt{\Delta t} \f$, precomputed because every scheme needs it on
    /// every step and the square root is not free.
    [[nodiscard]] double sqrt_step_size() const noexcept { return sqrt_step_size_; }

    /// Time at grid index i, with index 0 the start and index `steps()` maturity.
    ///
    /// Computed as i*T/M rather than by accumulating dt, which would drift: after
    /// 10^6 steps the accumulated error is visible, and the final time would not
    /// be exactly T. The last index returns maturity exactly.
    [[nodiscard]] double time_at(std::int64_t index) const noexcept;

private:
    TimeGrid(double maturity, std::int64_t steps, double step_size, double sqrt_step_size) noexcept
        : maturity_(maturity), steps_(steps), step_size_(step_size),
          sqrt_step_size_(sqrt_step_size) {}

    double maturity_;
    std::int64_t steps_;
    double step_size_;
    double sqrt_step_size_;
};

}  // namespace diffusionworks

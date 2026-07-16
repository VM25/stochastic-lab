#pragma once

namespace diffusionworks {

/// A two-sided confidence interval for an estimate.
///
/// Lives in core rather than beside the pricing engines because an interval is a
/// statistical object, not a pricing one: a convergence slope, a variance ratio,
/// and an option price all need one, and statistics/ must not depend on engines/
/// to say so.
struct ConfidenceInterval {
    double lower{};
    double upper{};

    /// Nominal coverage, e.g. 0.95. Reported alongside the bounds because an
    /// interval without its level is not interpretable.
    double level{};

    [[nodiscard]] double width() const noexcept { return upper - lower; }

    [[nodiscard]] bool contains(double x) const noexcept { return x >= lower && x <= upper; }
};

}  // namespace diffusionworks

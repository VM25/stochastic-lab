#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/models/heston.hpp>

#include <array>

namespace diffusionworks {

/// A plain Heston parameter vector, unvalidated.
///
/// Distinct from HestonModel, which validates on construction and is immutable. The
/// optimizer needs a mutable five-vector it can transform and perturb; this is that
/// vector, and it becomes a HestonModel only when a price is actually needed.
struct HestonParameters {
    double initial_variance{};
    double mean_reversion{};
    double long_run_variance{};
    double vol_of_variance{};
    double correlation{};

    /// Builds a validated model, or fails exactly as HestonModel::create would.
    [[nodiscard]] Result<HestonModel> to_model() const;
};

/// Box bounds for the five parameters. The transform maps the open box strictly onto
/// all of R^5, so a bound is never touched exactly and the optimizer stays feasible.
struct HestonParameterBounds {
    HestonParameters lower;
    HestonParameters upper;

    /// Documented defaults, wide enough not to exclude a plausible calibration but
    /// finite so the transform is well defined. The correlation stays strictly inside
    /// (-1, 1): at exactly +/-1 the model degenerates and the transform diverges.
    [[nodiscard]] static HestonParameterBounds defaults();

    /// True when every lower bound is below its upper and the box is non-empty. A
    /// calibration cannot run against an inside-out box, so it is checked up front.
    [[nodiscard]] bool valid() const noexcept;

    /// True when the parameters lie strictly inside the box -- the precondition for
    /// mapping them to the unconstrained space.
    [[nodiscard]] bool contains(const HestonParameters& parameters) const noexcept;
};

/// Maps the constrained parameters to R^5 via a per-coordinate logit, so a
/// derivative-free optimizer can search an unconstrained space while every point it
/// proposes maps back to a feasible parameter set.
///
/// For a coordinate p in the open interval (lo, hi),
///
///   x = log((p - lo) / (hi - p)),        p = lo + (hi - lo) / (1 + e^{-x}).
///
/// The inverse is total: any real x maps to a p strictly inside (lo, hi), so the
/// optimizer cannot leave the box no matter how far it steps. The forward map requires
/// p strictly inside; a p on or past a bound has no finite image and is rejected.
[[nodiscard]] Result<std::array<double, 5>> to_unconstrained(const HestonParameters& parameters,
                                                             const HestonParameterBounds& bounds);

/// Maps an unconstrained point back to feasible parameters. Total: never fails.
[[nodiscard]] HestonParameters to_constrained(const std::array<double, 5>& point,
                                              const HestonParameterBounds& bounds);

}  // namespace diffusionworks

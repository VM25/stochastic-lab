#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace diffusionworks::test {

// Scale-aware cross-thread agreement.
//
// Changing the thread count changes an estimate only by the floating-point
// *reassociation* of an order-N reduction: combining T thread-local accumulators in a
// different grouping than a single sequential pass rounds differently, and that
// rounding scales with the magnitude of the quantity being reduced. A universal
// absolute constant is therefore the wrong shape for the tolerance -- it is too tight
// for a large price and meaningless for a small Greek. The bound is expressed relative
// to the reference magnitude instead.
//
// Two scales, because two quantities reassociate differently:
//   - A mean-like estimator (a price, a Greek) reassociates by a few parts in 1e-13 --
//     a handful of ulps across the T-1 merges of `OnlineMoments::merge`.
//   - A standard error, built from a variance that subtracts means, carries more
//     cancellation and reassociates more -- parts in 1e-7 at worst.
// Both are far below the Monte Carlo standard error itself: a genuine data race would
// move a figure by order its own value (relative 1e-2 or more), so neither tolerance
// can mask a real disagreement while both stay robust across engines whose prices,
// Greeks, and standard errors span several orders of magnitude.
//
// A small absolute floor keeps the bound sane for a near-zero reference (gamma, or a
// deeply knocked-out barrier price), where a purely relative tolerance would collapse
// to demanding exact equality.

/// Relative tolerance for a mean-like estimator (a price or a Greek).
inline constexpr double kMeanRelativeTolerance = 1e-9;

/// Relative tolerance for a standard error, whose variance accumulation carries more
/// cancellation and so reassociates more than the mean does.
inline constexpr double kErrorRelativeTolerance = 1e-5;

/// A tiny absolute floor, so a near-zero reference does not demand exact equality.
inline constexpr double kAbsoluteFloor = 1e-12;

/// Scale-aware tolerance for comparing a mean-like estimator across thread counts.
[[nodiscard]] inline double mean_tolerance(double reference) {
    return kMeanRelativeTolerance * std::abs(reference) + kAbsoluteFloor;
}

/// Scale-aware tolerance for comparing a standard error across thread counts.
[[nodiscard]] inline double error_tolerance(double reference) {
    return kErrorRelativeTolerance * std::abs(reference) + kAbsoluteFloor;
}

/// Asserts a mean-like estimator agrees with its single-thread reference to the
/// scale-aware mean tolerance.
inline void expect_mean_agrees(double actual, double reference, const std::string& what) {
    EXPECT_NEAR(actual, reference, mean_tolerance(reference))
        << what << ": " << actual << " vs single-thread " << reference
        << " exceeds the scale-aware tolerance " << mean_tolerance(reference);
}

/// Asserts a standard error agrees with its single-thread reference to the scale-aware
/// error tolerance.
inline void expect_error_agrees(double actual, double reference, const std::string& what) {
    EXPECT_NEAR(actual, reference, error_tolerance(reference))
        << what << ": " << actual << " vs single-thread " << reference
        << " exceeds the scale-aware tolerance " << error_tolerance(reference);
}

}  // namespace diffusionworks::test

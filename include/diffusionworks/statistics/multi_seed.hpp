#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace diffusionworks {

/// The outcome of one independent replication.
struct SeedResult {
    std::uint64_t seed{};
    double estimate{};
};

/// Aggregate behaviour of an estimator across independent seeds.
///
/// Why this type exists
/// --------------------
/// A single Monte Carlo run reports an estimate and a standard error, and the
/// standard error is itself an estimate. It can be wrong -- systematically so for
/// a skewed payoff -- and a run cannot detect that from the inside. Repeating the
/// whole run under independent seeds measures the estimator's actual dispersion
/// instead of its self-reported one.
///
/// MATHEMATICAL-SPEC section 6 requires published stochastic results to aggregate
/// multiple seeds, and FAILURE-MODES section 7 names publishing only favourable
/// seeds as a completion blocker. This type is what makes the multi-seed
/// statement the natural one to report.
struct MultiSeedSummary {
    /// Number of independent replications.
    std::uint64_t seed_count{};

    /// Mean estimate across seeds.
    double mean{};

    /// Standard deviation of the estimate across seeds. This is the estimator's
    /// realised dispersion: the quantity a single run's standard error is trying
    /// to predict.
    double standard_deviation{};

    /// Standard error of the across-seed mean.
    double standard_error{};

    /// Root mean squared error against a reference, when one is supplied.
    ///
    /// Absent when no reference exists, rather than zero: RMSE against nothing is
    /// not zero error, it is an unanswerable question.
    std::optional<double> rmse;

    /// Mean signed deviation from the reference, when one is supplied.
    ///
    /// Reported separately from RMSE because they answer different questions.
    /// RMSE mixes bias and variance; a method can have small RMSE and a
    /// stubbornly wrong centre, which is what discretization bias looks like
    /// (EXP-04).
    std::optional<double> bias;

    /// Smallest and largest estimate observed, so a reader can see the spread
    /// rather than only its summary.
    double minimum{};
    double maximum{};
};

/// Summarises independent replications, optionally against a reference value.
///
/// Requires at least two seeds: dispersion is undefined for one, and reporting
/// zero dispersion from a single run would assert a reliability that has not
/// been measured. That is the failure this type exists to prevent, so it is a
/// rejection rather than a special case.
[[nodiscard]] Result<MultiSeedSummary>
summarize_seeds(const std::vector<SeedResult>& results,
                std::optional<double> reference = std::nullopt);

}  // namespace diffusionworks

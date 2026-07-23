#pragma once

#include <diffusionworks/experiments/experiment.hpp>

namespace diffusionworks {

/// Parameters for EXP-15: numerical edge cases.
///
/// Does the system fail safely or remain correct under limiting and difficult
/// inputs? The edge cases themselves are specific by nature -- a maturity going to
/// zero, a barrier already breached, a correlation at the boundary -- so they are
/// built into the experiment; this config carries only the base market the safe
/// cases are measured against and the tolerances. Each case must do exactly one of
/// two things: produce the correct limiting value, or refuse with an explicit error.
/// A NaN, an infinity, a silently wrong number, or a silent fallback is a failure,
/// and any unresolved case blocks the record.
struct EdgeCaseExperimentConfig {
    double spot{100.0};
    double rate{0.05};
    double dividend_yield{0.0};
    double volatility{0.2};

    /// How close a limiting value must be to its theoretical target (intrinsic value
    /// or discounted forward) to count as correct.
    double limit_tolerance{1e-6};

    /// A price this small counts as zero (an already-knocked-out barrier, a deep
    /// out-of-the-money option in the limit).
    double zero_tolerance{1e-8};

    /// The small maturity and volatility used for the "approaching zero" cases.
    double tiny_maturity{1e-8};
    double tiny_volatility{1e-8};
};

/// EXP-15: under limiting and difficult inputs, does the system stay correct or fail
/// safely, and never let a non-finite or silently wrong value escape?
[[nodiscard]] Result<ExperimentRecord> run_edge_cases(const EdgeCaseExperimentConfig& config);

}  // namespace diffusionworks

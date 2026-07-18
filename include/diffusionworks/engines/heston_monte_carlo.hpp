#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/pricing_result.hpp>
#include <diffusionworks/instruments/european_option.hpp>
#include <diffusionworks/market/market_state.hpp>
#include <diffusionworks/models/heston.hpp>

#include <cstdint>
#include <optional>

namespace diffusionworks {

/// How the variance process is kept in bounds while it is discretised.
///
/// The two schemes exist to be compared: EXP-10 measures what a naive Euler does that
/// full truncation does not, so both are first-class rather than one being a private
/// baseline. The naive scheme is not offered as a way to price -- it is offered as the
/// thing whose failure justifies the extra care.
enum class HestonVarianceScheme : std::uint8_t {
    /// Full-truncation Euler (Lord, Koekkoek and van Dijk, 2010). The variance state
    /// is allowed to go negative; only its positive part enters the drift, the
    /// diffusion, and the spot step. No square root ever sees a negative argument, so
    /// the scheme cannot produce a non-finite path from the variance going below zero.
    FullTruncation,

    /// The naive discretisation a first implementation reaches for: the raw variance
    /// enters the square root directly, with no truncation. When a step drives the
    /// discretised variance below zero -- which the true CIR variance never does but
    /// this scheme routinely does once the Feller condition is stressed -- the square
    /// root of a negative number is not-a-number and the path is lost. It exists here
    /// only as EXP-10's diagnostic baseline; it is never a recommended way to price.
    NaiveEuler,
};

[[nodiscard]] const char* to_string(HestonVarianceScheme scheme) noexcept;

struct HestonMonteCarloConfig {
    std::int64_t paths{200000};

    /// Time steps per path. Unlike the exact log-normal GBM scheme, the Heston
    /// Euler discretisation carries a bias that shrinks with the step, so this is the
    /// knob EXP-10's convergence study turns.
    std::int64_t steps{100};

    std::uint64_t seed{20260717};
    double confidence_level{0.95};

    /// The default is full truncation. The naive scheme is selected only by the
    /// experiment that studies it, never as a way to actually price an option.
    HestonVarianceScheme scheme{HestonVarianceScheme::FullTruncation};

    /// Worker threads for the path loop. One is sequential and bit-identical to the
    /// single-threaded engine (ADR-011). More partition the paths deterministically
    /// across fixed workers with thread-local accumulators and diagnostics reduced in
    /// block order, so a fixed thread count is reproducible and different counts differ
    /// only by the floating-point reassociation of the payoff merge -- a documented
    /// effect, not a race. Each path draws from streams keyed by (seed, purpose, index),
    /// so there is no shared mutable RNG to contend on. The integer excursion counts and
    /// the minimum-variance depth reduce exactly (sum and min), so they are identical at
    /// any thread count and no worker's diagnostics can be lost.
    int threads{1};
};

/// What the Heston simulation observed about its own variance paths.
///
/// The variance process is where the difficulty lives: a full-truncation Euler step
/// can drive the discretised variance below zero even though the true CIR variance
/// never does, most often when the Feller condition fails. These diagnostics make
/// that visible rather than hiding it behind a smooth-looking price.
struct HestonSimulationDiagnostics {
    std::int64_t paths{};
    std::int64_t steps{};

    /// Variance-step events where the pre-truncation variance came out negative and
    /// was floored at zero. The count is over all path-steps, so its natural scale
    /// is `paths * steps`. Zero does not certify correctness, but a large fraction
    /// is a warning that the discretisation is straining -- and it rises sharply as
    /// the Feller condition is violated.
    std::int64_t negative_variance_events{};

    /// The most negative pre-truncation variance any path-step reached (or the
    /// starting variance if none went lower). This is the depth of the excursion the
    /// truncation had to absorb: a small negative number is a scheme brushing the
    /// boundary, a large one is a scheme being held together by the flooring. It
    /// quantifies the invalid-variance behaviour the exit gate says must be visible.
    double minimum_variance{};

    /// Paths that produced a non-finite spot or variance. A single one blocks the
    /// price: a diverged path is not a sample to be dropped, it is evidence the
    /// simulation failed for these parameters (FAILURE-MODES). Full truncation never
    /// produces one from a negative variance; the naive scheme does, which is the
    /// whole reason full truncation is the default.
    std::int64_t non_finite_paths{};
};

/// A completed simulation: what happened to the variance, and the price if one is
/// admissible.
///
/// The price is optional on purpose. A run in which some path went non-finite has no
/// admissible price -- averaging the survivors would report a clean number for a
/// simulation that blew up -- but it still produced diagnostics worth reading, and for
/// the naive scheme those diagnostics *are* the result. Separating the two lets the
/// experiment quantify a scheme's failures without the failure erasing the evidence.
struct HestonSimulationOutcome {
    HestonSimulationDiagnostics diagnostics;

    /// Present only when every path stayed finite. Absent is not an error here: it is
    /// the finding that these parameters and this scheme cannot be priced.
    std::optional<PricingResult> price;
};

/// Prices European options under Heston by simulating variance and spot paths with
/// the full-truncation Euler scheme.
///
/// Full-truncation Euler (Lord, Koekkoek and van Dijk, 2010)
/// ---------------------------------------------------------
/// The variance is stepped as
///
///   v_{n+1} = v_n + kappa (theta - v_n^+) dt + xi sqrt(v_n^+ dt) Z^v
///
/// where v_n^+ = max(v_n, 0). The *state* v_n is allowed to go negative; only its
/// truncation v_n^+ enters the drift, the diffusion, and the spot step. This is the
/// scheme with the smallest bias among the simple fixes for the CIR positivity
/// problem, and -- unlike reflection or absorption -- it does not distort the mean.
/// The spot is stepped in log form,
///
///   ln S_{n+1} = ln S_n + (r - q - v_n^+/2) dt + sqrt(v_n^+ dt) Z^S,
///
/// with Z^S and Z^v carrying the model's correlation rho, drawn from two independent
/// streams and combined by the Cholesky factor.
///
/// The bias, not hidden
/// --------------------
/// The scheme is biased: the discretised price differs from the true Heston price by
/// an amount that decays with the step (roughly first order). This engine does not
/// pretend otherwise. EXP-10 measures the bias against the semi-analytic reference
/// and its decay with the step, and the negative-variance diagnostic reports how hard
/// the truncation is working. A non-finite path is a failure, not a dropped sample.
class HestonMonteCarloEngine {
public:
    /// Prices the option, or fails. A non-finite path is a hard failure
    /// (NonFiniteValue): the exit gate requires unresolved non-finite paths to block
    /// completion rather than be averaged away.
    [[nodiscard]] static Result<PricingResult> price(const MarketState& market,
                                                     const EuropeanOption& option,
                                                     const HestonModel& model,
                                                     const HestonMonteCarloConfig& config);

    /// Runs the simulation and returns its diagnostics together with a price when one
    /// is admissible. Fails only on an invalid request (too few paths, non-positive
    /// maturity, a bad grid or correlation); a run that goes non-finite is reported
    /// through the outcome, not as an error, so a scheme's failure can be measured.
    [[nodiscard]] static Result<HestonSimulationOutcome>
    simulate(const MarketState& market,
             const EuropeanOption& option,
             const HestonModel& model,
             const HestonMonteCarloConfig& config);
};

}  // namespace diffusionworks

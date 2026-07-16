#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/random/random_stream.hpp>
#include <diffusionworks/simulation/gbm_stepper.hpp>
#include <diffusionworks/simulation/time_grid.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace diffusionworks {

/// What happened while generating a path, beyond the numbers themselves.
struct PathDiagnostics {
    /// States that reached zero or went negative.
    ///
    /// Not a defect and not an error: Euler-Maruyama and Milstein step the price
    /// itself, so a sufficiently negative shock drives
    /// \f$ 1 + (r-q)\Delta t + \sigma\sqrt{\Delta t}\,Z \f$ below zero, and the
    /// price follows. The exact scheme steps log-price and cannot.
    ///
    /// The consequence is real and must not be hidden. A call payoff clamps a
    /// negative terminal price to zero and returns a perfectly ordinary number,
    /// so the resulting bias is invisible in the price alone. Counting the
    /// excursions makes it measurable, which is what EXP-04 needs, and warning on
    /// them keeps the reader from mistaking a discretisation artifact for a
    /// price.
    std::int64_t non_positive_states{0};
};

/// Which member of an antithetic pair to generate.
enum class PathVariate : std::uint8_t {
    /// The path driven by Z.
    Primary,

    /// The path driven by -Z, sharing the primary's coordinates exactly.
    ///
    /// A reflection of the same draw, not another sample. Because the uniform is
    /// mapped through a monotone inverse CDF that is odd about u = 0.5, the
    /// antithetic shock is the exact negation of the primary's -- so the pair is
    /// genuinely antithetic rather than approximately so.
    Antithetic,
};

/// Generates GBM paths on a fixed grid.
///
/// A path is a pure function of its index. Path 500 is the same sequence whether
/// it is generated first or last, on one thread or sixty-four, because the
/// underlying stream is addressed by coordinates rather than advanced by use
/// (ADR-010).
///
/// The same path index under two different schemes consumes the *same* shocks.
/// That is what makes strong convergence measurable at all: the error of a
/// discretised path is only meaningful against the true path driven by the same
/// Brownian motion, and comparing two paths built from different draws would
/// measure sampling noise instead (FAILURE-MODES section 6).
///
/// Paths are written into a caller-supplied buffer so that the buffer is
/// allocated once per simulation rather than once per path (TECHNICAL-DESIGN
/// section 11).
class GbmPathGenerator {
public:
    GbmPathGenerator(const MarketState& market,
                     const BlackScholesModel& model,
                     const TimeGrid& grid,
                     DiscretizationScheme scheme) noexcept;

    /// Number of states a path occupies: the initial spot plus one per step.
    [[nodiscard]] std::size_t path_size() const noexcept {
        return static_cast<std::size_t>(grid_.steps()) + 1;
    }

    /// Generates one path into `out`, which must have exactly path_size() entries.
    ///
    /// out[0] is the initial spot; out[i] is the state at grid time i.
    ///
    /// Fails with PathFailure if any state is non-finite. That is not a
    /// diagnostic to be counted: a NaN state makes every downstream number
    /// meaningless, and a payoff evaluated on it would silently contribute
    /// garbage -- or worse, a plausible zero -- to the estimator.
    ///
    /// A non-positive state is reported in the diagnostics rather than failing,
    /// because it is a genuine and expected property of the explicit schemes
    /// whose bias the experiments exist to measure.
    [[nodiscard]] Result<PathDiagnostics>
    generate(std::uint64_t master_seed,
             std::uint64_t path_index,
             std::span<double> out,
             PathVariate variate = PathVariate::Primary) const;

    [[nodiscard]] const TimeGrid& grid() const noexcept { return grid_; }

    [[nodiscard]] const GbmStepper& stepper() const noexcept { return stepper_; }

private:
    TimeGrid grid_;
    GbmStepper stepper_;
    double initial_spot_;
};

}  // namespace diffusionworks

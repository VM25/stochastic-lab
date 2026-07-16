#include <diffusionworks/simulation/gbm_path_generator.hpp>

#include <fmt/format.h>

#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "GbmPathGenerator";

}  // namespace

GbmPathGenerator::GbmPathGenerator(const MarketState& market,
                                   const BlackScholesModel& model,
                                   const TimeGrid& grid,
                                   DiscretizationScheme scheme) noexcept
    : grid_(grid), stepper_(GbmStepper::create(market, model, grid, scheme)),
      initial_spot_(market.spot()) {}

Result<PathDiagnostics> GbmPathGenerator::generate(std::uint64_t master_seed,
                                                   std::uint64_t path_index,
                                                   std::span<double> out) const {
    if (out.size() != path_size()) {
        return Result<PathDiagnostics>::failure(
            ErrorCode::InvalidArgument,
            fmt::format(
                "output buffer holds {} states but the grid needs {}", out.size(), path_size()),
            kContext);
    }

    // The stream is addressed by (seed, purpose, path); the step index is its
    // position. Nothing about scheduling, or about which scheme is running, can
    // change which shocks this path sees.
    RandomStream stream(master_seed, StreamPurpose::AssetShock, path_index);

    PathDiagnostics diagnostics;
    out[0] = initial_spot_;

    for (std::int64_t step = 0; step < grid_.steps(); ++step) {
        const double previous = out[static_cast<std::size_t>(step)];
        const double next = stepper_.advance(previous, stream.next_normal());
        out[static_cast<std::size_t>(step) + 1] = next;

        if (!std::isfinite(next)) {
            // Reported, never counted and continued. A NaN state makes every
            // number downstream of it meaningless, and a payoff evaluated on one
            // would contribute a plausible-looking zero to the estimator.
            return Result<PathDiagnostics>::failure(
                ErrorCode::PathFailure,
                fmt::format("path {} reached a non-finite state ({}) at step {} of {} under the "
                            "{} scheme; the previous state was {}",
                            path_index,
                            next,
                            step + 1,
                            grid_.steps(),
                            to_string(stepper_.scheme()),
                            previous),
                kContext);
        }

        if (next <= 0.0) {
            ++diagnostics.non_positive_states;
        }
    }

    return Result<PathDiagnostics>::success(diagnostics);
}

}  // namespace diffusionworks

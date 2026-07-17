#include <diffusionworks/pde/asset_grid.hpp>

#include <fmt/format.h>

#include <cmath>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "AssetGrid";

/// The fewest nodes a scheme can step on: two boundaries and one interior point.
constexpr std::int64_t kMinimumNodes = 3;

}  // namespace

AssetGrid::AssetGrid(double s_max, double spacing, std::vector<double> values) noexcept
    : s_max_(s_max), spacing_(spacing), values_(std::move(values)) {}

Result<AssetGrid> AssetGrid::uniform(double s_max, std::int64_t nodes) {
    if (!(s_max > 0.0) || !std::isfinite(s_max)) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("s_max must be positive and finite, got {}", s_max),
            kContext);
    }
    if (nodes < kMinimumNodes) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("a grid needs at least {} nodes -- two boundaries and one interior point "
                        "for the scheme to step -- but {} were requested",
                        kMinimumNodes,
                        nodes),
            kContext);
    }

    const auto count = static_cast<std::size_t>(nodes);
    const double spacing = s_max / static_cast<double>(nodes - 1);

    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        // i * dS rather than an accumulated sum. Accumulating drifts: after a few
        // thousand additions the last node is not exactly s_max, and the boundary
        // condition would then be applied at the wrong place.
        values[i] = static_cast<double>(i) * spacing;
    }
    // The endpoints are pinned exactly. The multiplication above is off by an ulp
    // at the top for most spacings, and the boundary condition is written at
    // S_max, not at almost-S_max.
    values.front() = 0.0;
    values.back() = s_max;

    return Result<AssetGrid>::success(AssetGrid(s_max, spacing, std::move(values)));
}

Result<AssetGrid>
AssetGrid::aligned_to(double s_max, std::int64_t nodes, double level, const char* name) {
    if (!(level > 0.0) || !std::isfinite(level)) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("{} must be positive and finite, got {}", name, level),
            kContext);
    }
    if (!(s_max > level)) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("s_max ({}) must exceed the {} ({}), or it lies outside the grid and the "
                        "solution near it is never resolved",
                        s_max,
                        name,
                        level),
            kContext);
    }
    if (nodes < kMinimumNodes) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format(
                "a grid needs at least {} nodes, but {} were requested", kMinimumNodes, nodes),
            kContext);
    }

    // Choose the spacing so that both the level and S_max land on nodes.
    //
    // Aim for the requested spacing, then round the level's node index to the
    // nearest integer and recompute dS from it. The level then sits exactly on node
    // `level_index`, and S_max on the last node, at the cost of a node count that
    // differs slightly from the request.
    const double target_spacing = s_max / static_cast<double>(nodes - 1);
    const auto level_index = static_cast<std::int64_t>(std::round(level / target_spacing));
    if (level_index < 1) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("the grid is too coarse to place the {} ({}) on a node: at a spacing of {} "
                        "it would fall on node {}, which is the S=0 boundary. Use more nodes or a "
                        "smaller s_max.",
                        name,
                        level,
                        target_spacing,
                        level_index),
            kContext);
    }

    const double spacing = level / static_cast<double>(level_index);
    const auto interval_count = static_cast<std::int64_t>(std::round(s_max / spacing));
    if (interval_count < kMinimumNodes - 1) {
        return Result<AssetGrid>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("aligning the {} ({}) yields a spacing of {}, which spans s_max ({}) in "
                        "only {} intervals -- too few to step",
                        name,
                        level,
                        spacing,
                        s_max,
                        interval_count),
            kContext);
    }

    // S_max moves to the nearest multiple of the spacing. Reported through the
    // grid's own s_max() rather than silently differing from the request: the
    // boundary condition is applied at whatever the top node actually is, and a
    // caller comparing against their requested s_max should see the truth.
    const double aligned_s_max = spacing * static_cast<double>(interval_count);

    const auto count = static_cast<std::size_t>(interval_count + 1);
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<double>(i) * spacing;
    }
    values.front() = 0.0;
    values.back() = aligned_s_max;
    // The level is pinned exactly too. i*dS reproduces it to within an ulp, and the
    // whole purpose of this grid is that it sits *exactly* on a node rather than a
    // rounding error away from one.
    values[static_cast<std::size_t>(level_index)] = level;

    return Result<AssetGrid>::success(AssetGrid(aligned_s_max, spacing, std::move(values)));
}

Result<AssetGrid> AssetGrid::with_strike_on_node(double s_max, std::int64_t nodes, double strike) {
    return aligned_to(s_max, nodes, strike, "strike");
}

Result<AssetGrid>
AssetGrid::with_barrier_on_node(double s_max, std::int64_t nodes, double barrier) {
    return aligned_to(s_max, nodes, barrier, "barrier");
}

double AssetGrid::at(std::int64_t index) const noexcept {
    return values_[static_cast<std::size_t>(index)];
}

std::optional<std::int64_t> AssetGrid::nearest_index(double s) const noexcept {
    if (!std::isfinite(s) || s < 0.0 || s > s_max_) {
        return std::nullopt;
    }
    const auto index = static_cast<std::int64_t>(std::round(s / spacing_));
    return std::min(index, nodes() - 1);
}

}  // namespace diffusionworks

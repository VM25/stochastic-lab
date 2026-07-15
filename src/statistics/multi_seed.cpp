#include <diffusionworks/statistics/multi_seed.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "multi_seed";

}  // namespace

Result<MultiSeedSummary> summarize_seeds(const std::vector<SeedResult>& results,
                                         std::optional<double> reference) {
    if (results.size() < 2) {
        return Result<MultiSeedSummary>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("multi-seed aggregation needs at least 2 independent replications but "
                        "received {}; dispersion cannot be measured from a single run",
                        results.size()),
            kContext);
    }

    // Repeated seeds are not independent replications. Left unchecked they would
    // shrink the apparent dispersion, making the estimator look more reliable
    // precisely because the evidence is weaker.
    std::set<std::uint64_t> distinct_seeds;
    for (const SeedResult& result : results) {
        if (!distinct_seeds.insert(result.seed).second) {
            return Result<MultiSeedSummary>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("seed {} appears more than once; replications must be independent, "
                            "and repeating a seed understates the estimator's dispersion",
                            result.seed),
                kContext);
        }
        if (!std::isfinite(result.estimate)) {
            return Result<MultiSeedSummary>::failure(
                ErrorCode::NonFiniteValue,
                fmt::format("estimate from seed {} is not finite", result.seed),
                kContext);
        }
    }

    OnlineMoments moments;
    OnlineMoments squared_errors;
    OnlineMoments signed_errors;

    double minimum = results.front().estimate;
    double maximum = results.front().estimate;

    for (const SeedResult& result : results) {
        moments.add(result.estimate);
        minimum = std::min(minimum, result.estimate);
        maximum = std::max(maximum, result.estimate);

        if (reference.has_value()) {
            const double error = result.estimate - *reference;
            squared_errors.add(error * error);
            signed_errors.add(error);
        }
    }

    auto variance = moments.sample_variance();
    if (!variance) {
        return Result<MultiSeedSummary>::failure(std::move(variance).error());
    }
    auto error = moments.standard_error();
    if (!error) {
        return Result<MultiSeedSummary>::failure(std::move(error).error());
    }

    MultiSeedSummary summary;
    summary.seed_count = moments.count();
    summary.mean = moments.mean();
    summary.standard_deviation = std::sqrt(variance.value());
    summary.standard_error = error.value();
    summary.minimum = minimum;
    summary.maximum = maximum;

    if (reference.has_value()) {
        // RMSE uses the mean of the squared errors, not the unbiased variance:
        // it is defined about the reference rather than about the sample mean,
        // so there is no degree of freedom to lose.
        summary.rmse = std::sqrt(squared_errors.mean());
        summary.bias = signed_errors.mean();
    }

    return Result<MultiSeedSummary>::success(summary);
}

}  // namespace diffusionworks

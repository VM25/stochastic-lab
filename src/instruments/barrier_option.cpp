#include <diffusionworks/instruments/barrier_option.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "BarrierOption";

}  // namespace

std::string_view to_string(BarrierType type) noexcept {
    switch (type) {
        case BarrierType::DownAndOut:
            return "down_and_out";
        case BarrierType::UpAndOut:
            return "up_and_out";
        case BarrierType::DownAndIn:
            return "down_and_in";
        case BarrierType::UpAndIn:
            return "up_and_in";
    }
    return "unknown";
}

std::optional<BarrierType> parse_barrier_type(std::string_view text) noexcept {
    if (text == "down_and_out") {
        return BarrierType::DownAndOut;
    }
    if (text == "up_and_out") {
        return BarrierType::UpAndOut;
    }
    if (text == "down_and_in") {
        return BarrierType::DownAndIn;
    }
    if (text == "up_and_in") {
        return BarrierType::UpAndIn;
    }
    return std::nullopt;
}

std::string_view to_string(MonitoringConvention convention) noexcept {
    switch (convention) {
        case MonitoringConvention::Continuous:
            return "continuous";
        case MonitoringConvention::Discrete:
            return "discrete";
        case MonitoringConvention::BrownianBridge:
            return "brownian_bridge";
    }
    return "unknown";
}

std::optional<MonitoringConvention> parse_monitoring_convention(std::string_view text) noexcept {
    if (text == "continuous") {
        return MonitoringConvention::Continuous;
    }
    if (text == "discrete") {
        return MonitoringConvention::Discrete;
    }
    if (text == "brownian_bridge") {
        return MonitoringConvention::BrownianBridge;
    }
    return std::nullopt;
}

BarrierOption::BarrierOption(OptionType type,
                             BarrierType barrier_type,
                             double strike,
                             double barrier,
                             double maturity,
                             MonitoringConvention convention,
                             std::optional<std::int64_t> monitoring_count) noexcept
    : type_(type), barrier_type_(barrier_type), strike_(strike), barrier_(barrier),
      maturity_(maturity), convention_(convention), monitoring_count_(monitoring_count) {}

Result<BarrierOption> BarrierOption::create(OptionType type,
                                            BarrierType barrier_type,
                                            double strike,
                                            double barrier,
                                            double maturity,
                                            MonitoringConvention convention,
                                            std::optional<std::int64_t> monitoring_count) {
    if (!(strike > 0.0) || !std::isfinite(strike)) {
        return Result<BarrierOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("strike must be positive and finite, got {}", strike),
            kContext);
    }
    if (!(barrier > 0.0) || !std::isfinite(barrier)) {
        return Result<BarrierOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("barrier must be positive and finite, got {}", barrier),
            kContext);
    }
    if (!(maturity >= 0.0) || !std::isfinite(maturity)) {
        return Result<BarrierOption>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("maturity must be non-negative and finite, got {}", maturity),
            kContext);
    }

    if (convention == MonitoringConvention::Continuous) {
        if (monitoring_count.has_value()) {
            // Refused rather than ignored. A continuously monitored barrier has no
            // observation dates; accepting a count would let a caller believe
            // theirs was used, and the resulting price would be a perfectly
            // plausible answer to a different contract.
            return Result<BarrierOption>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("continuous monitoring has no observation dates, but a count of {} was "
                            "supplied. Use the discrete or brownian_bridge convention if the "
                            "barrier is observed at fixes.",
                            *monitoring_count),
                kContext);
        }
    } else {
        if (!monitoring_count.has_value() || *monitoring_count < 1) {
            return Result<BarrierOption>::failure(
                ErrorCode::InvalidArgument,
                fmt::format("{} monitoring needs at least one observation date, but {} was given",
                            to_string(convention),
                            monitoring_count.has_value() ? std::to_string(*monitoring_count)
                                                         : "none"),
                kContext);
        }
    }

    // The barrier's position relative to the spot is deliberately not checked:
    // the spot is not a contract term, and an already-breached barrier is a
    // limiting case with a well-defined price (zero, for a knock-out), not an
    // invalid contract.
    return Result<BarrierOption>::success(
        BarrierOption(type, barrier_type, strike, barrier, maturity, convention, monitoring_count));
}

bool BarrierOption::breaches(double s) const noexcept {
    // Touching the barrier counts as breaching it. The standard convention, and
    // the one that matters on a PDE grid: with the barrier on a node, "touching"
    // is not a measure-zero event but every path that reaches that node.
    return is_down_barrier(barrier_type_) ? s <= barrier_ : s >= barrier_;
}

double BarrierOption::payoff(double terminal, bool barrier_hit) const noexcept {
    const double intrinsic = type_ == OptionType::Call ? std::max(terminal - strike_, 0.0)
                                                       : std::max(strike_ - terminal, 0.0);

    // A knock-out pays unless hit; a knock-in pays only if hit. Written as one
    // condition rather than a four-way switch, so the in/out duality is visible
    // and the two cannot drift apart.
    const bool pays = is_knock_out(barrier_type_) ? !barrier_hit : barrier_hit;
    return pays ? intrinsic : 0.0;
}

EuropeanOption BarrierOption::vanilla() const {
    // Cannot fail: the barrier option's strike and maturity were validated against
    // the same conditions at construction.
    return EuropeanOption::create(type_, strike_, maturity_).value();
}

}  // namespace diffusionworks

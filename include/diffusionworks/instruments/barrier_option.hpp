#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace diffusionworks {

/// Which side the barrier sits on, and what crossing it does.
enum class BarrierType : std::uint8_t {
    /// Knocked out if the minimum falls to or below B. Requires S_0 > B.
    DownAndOut,

    /// Knocked out if the maximum rises to or above B. Requires S_0 < B.
    UpAndOut,

    /// Knocked in if the minimum falls to or below B. Worthless otherwise.
    DownAndIn,

    /// Knocked in if the maximum rises to or above B.
    UpAndIn,
};

[[nodiscard]] std::string_view to_string(BarrierType type) noexcept;

[[nodiscard]] std::optional<BarrierType> parse_barrier_type(std::string_view text) noexcept;

[[nodiscard]] constexpr bool is_knock_out(BarrierType type) noexcept {
    return type == BarrierType::DownAndOut || type == BarrierType::UpAndOut;
}

[[nodiscard]] constexpr bool is_down_barrier(BarrierType type) noexcept {
    return type == BarrierType::DownAndOut || type == BarrierType::DownAndIn;
}

/// How the barrier is observed.
///
/// This is a contract term, not a numerical detail, and it changes the price by
/// more than most model choices do. MATHEMATICAL-SPEC section 5 requires it be
/// explicit, and this enum is why: there is no default that is safe to assume.
enum class MonitoringConvention : std::uint8_t {
    /// The barrier is watched at every instant.
    ///
    /// The textbook closed forms (Merton, Reiner-Rubinstein) price this. It is
    /// also the convention almost no traded contract uses -- a real barrier is
    /// observed at fixes, usually daily closes -- so a continuous price is
    /// systematically *cheaper* for a knock-out than the contract it is meant to
    /// value, because continuous monitoring finds every excursion and discrete
    /// monitoring misses the ones between observations.
    Continuous,

    /// The barrier is checked only at `monitoring_count` equally spaced dates.
    ///
    /// What a simulation naturally does, and what a real contract usually says.
    /// The bias against the continuous price is O(1/sqrt(m)) in the number of
    /// monitoring dates -- famously slow, which is what makes EXP-07 worth running
    /// rather than assuming daily monitoring is "close enough" to continuous.
    Discrete,

    /// Discrete dates, with a Brownian-bridge correction for the probability that
    /// the path crossed *between* observations.
    ///
    /// Between two observed points a Brownian bridge has a known crossing
    /// probability, so a simulation can account for the excursions it did not
    /// observe instead of pretending they did not happen. This converges to the
    /// continuous price as the correction becomes exact, and it corrects the
    /// *discretisation of the observation*, not the discretisation of the path --
    /// two different things that are easy to conflate.
    BrownianBridge,
};

[[nodiscard]] std::string_view to_string(MonitoringConvention convention) noexcept;

[[nodiscard]] std::optional<MonitoringConvention>
parse_monitoring_convention(std::string_view text) noexcept;

/// A single-barrier European option.
///
/// Contract terms only, per ADR-007.
class BarrierOption {
public:
    /// Validates and constructs.
    ///
    /// Requires strike > 0, barrier > 0, maturity >= 0, all finite. Discrete and
    /// bridge conventions require monitoring_count >= 1; the continuous convention
    /// requires it absent, since a continuously monitored barrier has no
    /// observation dates and accepting a count would invite the belief that it
    /// was used.
    ///
    /// The barrier's position relative to the spot is *not* validated here: an
    /// already-breached contract is a required limiting case
    /// (MATHEMATICAL-SPEC section 18), not an error. A down-and-out with S_0 <= B
    /// is worth zero, and that is a price rather than a rejection. Engines decide.
    [[nodiscard]] static Result<BarrierOption> create(OptionType type,
                                                      BarrierType barrier_type,
                                                      double strike,
                                                      double barrier,
                                                      double maturity,
                                                      MonitoringConvention convention,
                                                      std::optional<std::int64_t> monitoring_count);

    [[nodiscard]] OptionType type() const noexcept { return type_; }

    [[nodiscard]] BarrierType barrier_type() const noexcept { return barrier_type_; }

    [[nodiscard]] double strike() const noexcept { return strike_; }

    [[nodiscard]] double barrier() const noexcept { return barrier_; }

    [[nodiscard]] double maturity() const noexcept { return maturity_; }

    [[nodiscard]] MonitoringConvention convention() const noexcept { return convention_; }

    /// Number of monitoring dates. Absent for continuous monitoring.
    [[nodiscard]] std::optional<std::int64_t> monitoring_count() const noexcept {
        return monitoring_count_;
    }

    /// The terminal payoff, given whether the barrier was hit.
    ///
    /// The knock decision is the caller's: the instrument does not know how the
    /// path was monitored, and folding that in here would put valuation logic in a
    /// contract.
    [[nodiscard]] double payoff(double terminal, bool barrier_hit) const noexcept;

    /// Whether `s` breaches the barrier.
    ///
    /// Breach is `<=` for a down barrier and `>=` for an up barrier. The boundary
    /// itself counts as breached, which is the standard convention and is stated
    /// because the alternative is defensible and produces a different price for a
    /// path that touches B exactly. On a PDE grid with the barrier on a node that
    /// is not a measure-zero event -- it is every path that reaches the node.
    [[nodiscard]] bool breaches(double s) const noexcept;

    /// The vanilla option this barrier contract is built on.
    ///
    /// Knock-in plus knock-out equals this, when the contracts and monitoring
    /// conventions match (MATHEMATICAL-SPEC section 18). That identity is the
    /// sharpest available check on a barrier engine, because it holds exactly and
    /// independently of the model.
    [[nodiscard]] EuropeanOption vanilla() const;

private:
    BarrierOption(OptionType type,
                  BarrierType barrier_type,
                  double strike,
                  double barrier,
                  double maturity,
                  MonitoringConvention convention,
                  std::optional<std::int64_t> monitoring_count) noexcept;

    OptionType type_{};
    BarrierType barrier_type_{};
    double strike_{};
    double barrier_{};
    double maturity_{};
    MonitoringConvention convention_{};
    std::optional<std::int64_t> monitoring_count_;
};

}  // namespace diffusionworks

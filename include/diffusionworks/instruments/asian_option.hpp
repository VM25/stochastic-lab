#pragma once

#include <diffusionworks/core/result.hpp>
#include <diffusionworks/instruments/european_option.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace diffusionworks {

/// How the average is taken.
enum class AveragingType : std::uint8_t {
    /// \f$ A = \frac{1}{M}\sum_i S_{t_i} \f$. No closed form under GBM: the sum
    /// of lognormals is not lognormal, which is precisely why this instrument
    /// needs Monte Carlo.
    Arithmetic,

    /// \f$ G = \exp\!\big(\frac{1}{M}\sum_i \ln S_{t_i}\big) \f$. Lognormal under
    /// GBM, hence a closed form, hence its use as a control variate for the
    /// arithmetic average it closely tracks (MATHEMATICAL-SPEC section 11).
    Geometric,
};

[[nodiscard]] std::string_view to_string(AveragingType averaging) noexcept;

[[nodiscard]] std::optional<AveragingType> parse_averaging_type(std::string_view text) noexcept;

/// A fixed-strike Asian option with discrete monitoring.
///
/// Monitoring convention, stated because leaving it implicit is itself a failure
/// (VALIDATION-PLAN section 11): the average is taken over M equally spaced dates
///
/// \f[ t_i = \frac{i\,T}{M}, \qquad i = 1,\dots,M \f]
///
/// The initial spot is **excluded**. It is known at inception and carries no
/// uncertainty, so including it would blend a constant into the average and
/// change the option's value. The final date is maturity, so the terminal state
/// is always observed.
///
/// Contract terms only (ADR-007): the averaging is defined here, the pricing is
/// not.
class AsianOption {
public:
    /// Validates and constructs.
    ///
    /// Requires strike > 0, maturity > 0, and monitoring_count >= 1.
    ///
    /// Unlike EuropeanOption, maturity must be strictly positive: an average over
    /// zero dates spanning zero time is not a limiting case with a defined value,
    /// it is a contract that does not exist.
    [[nodiscard]] static Result<AsianOption> create(OptionType type,
                                                    AveragingType averaging,
                                                    double strike,
                                                    double maturity,
                                                    std::int64_t monitoring_count);

    [[nodiscard]] OptionType type() const noexcept { return type_; }

    [[nodiscard]] AveragingType averaging() const noexcept { return averaging_; }

    [[nodiscard]] double strike() const noexcept { return strike_; }

    [[nodiscard]] double maturity() const noexcept { return maturity_; }

    [[nodiscard]] std::int64_t monitoring_count() const noexcept { return monitoring_count_; }

    /// The i-th monitoring date, for i = 1..M.
    [[nodiscard]] double monitoring_time(std::int64_t index) const noexcept;

    /// Payoff given the realised average: \f$ (A-K)^+ \f$ or \f$ (K-A)^+ \f$.
    [[nodiscard]] double payoff(double average) const noexcept;

private:
    AsianOption(OptionType type,
                AveragingType averaging,
                double strike,
                double maturity,
                std::int64_t monitoring_count) noexcept
        : type_(type), averaging_(averaging), strike_(strike), maturity_(maturity),
          monitoring_count_(monitoring_count) {}

    OptionType type_;
    AveragingType averaging_;
    double strike_;
    double maturity_;
    std::int64_t monitoring_count_;
};

}  // namespace diffusionworks

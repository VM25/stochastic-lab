#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace diffusionworks {

/// A two-sided confidence interval for a stochastic estimate.
struct ConfidenceInterval {
    double lower{};
    double upper{};

    /// Nominal coverage, e.g. 0.95. Reported alongside the bounds because an
    /// interval without its level is not interpretable.
    double level{};

    [[nodiscard]] double width() const noexcept { return upper - lower; }

    [[nodiscard]] bool contains(double x) const noexcept { return x >= lower && x <= upper; }
};

/// First- and second-order sensitivities.
///
/// Units, stated explicitly because the alternative is a silent factor-of-100
/// error (MATHEMATICAL-SPEC section 3):
///
///   delta  dV/dS       per unit of spot
///   gamma  d2V/dS2     per unit of spot squared
///   vega   dV/dsigma   per unit of volatility (NOT per volatility point;
///                      divide by 100 for a 1% move)
///   theta  dV/dt       per year (negative for a long option, calendar time)
///   rho    dV/dr       per unit of rate (NOT per basis point or per 1%)
///
/// Per-unit is the derivative as written in the specification. Scaling happens
/// at presentation, never in the engine.
struct Greeks {
    double delta{};
    double gamma{};
    double vega{};
    double theta{};
    double rho{};
};

/// A named diagnostic emitted alongside a result.
///
/// Diagnostics carry the evidence a reader needs to judge a number: which branch
/// a formula took, how many iterations a solver used, whether the Feller
/// condition held. The variant lets a diagnostic be numeric, integral, boolean,
/// or textual without forcing everything through double, where an iteration
/// count or a "satisfied"/"violated" flag would be misrepresented.
struct Diagnostic {
    using Value = std::variant<double, std::int64_t, bool, std::string>;

    std::string name;
    Value value;

    Diagnostic(std::string diagnostic_name, Value diagnostic_value)
        : name(std::move(diagnostic_name)), value(std::move(diagnostic_value)) {}
};

/// The outcome of a successful valuation.
///
/// Per ADR-009 no engine returns a bare double. A number without its method,
/// uncertainty, diagnostics, and warnings cannot be judged, and a plausible
/// number without evidence is exactly what this project exists to distrust.
///
/// Fields that do not apply to a method stay empty rather than being filled with
/// a placeholder (TECHNICAL-DESIGN section 19). An analytic price has no standard
/// error, and reporting 0.0 there would claim perfect certainty rather than
/// inapplicability.
struct PricingResult {
    /// The value. Present on every successful result.
    double value{};

    /// Identifies the method that produced `value`, e.g. "black_scholes_analytic".
    std::string method;

    /// Sampling uncertainty. Present only for stochastic estimators.
    std::optional<double> standard_error;

    /// Present only for stochastic estimators.
    std::optional<ConfidenceInterval> confidence_interval;

    /// Present when the method computed sensitivities.
    std::optional<Greeks> greeks;

    /// Conditions that do not invalidate the result but qualify it -- a
    /// degenerate limit, a boundary regime, a parameter near a known weak spot.
    /// A warning must never stand in for a failure: an invalid result is an
    /// Error, not a warning.
    std::vector<std::string> warnings;

    /// Evidence about how the value was obtained.
    std::vector<Diagnostic> diagnostics;

    /// Wall-clock seconds spent in the valuation. Excludes setup and I/O so that
    /// it measures the method rather than the harness.
    double runtime_seconds{};

    void add_warning(std::string message) { warnings.push_back(std::move(message)); }

    void add_diagnostic(std::string diagnostic_name, Diagnostic::Value diagnostic_value) {
        diagnostics.emplace_back(std::move(diagnostic_name), std::move(diagnostic_value));
    }

    [[nodiscard]] bool has_warnings() const noexcept { return !warnings.empty(); }
};

}  // namespace diffusionworks

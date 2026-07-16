#include "simulate_command.hpp"

#include <diffusionworks/config/domain_parsers.hpp>
#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/simulation/gbm_path_generator.hpp>
#include <diffusionworks/statistics/online_moments.hpp>

#include "output.hpp"

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "simulate";

/// The closed-form terminal law of GBM, against which the sample is judged.
struct TheoreticalMoments {
    double mean{};
    double variance{};
    double log_mean{};
    double log_variance{};
};

[[nodiscard]] TheoreticalMoments
theoretical_moments(const MarketState& market, const BlackScholesModel& model, double maturity) {
    const double carry = market.rate() - market.dividend_yield();
    const double variance_rate = model.volatility() * model.volatility();

    TheoreticalMoments moments;
    // E[S_T] = S_0 e^{(r-q)T}
    moments.mean = market.spot() * std::exp(carry * maturity);
    // Var[S_T] = S_0^2 e^{2(r-q)T} (e^{sigma^2 T} - 1)
    // expm1 rather than exp(x) - 1: for a short maturity or a low volatility the
    // exponent is tiny and the subtraction would cancel away the answer.
    moments.variance = market.spot() * market.spot() * std::exp(2.0 * carry * maturity) *
                       std::expm1(variance_rate * maturity);
    // log(S_T/S_0) ~ N((r - q - sigma^2/2)T, sigma^2 T)
    moments.log_mean = (carry - 0.5 * variance_rate) * maturity;
    moments.log_variance = variance_rate * maturity;
    return moments;
}

}  // namespace

Result<nlohmann::json> run_simulate(const ConfigDocument& config, const Options& options) {
    const auto start = std::chrono::steady_clock::now();
    const ConfigNode root = config.root();

    const Status unknown = root.reject_unknown_keys(
        {"schema_version", "command", "market", "model", "method", "maturity", "output"});
    if (!unknown) {
        return Result<nlohmann::json>::failure(unknown.error());
    }

    auto market_node = root.object("market");
    if (!market_node) {
        return Result<nlohmann::json>::failure(std::move(market_node).error());
    }
    auto market = parse_market_state(market_node.value());
    if (!market) {
        return Result<nlohmann::json>::failure(std::move(market).error());
    }

    auto model_node = root.object("model");
    if (!model_node) {
        return Result<nlohmann::json>::failure(std::move(model_node).error());
    }
    auto model = parse_black_scholes_model(model_node.value());
    if (!model) {
        return Result<nlohmann::json>::failure(std::move(model).error());
    }

    // simulate has no instrument, so the horizon is its own field rather than a
    // contract's maturity.
    auto maturity = root.positive_number("maturity");
    if (!maturity) {
        return Result<nlohmann::json>::failure(std::move(maturity).error());
    }

    auto method_node = root.object("method");
    if (!method_node) {
        return Result<nlohmann::json>::failure(std::move(method_node).error());
    }
    auto method_type = method_node.value().string("type");
    if (!method_type) {
        return Result<nlohmann::json>::failure(std::move(method_type).error());
    }
    if (method_type.value() != "monte_carlo") {
        return Result<nlohmann::json>::failure(
            ErrorCode::UnsupportedCombination,
            fmt::format("'method.type' is '{}'; simulate generates paths and requires "
                        "'monte_carlo'",
                        method_type.value()),
            kContext);
    }

    auto method = parse_monte_carlo_config(method_node.value(), options.seed);
    if (!method) {
        return Result<nlohmann::json>::failure(std::move(method).error());
    }
    const MonteCarloConfig& mc = method.value();

    if (mc.paths < 2) {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidArgument,
            fmt::format("paths must be at least 2 but is {}; a single path has no dispersion to "
                        "compare against theory",
                        mc.paths),
            kContext);
    }

    auto grid = TimeGrid::uniform(maturity.value(), mc.steps);
    if (!grid) {
        return Result<nlohmann::json>::failure(std::move(grid).error());
    }

    // --- Simulate -----------------------------------------------------------

    const GbmPathGenerator generator(market.value(), model.value(), grid.value(), mc.scheme);

    OnlineMoments terminal;
    OnlineMoments log_returns;
    std::int64_t non_positive_states = 0;
    std::int64_t paths_with_non_positive_states = 0;
    std::int64_t log_domain_failures = 0;

    std::vector<double> path(generator.path_size());
    for (std::int64_t index = 0; index < mc.paths; ++index) {
        auto diagnostics = generator.generate(mc.seed, static_cast<std::uint64_t>(index), path);
        if (!diagnostics) {
            // A non-finite state fails the run. It is not a statistic to be
            // summarised: every moment computed from it would be meaningless.
            return Result<nlohmann::json>::failure(std::move(diagnostics).error());
        }
        if (diagnostics.value().non_positive_states > 0) {
            non_positive_states += diagnostics.value().non_positive_states;
            ++paths_with_non_positive_states;
        }

        terminal.add(path.back());

        // A non-positive terminal state has no log. Counted rather than skipped
        // silently, so the log-moment comparison below is read against how many
        // paths it actually covers.
        if (path.back() > 0.0) {
            log_returns.add(std::log(path.back() / market.value().spot()));
        } else {
            ++log_domain_failures;
        }
    }

    const double runtime =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // --- Compare against theory ---------------------------------------------

    const TheoreticalMoments theory =
        theoretical_moments(market.value(), model.value(), maturity.value());

    auto terminal_variance = terminal.sample_variance();
    if (!terminal_variance) {
        return Result<nlohmann::json>::failure(std::move(terminal_variance).error());
    }

    // Deviations are reported in standard errors, not absolutely. An absolute
    // difference is uninterpretable without knowing the sampling noise: 0.01 is
    // damning at 10^8 paths and meaningless at 100.
    const double mean_standard_error = std::sqrt(theory.variance / static_cast<double>(mc.paths));
    const double mean_deviation_in_errors = (terminal.mean() - theory.mean) / mean_standard_error;

    nlohmann::json document;
    document["status"] = "ok";
    document["command"] = "simulate";
    document["market"] = nlohmann::json{
        {"spot", market.value().spot()},
        {"rate", market.value().rate()},
        {"dividend_yield", market.value().dividend_yield()},
    };
    document["model"] = nlohmann::json{
        {"type", "black_scholes"},
        {"volatility", model.value().volatility()},
    };
    document["method"] = nlohmann::json{
        {"type", "monte_carlo"},
        {"paths", mc.paths},
        {"steps", mc.steps},
        {"scheme", to_string(mc.scheme)},
        {"seed", mc.seed},
        {"maturity", maturity.value()},
    };

    nlohmann::json result;
    result["terminal"] = nlohmann::json{
        {"sample_mean", terminal.mean()},
        {"sample_variance", terminal_variance.value()},
        {"theoretical_mean", theory.mean},
        {"theoretical_variance", theory.variance},
        {"mean_standard_error", mean_standard_error},
        {"mean_deviation_in_standard_errors", mean_deviation_in_errors},
    };

    if (log_returns.count() >= 2) {
        auto log_variance = log_returns.sample_variance();
        if (!log_variance) {
            return Result<nlohmann::json>::failure(std::move(log_variance).error());
        }
        // The log domain isolates the drift and diffusion terms without the
        // lognormal's heavy tail inflating the sampling error, so it is the
        // sharper of the two comparisons.
        result["log_return"] = nlohmann::json{
            {"sample_mean", log_returns.mean()},
            {"sample_variance", log_variance.value()},
            {"theoretical_mean", theory.log_mean},
            {"theoretical_variance", theory.log_variance},
            {"paths_included", log_returns.count()},
        };
    }

    result["diagnostics"] = nlohmann::json{
        {"non_positive_states", non_positive_states},
        {"paths_with_non_positive_states", paths_with_non_positive_states},
        {"log_domain_failures", log_domain_failures},
    };

    nlohmann::json warnings = nlohmann::json::array();
    if (non_positive_states > 0) {
        warnings.push_back(fmt::format(
            "{} state(s) across {} path(s) reached zero or went negative under the {} scheme. "
            "Euler-Maruyama and Milstein step the price itself and can cross zero; the exact "
            "scheme cannot.",
            non_positive_states,
            paths_with_non_positive_states,
            to_string(mc.scheme)));
    }
    if (log_domain_failures > 0) {
        warnings.push_back(
            fmt::format("{} path(s) ended at a non-positive state and are excluded from the "
                        "log-return moments, which therefore describe a truncated sample.",
                        log_domain_failures));
    }
    // Reported in standard errors so the reader can judge it. Three is a
    // conventional line, not a verdict: at these path counts a genuine defect
    // shows up at tens of standard errors, and this is a prompt to look rather
    // than a failure.
    if (std::abs(mean_deviation_in_errors) > 3.0) {
        warnings.push_back(fmt::format(
            "the sample mean deviates from theory by {:.1f} standard errors, which is unlikely "
            "under correct simulation. For the exact scheme this suggests a defect; for the "
            "explicit schemes it is discretisation bias and shrinks with more steps.",
            mean_deviation_in_errors));
    }
    result["warnings"] = std::move(warnings);
    result["runtime_seconds"] = runtime;

    document["result"] = std::move(result);
    document["configuration"] = config.json();
    if (!config.source().empty()) {
        document["configuration_source"] = config.source().string();
    }
    document["build_metadata"] = to_json(collect_build_info());

    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli

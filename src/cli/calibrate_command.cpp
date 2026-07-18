#include "calibrate_command.hpp"

#include <diffusionworks/calibration/heston_calibrator.hpp>
#include <diffusionworks/calibration/volatility_surface.hpp>
#include <diffusionworks/config/domain_parsers.hpp>
#include <diffusionworks/core/build_info.hpp>

#include "output.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <string>
#include <vector>

namespace diffusionworks::cli {
namespace {

constexpr const char* kContext = "calibrate";

[[nodiscard]] nlohmann::json to_json(const HestonParameters& p) {
    return nlohmann::json{{"initial_variance", p.initial_variance},
                          {"mean_reversion", p.mean_reversion},
                          {"long_run_variance", p.long_run_variance},
                          {"vol_of_variance", p.vol_of_variance},
                          {"correlation", p.correlation}};
}

/// Parses one Heston parameter vector (an object with the five named fields).
[[nodiscard]] Result<HestonParameters> parse_parameters(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys({"initial_variance",
                                                     "mean_reversion",
                                                     "long_run_variance",
                                                     "vol_of_variance",
                                                     "correlation"});
    if (!unknown) {
        return Result<HestonParameters>::failure(unknown.error());
    }
    const auto v0 = node.number("initial_variance");
    const auto kappa = node.number("mean_reversion");
    const auto theta = node.number("long_run_variance");
    const auto xi = node.number("vol_of_variance");
    const auto rho = node.number("correlation");
    if (!v0) {
        return Result<HestonParameters>::failure(v0.error());
    }
    if (!kappa) {
        return Result<HestonParameters>::failure(kappa.error());
    }
    if (!theta) {
        return Result<HestonParameters>::failure(theta.error());
    }
    if (!xi) {
        return Result<HestonParameters>::failure(xi.error());
    }
    if (!rho) {
        return Result<HestonParameters>::failure(rho.error());
    }
    return Result<HestonParameters>::success(HestonParameters{.initial_variance = v0.value(),
                                                              .mean_reversion = kappa.value(),
                                                              .long_run_variance = theta.value(),
                                                              .vol_of_variance = xi.value(),
                                                              .correlation = rho.value()});
}

/// Parses one implied-volatility quote.
[[nodiscard]] Result<ImpliedVolatilityQuote> parse_quote(const ConfigNode& node) {
    const Status unknown = node.reject_unknown_keys(
        {"option_type", "strike", "maturity", "implied_volatility", "weight"});
    if (!unknown) {
        return Result<ImpliedVolatilityQuote>::failure(unknown.error());
    }
    ImpliedVolatilityQuote quote;
    const auto type = node.string_or("option_type", std::string("call"));
    if (!type) {
        return Result<ImpliedVolatilityQuote>::failure(type.error());
    }
    const auto parsed_type = parse_option_type(type.value());
    if (!parsed_type.has_value()) {
        return Result<ImpliedVolatilityQuote>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("option_type must be 'call' or 'put', got '{}'", type.value()),
            kContext);
    }
    quote.type = parsed_type.value();

    const auto strike = node.positive_number("strike");
    if (!strike) {
        return Result<ImpliedVolatilityQuote>::failure(strike.error());
    }
    quote.strike = strike.value();
    const auto maturity = node.positive_number("maturity");
    if (!maturity) {
        return Result<ImpliedVolatilityQuote>::failure(maturity.error());
    }
    quote.maturity = maturity.value();
    const auto iv = node.positive_number("implied_volatility");
    if (!iv) {
        return Result<ImpliedVolatilityQuote>::failure(iv.error());
    }
    quote.implied_volatility = iv.value();
    const auto weight = node.number_or("weight", 1.0);
    if (!weight) {
        return Result<ImpliedVolatilityQuote>::failure(weight.error());
    }
    if (weight.value() < 0.0) {
        return Result<ImpliedVolatilityQuote>::failure(
            ErrorCode::InvalidConfiguration, "a quote weight must be non-negative", kContext);
    }
    quote.weight = weight.value();
    return Result<ImpliedVolatilityQuote>::success(quote);
}

[[nodiscard]] nlohmann::json residual_to_json(const QuoteResidual& r) {
    return nlohmann::json{{"option_type", to_string(r.type)},
                          {"strike", r.strike},
                          {"maturity", r.maturity},
                          {"weight", r.weight},
                          {"market_price", r.market_price},
                          {"model_price", r.model_price},
                          {"market_implied_volatility", r.market_implied_volatility},
                          {"model_implied_volatility", r.model_implied_volatility}};
}

[[nodiscard]] nlohmann::json start_to_json(const CalibrationStart& s) {
    return nlohmann::json{{"started", s.started},
                          {"note", s.note},
                          {"status", to_string(s.status)},
                          {"objective", s.objective_value},
                          {"implied_vol_rmse", s.implied_vol_rmse},
                          {"price_rmse", s.price_rmse},
                          {"quotes_failed", s.quotes_failed},
                          {"iterations", s.iterations},
                          {"function_evaluations", s.function_evaluations},
                          {"initial", to_json(s.initial)},
                          {"calibrated", to_json(s.calibrated)}};
}

}  // namespace

Result<nlohmann::json> run_calibrate(const ConfigDocument& config, const Options& options) {
    const ConfigNode root = config.root();

    const Status unknown = root.reject_unknown_keys(
        {"schema_version", "command", "market", "surface", "calibration", "output"});
    if (!unknown) {
        return Result<nlohmann::json>::failure(unknown.error());
    }

    // --- Market -------------------------------------------------------------
    auto market_node = root.object("market");
    if (!market_node) {
        return Result<nlohmann::json>::failure(std::move(market_node).error());
    }
    auto market = parse_market_state(market_node.value());
    if (!market) {
        return Result<nlohmann::json>::failure(std::move(market).error());
    }

    // --- Surface ------------------------------------------------------------
    auto surface_node = root.object("surface");
    if (!surface_node) {
        return Result<nlohmann::json>::failure(std::move(surface_node).error());
    }
    const Status surface_unknown =
        surface_node.value().reject_unknown_keys({"source", "as_of", "quotes"});
    if (!surface_unknown) {
        return Result<nlohmann::json>::failure(surface_unknown.error());
    }
    auto source = surface_node.value().string_or("source", std::string("unspecified"));
    if (!source) {
        return Result<nlohmann::json>::failure(std::move(source).error());
    }
    auto as_of = surface_node.value().string_or("as_of", std::string("unspecified"));
    if (!as_of) {
        return Result<nlohmann::json>::failure(std::move(as_of).error());
    }
    auto quotes_node = surface_node.value().array("quotes");
    if (!quotes_node) {
        return Result<nlohmann::json>::failure(std::move(quotes_node).error());
    }
    std::vector<ImpliedVolatilityQuote> quotes;
    for (std::size_t i = 0; i < quotes_node.value().size(); ++i) {
        auto element = quotes_node.value().at(i);
        if (!element) {
            return Result<nlohmann::json>::failure(std::move(element).error());
        }
        auto quote = parse_quote(element.value());
        if (!quote) {
            return Result<nlohmann::json>::failure(std::move(quote).error());
        }
        quotes.push_back(quote.value());
    }

    const auto surface = build_surface_from_implied_vols(market.value().spot(),
                                                         market.value().rate(),
                                                         market.value().dividend_yield(),
                                                         quotes,
                                                         source.value(),
                                                         as_of.value());
    if (!surface) {
        return Result<nlohmann::json>::failure(std::move(surface).error());
    }

    // --- Calibration configuration -----------------------------------------
    auto calibration_node = root.object("calibration");
    if (!calibration_node) {
        return Result<nlohmann::json>::failure(std::move(calibration_node).error());
    }
    const Status calibration_unknown = calibration_node.value().reject_unknown_keys(
        {"objective", "quadrature_nodes", "max_iterations", "initial_guesses"});
    if (!calibration_unknown) {
        return Result<nlohmann::json>::failure(calibration_unknown.error());
    }

    CalibrationConfig calibration;
    auto objective =
        calibration_node.value().string_or("objective", std::string("implied_volatility"));
    if (!objective) {
        return Result<nlohmann::json>::failure(std::move(objective).error());
    }
    if (objective.value() == "implied_volatility") {
        calibration.objective = CalibrationObjectiveType::ImpliedVolatility;
    } else if (objective.value() == "price") {
        calibration.objective = CalibrationObjectiveType::Price;
    } else {
        return Result<nlohmann::json>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("calibration.objective must be 'implied_volatility' or 'price', got '{}'",
                        objective.value()),
            kContext);
    }
    auto nodes = calibration_node.value().integer_or("quadrature_nodes",
                                                     calibration.pricing.quadrature_nodes);
    if (!nodes) {
        return Result<nlohmann::json>::failure(std::move(nodes).error());
    }
    calibration.pricing.quadrature_nodes = nodes.value();
    calibration.pricing.convergence_tolerance = 1.0e-5;
    auto iterations =
        calibration_node.value().integer_or("max_iterations", calibration.optimizer.max_iterations);
    if (!iterations) {
        return Result<nlohmann::json>::failure(std::move(iterations).error());
    }
    calibration.optimizer.max_iterations = static_cast<int>(iterations.value());

    auto guesses_node = calibration_node.value().array("initial_guesses");
    if (!guesses_node) {
        return Result<nlohmann::json>::failure(std::move(guesses_node).error());
    }
    for (std::size_t i = 0; i < guesses_node.value().size(); ++i) {
        auto element = guesses_node.value().at(i);
        if (!element) {
            return Result<nlohmann::json>::failure(std::move(element).error());
        }
        auto parsed = parse_parameters(element.value());
        if (!parsed) {
            return Result<nlohmann::json>::failure(std::move(parsed).error());
        }
        calibration.initial_guesses.push_back(parsed.value());
    }

    // --- Calibrate ----------------------------------------------------------
    const auto calibrated = calibrate_heston(surface.value(), calibration);
    if (!calibrated) {
        return Result<nlohmann::json>::failure(std::move(calibrated).error());
    }
    const CalibrationResult& result = calibrated.value();

    nlohmann::json starts = nlohmann::json::array();
    for (const CalibrationStart& s : result.starts) {
        starts.push_back(start_to_json(s));
    }
    nlohmann::json residuals = nlohmann::json::array();
    for (const QuoteResidual& r : result.best_residuals) {
        residuals.push_back(residual_to_json(r));
    }
    nlohmann::json similar = nlohmann::json::array();
    for (const HestonParameters& p : result.similar_fits) {
        similar.push_back(to_json(p));
    }

    // A calibration that ended in non-uniqueness, a non-converged best, or quotes it
    // could not fully evaluate is surfaced as a warning rather than a clean "ok": the
    // number is real but there is something the reader must weigh before using it.
    const bool clean = !result.non_unique && result.best.quotes_failed == 0 &&
                       result.best.status == NelderMeadStatus::Converged;

    nlohmann::json document;
    document["status"] = clean ? "ok" : "warning";
    document["command"] = "calibrate";
    document["market"] = nlohmann::json{{"spot", market.value().spot()},
                                        {"rate", market.value().rate()},
                                        {"dividend_yield", market.value().dividend_yield()}};
    document["surface"] = nlohmann::json{{"source", surface.value().source},
                                         {"as_of", surface.value().as_of},
                                         {"quote_count", surface.value().quotes.size()}};
    document["objective"] = to_string(calibration.objective);
    document["result"] =
        nlohmann::json{{"best", start_to_json(result.best)},
                       {"starts", std::move(starts)},
                       {"residual_surface", std::move(residuals)},
                       {"non_unique", result.non_unique},
                       {"max_similar_fit_distance", result.max_similar_fit_distance},
                       {"similar_fits", std::move(similar)},
                       {"started_count", result.started_count},
                       {"objective_min", result.objective_min},
                       {"objective_mean", result.objective_mean},
                       {"objective_max", result.objective_max},
                       {"parameter_mean", to_json(result.parameter_mean)},
                       {"parameter_dispersion", to_json(result.parameter_stddev)}};

    document["configuration"] = config.json();
    if (!config.source().empty()) {
        document["configuration_source"] = config.source().string();
    }
    if (options.threads.has_value() && options.threads.value() != 1) {
        document["warnings"] = nlohmann::json::array(
            {fmt::format("--threads {} was supplied but calibration runs single-threaded in this "
                         "build",
                         options.threads.value())});
    }
    document["build_metadata"] = to_json(collect_build_info());
    return Result<nlohmann::json>::success(std::move(document));
}

}  // namespace diffusionworks::cli

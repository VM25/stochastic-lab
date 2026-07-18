#pragma once

#include <diffusionworks/calibration/parameter_transform.hpp>
#include <diffusionworks/calibration/volatility_surface.hpp>
#include <diffusionworks/core/result.hpp>
#include <diffusionworks/engines/heston_analytic.hpp>
#include <diffusionworks/engines/implied_volatility.hpp>
#include <diffusionworks/numerics/nelder_mead.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace diffusionworks {

/// What the calibration objective measures the residual in.
enum class CalibrationObjectiveType : std::uint8_t {
    /// Weighted squared error in Black-Scholes implied volatility -- the primary
    /// objective of MATHEMATICAL-SPEC section 16. Each model price is inverted to its
    /// implied volatility and compared to the quote's.
    ImpliedVolatility,

    /// Weighted squared error in price. Cheaper -- no inversion per evaluation -- and
    /// permitted by the spec if documented, but it weights the deep, high-priced
    /// quotes far more heavily than the wings.
    Price,
};

[[nodiscard]] const char* to_string(CalibrationObjectiveType type) noexcept;

/// Euclidean distance between two parameter vectors with each coordinate scaled by its
/// bound range, so the five parameters -- which live on very different scales -- are
/// comparable. This is the metric the calibrator uses to decide whether two fits are
/// materially different, exposed because a recovery study needs the same measure to
/// compare a calibrated fit against the truth.
[[nodiscard]] double normalized_parameter_distance(const HestonParameters& a,
                                                   const HestonParameters& b,
                                                   const HestonParameterBounds& bounds);

/// Configuration for one calibration.
struct CalibrationConfig {
    HestonParameterBounds bounds{HestonParameterBounds::defaults()};

    /// The starting points. At least one is required, and using more than one -- none
    /// of them the true parameters -- is how the experiment probes whether the fit is
    /// unique. A single guess equal to the truth is the failure condition the catalog
    /// names, so multiple diverse guesses are the intended use.
    std::vector<HestonParameters> initial_guesses;

    CalibrationObjectiveType objective{CalibrationObjectiveType::ImpliedVolatility};

    NelderMeadConfig optimizer{};

    /// The pricing accuracy each objective evaluation runs at. Lighter than the
    /// reference default because calibration prices the surface thousands of times and
    /// does not need 1e-13 per evaluation -- but heavy enough that the doubling check
    /// does not refuse a feasible parameter set and hand the optimizer an artificial
    /// cliff.
    HestonAnalyticConfig pricing{.quadrature_nodes = 256, .convergence_tolerance = 1e-6};

    ImpliedVolatilityConfig implied_volatility{};

    /// The penalty each unpriceable or uninvertible quote adds to the objective. It
    /// must be large against a typical squared residual so a region the quadrature
    /// cannot resolve is discouraged, yet finite so the optimizer descends on the
    /// quotes that do resolve rather than freezing on an infinite plateau. Configurable
    /// so its sensitivity can be studied: too small and the optimizer treats a region
    /// with many failed quotes as competitive; large enough and it does not.
    double quote_penalty{1.0};

    /// A start's fit counts as "similar to the best" when its objective is within this
    /// factor of the best objective. The window in which two fits are considered to
    /// explain the surface equally well.
    double similar_fit_factor{1.10};

    /// Two similar fits are "materially different" when their parameter vectors, each
    /// coordinate normalised by its bound range, are farther apart than this. The
    /// threshold below which a calibration is reported as non-unique.
    double material_parameter_distance{0.10};
};

/// The residual at one quote, in both price and implied-volatility units, so the
/// residual surface can be read either way regardless of which the objective used.
struct QuoteResidual {
    OptionType type{OptionType::Call};
    double strike{};
    double maturity{};
    double weight{};

    double market_price{};
    double model_price{};
    double market_implied_volatility{};
    double model_implied_volatility{};
};

/// The outcome of a single starting point.
struct CalibrationStart {
    HestonParameters initial;
    HestonParameters calibrated;

    /// True when the start ran. A guess outside the bounds has no unconstrained image
    /// and is not run; it is recorded with started=false and a reason rather than
    /// dropped, so the record shows every guess that was tried.
    bool started{};
    std::string note;

    /// The weighted sum of squared residuals in the objective's units.
    double objective_value{};

    /// Root-mean-square residuals across the quotes, in each unit. Reported separately
    /// from the objective because a small objective in one unit does not certify a
    /// good fit in the other, and because they are the numbers a reader compares
    /// across surfaces.
    double price_rmse{};
    double implied_vol_rmse{};

    NelderMeadStatus status{NelderMeadStatus::MaxIterationsReached};
    int iterations{};
    int function_evaluations{};

    /// Quotes the calibration could not price or invert at the calibrated parameters.
    /// Zero for a clean fit. A non-zero value is a caveat on the fit, not a hidden
    /// hole: it means the objective carried a penalty for those quotes rather than a
    /// residual, and the reader should know the RMSEs are over the rest.
    int quotes_failed{};
};

/// A complete calibration: the best fit, every start, the residual surface, and the
/// evidence on whether the fit is unique.
struct CalibrationResult {
    /// The lowest-objective start. Its status says whether it actually converged --
    /// low objective and non-convergence are different findings and both are visible.
    CalibrationStart best;

    /// Every start, in the order given, including the ones that did not run.
    std::vector<CalibrationStart> starts;

    /// The residual at each quote, evaluated at the best parameters.
    std::vector<QuoteResidual> best_residuals;

    /// True when the best fit leaned on the penalty: at least one quote at the best
    /// parameters could not be priced or inverted, so part of its objective is penalty
    /// mass rather than fit. A calibration for which this is true must not be reported
    /// as a clean success -- the penalty made the objective look better than the fit
    /// actually is. This is the gate the penalty-sensitivity requirement turns on.
    bool relied_on_penalties{};

    /// True when two started fits explain the surface about equally well (objectives
    /// within similar_fit_factor) yet sit materially apart in parameter space. This is
    /// the non-uniqueness the exit gate requires be reported, not hidden.
    bool non_unique{};

    /// The largest normalised parameter distance among the similar-fit starts. Zero
    /// when only one start fit well.
    double max_similar_fit_distance{};

    /// The materially-different parameter vectors that fit about as well as the best.
    /// Preserved as evidence: these are the calibrations a low objective alone would
    /// have hidden.
    std::vector<HestonParameters> similar_fits;

    /// Dispersion of the calibrated parameters across the started guesses -- the
    /// sensitivity-to-initial-conditions the spec requires.
    HestonParameters parameter_mean;
    HestonParameters parameter_stddev;

    double objective_min{};
    double objective_max{};
    double objective_mean{};

    /// How many starts actually ran.
    std::uint64_t started_count{};
};

/// Calibrates Heston parameters to a volatility surface from multiple starting points.
///
/// Separates, and reports separately, the four things a calibration can get right or
/// wrong (MATHEMATICAL-SPEC section 16, EXP-11): optimizer convergence (per-start
/// status), surface-fit quality (the RMSEs and residual surface), parameter recovery
/// (against a known truth, by the caller), and identifiability (whether materially
/// different parameters fit equally well). A low objective is never on its own read as
/// success.
///
/// Fails only when no starting guess is feasible, or the surface is empty -- a
/// calibration with nothing to fit or nowhere to start from. A start that does not
/// converge is not a failure: it is recorded with its status so the reader sees it.
[[nodiscard]] Result<CalibrationResult> calibrate_heston(const VolatilitySurface& surface,
                                                         const CalibrationConfig& config);

}  // namespace diffusionworks

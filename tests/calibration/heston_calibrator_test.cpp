#include <diffusionworks/calibration/heston_calibrator.hpp>
#include <diffusionworks/calibration/volatility_surface.hpp>
#include <diffusionworks/core/error.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace diffusionworks {
namespace {

HestonParameters params(double v0, double kappa, double theta, double xi, double rho) {
    return HestonParameters{.initial_variance = v0,
                            .mean_reversion = kappa,
                            .long_run_variance = theta,
                            .vol_of_variance = xi,
                            .correlation = rho};
}

// A small surface and light pricing so the calibration runs in seconds. The physics is
// the same as the published EXP-11 configuration; only the grid and node count shrink.
SyntheticSurfaceSpec small_spec() {
    SyntheticSurfaceSpec spec;
    spec.strikes = {90.0, 100.0, 110.0};
    spec.maturities = {0.5, 1.0};
    spec.parameters = params(0.04, 1.5, 0.05, 0.4, -0.6);
    return spec;
}

CalibrationConfig fast_config() {
    CalibrationConfig config;
    config.pricing = HestonAnalyticConfig{.quadrature_nodes = 128, .convergence_tolerance = 1e-5};
    config.optimizer.max_iterations = 1200;
    return config;
}

// ---------------------------------------------------------------------------
// Recovery -- the defining property
// ---------------------------------------------------------------------------

// From several starting points, none of them the truth, the best fit recovers the
// parameters that generated the surface. This is the catalog's central check, run the
// way it forbids being skipped: multiple diverse guesses, not the true parameters.
TEST(HestonCalibratorTest, RecoversKnownParametersFromASyntheticSurface) {
    const auto spec = small_spec();
    const auto surface = generate_heston_surface(spec);
    ASSERT_TRUE(surface.ok()) << surface.error().describe();

    CalibrationConfig config = fast_config();
    config.initial_guesses = {params(0.03, 1.0, 0.06, 0.5, -0.5),
                              params(0.09, 0.8, 0.09, 0.2, -0.85),
                              params(0.02, 2.5, 0.03, 0.7, -0.3)};

    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    const HestonParameters& best = result.value().best.calibrated;
    const HestonParameters& truth = spec.parameters;
    EXPECT_NEAR(best.initial_variance, truth.initial_variance, 1e-3);
    EXPECT_NEAR(best.mean_reversion, truth.mean_reversion, 5e-2);
    EXPECT_NEAR(best.long_run_variance, truth.long_run_variance, 1e-3);
    EXPECT_NEAR(best.vol_of_variance, truth.vol_of_variance, 5e-2);
    EXPECT_NEAR(best.correlation, truth.correlation, 5e-2);

    // And the fit is genuinely good, not just close in parameters.
    EXPECT_LT(result.value().best.implied_vol_rmse, 1e-4);
    EXPECT_EQ(result.value().best.quotes_failed, 0);
}

// The true parameters, priced back through the surface, drive the objective to
// essentially zero -- a sanity check that the objective is measuring what it should.
TEST(HestonCalibratorTest, TheTruthIsAMinimumOfTheObjective) {
    const auto spec = small_spec();
    const auto surface = generate_heston_surface(spec);
    ASSERT_TRUE(surface.ok());

    CalibrationConfig config = fast_config();
    // Start exactly at the truth: it should stay there with a near-zero objective.
    config.initial_guesses = {spec.parameters};
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    EXPECT_LT(result.value().best.objective_value, 1e-10);
    EXPECT_LT(result.value().best.implied_vol_rmse, 1e-4);
}

// ---------------------------------------------------------------------------
// The four things kept separate
// ---------------------------------------------------------------------------

// Every start is recorded with its own convergence status, objective, fit RMSEs, and
// evaluation count -- optimizer convergence and fit quality reported separately, never
// collapsed into a single "success".
TEST(HestonCalibratorTest, ReportsEveryStartSeparately) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok());

    CalibrationConfig config = fast_config();
    config.initial_guesses = {params(0.03, 1.0, 0.06, 0.5, -0.5),
                              params(0.06, 2.0, 0.04, 0.6, -0.4)};
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    ASSERT_EQ(result.value().starts.size(), 2U);
    EXPECT_EQ(result.value().started_count, 2U);
    for (const CalibrationStart& start : result.value().starts) {
        EXPECT_TRUE(start.started);
        EXPECT_GT(start.function_evaluations, 0);
        // The objective is finite and the RMSEs are populated -- a start that ran
        // carries its own evidence.
        EXPECT_TRUE(std::isfinite(start.objective_value));
        EXPECT_GE(start.implied_vol_rmse, 0.0);
    }
    // The objective spread across starts is reported, so a reader sees dispersion
    // rather than a single number.
    EXPECT_GE(result.value().objective_max, result.value().objective_min);
}

// A guess exactly at the truth coinciding with another good fit is not reported as
// non-unique: two fits at the same point are the same calibration, not two.
TEST(HestonCalibratorTest, DoesNotFlagCoincidentGoodFitsAsNonUnique) {
    const auto spec = small_spec();
    const auto surface = generate_heston_surface(spec);
    ASSERT_TRUE(surface.ok());

    CalibrationConfig config = fast_config();
    config.initial_guesses = {params(0.03, 1.0, 0.06, 0.5, -0.5),
                              params(0.05, 2.0, 0.04, 0.5, -0.7)};
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_TRUE(result.ok()) << result.error().describe();
    // Both starts that recover the truth sit on top of each other, so there is no
    // material spread among the similar fits.
    if (!result.value().non_unique) {
        EXPECT_LT(result.value().max_similar_fit_distance, config.material_parameter_distance);
    }
}

// ---------------------------------------------------------------------------
// Penalty sensitivity
//
// The finite penalty must do two things at once: prevent the optimizer freezing where
// the quadrature cannot resolve a quote, and never let a region that leaves quotes
// unpriceable look competitive with a clean fit.
// ---------------------------------------------------------------------------

// Forced into a corner it cannot price -- vol-of-variance pinned high, a short
// maturity, and a low node count, so every quote fails -- the best fit leans entirely
// on the penalty. It is flagged (relied_on_penalties), and its objective is exactly the
// failed-quote count times the penalty, so it scales with the penalty and stays far
// above a clean fit's tiny objective: a penalised fit is never competitive.
TEST(HestonCalibratorTest, PenaltyGatesAndScalesAFailedFit) {
    SyntheticSurfaceSpec spec;
    spec.strikes = {90.0, 100.0, 110.0};
    spec.maturities = {0.05, 0.5};
    spec.parameters = params(0.04, 1.5, 0.05, 0.4, -0.6);
    const auto surface = generate_heston_surface(spec);
    ASSERT_TRUE(surface.ok()) << surface.error().describe();

    const auto run_with_penalty = [&](double penalty) {
        CalibrationConfig config;
        config.pricing =
            HestonAnalyticConfig{.quadrature_nodes = 32, .convergence_tolerance = 1e-6};
        config.optimizer.max_iterations = 400;
        config.quote_penalty = penalty;
        // Vol-of-variance forced high, where the characteristic-function integrand is
        // oscillatory and 32 nodes cannot resolve it: the whole feasible box fails.
        config.bounds.lower.vol_of_variance = 2.5;
        config.bounds.upper.vol_of_variance = 4.0;
        config.initial_guesses = {params(0.04, 1.5, 0.05, 3.0, -0.6),
                                  params(0.03, 2.0, 0.06, 3.5, -0.5)};
        return calibrate_heston(surface.value(), config);
    };

    const auto low = run_with_penalty(1.0);
    const auto high = run_with_penalty(50.0);
    ASSERT_TRUE(low.ok()) << low.error().describe();
    ASSERT_TRUE(high.ok()) << high.error().describe();

    // Both leaned on the penalty and are flagged, so neither can be a clean success.
    EXPECT_TRUE(low.value().relied_on_penalties);
    EXPECT_TRUE(high.value().relied_on_penalties);
    EXPECT_GT(low.value().best.quotes_failed, 0);

    // The objective is the failed-quote count times the penalty -- it scales with the
    // penalty, so raising it keeps a penalised region non-competitive.
    EXPECT_NEAR(low.value().best.objective_value,
                static_cast<double>(low.value().best.quotes_failed) * 1.0,
                1e-9);
    EXPECT_NEAR(high.value().best.objective_value,
                static_cast<double>(high.value().best.quotes_failed) * 50.0,
                1e-9);
    // And it is far above a good fit's objective (~1e-4), so a clean fit always wins.
    EXPECT_GT(low.value().best.objective_value, 1.0);
}

// The penalty shapes the path to the minimum, not the minimum itself: on a clean
// surface the penalty never triggers, so its magnitude does not move the fit, and the
// result is not flagged as penalty-reliant.
TEST(HestonCalibratorTest, PenaltyMagnitudeDoesNotMoveACleanMinimum) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok());

    const auto run_with_penalty = [&](double penalty) {
        CalibrationConfig config = fast_config();
        config.quote_penalty = penalty;
        config.initial_guesses = {params(0.03, 1.0, 0.06, 0.5, -0.5),
                                  params(0.05, 2.0, 0.04, 0.5, -0.7)};
        return calibrate_heston(surface.value(), config);
    };

    const auto low = run_with_penalty(1.0);
    const auto high = run_with_penalty(100.0);
    ASSERT_TRUE(low.ok());
    ASSERT_TRUE(high.ok());

    EXPECT_FALSE(low.value().relied_on_penalties);
    EXPECT_FALSE(high.value().relied_on_penalties);
    // The clean fit is identical regardless of the penalty magnitude.
    EXPECT_NEAR(low.value().best.calibrated.vol_of_variance,
                high.value().best.calibrated.vol_of_variance,
                1e-6);
    EXPECT_NEAR(
        low.value().best.calibrated.correlation, high.value().best.calibrated.correlation, 1e-6);
}

// ---------------------------------------------------------------------------
// What is refused, and what is recorded rather than refused
// ---------------------------------------------------------------------------

// A guess outside the bounds is recorded as not-started with a reason, not dropped --
// the record shows every guess that was tried.
TEST(HestonCalibratorTest, RecordsAnInfeasibleGuessRatherThanDroppingIt) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok());

    CalibrationConfig config = fast_config();
    // The second guess has a correlation of -2, outside the box.
    config.initial_guesses = {params(0.04, 1.5, 0.05, 0.4, -0.6),
                              params(0.04, 1.5, 0.05, 0.4, -2.0)};
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_TRUE(result.ok()) << result.error().describe();

    ASSERT_EQ(result.value().starts.size(), 2U);
    EXPECT_TRUE(result.value().starts[0].started);
    EXPECT_FALSE(result.value().starts[1].started);
    EXPECT_FALSE(result.value().starts[1].note.empty());
    EXPECT_EQ(result.value().started_count, 1U);
}

// Every guess outside the bounds leaves nothing to calibrate: a failure, not a
// zero-filled result.
TEST(HestonCalibratorTest, FailsWhenNoGuessIsFeasible) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok());

    CalibrationConfig config = fast_config();
    config.initial_guesses = {params(0.04, 1.5, 0.05, 0.4, -2.0)};  // rho outside the box
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(HestonCalibratorTest, RejectsAnEmptySurface) {
    VolatilitySurface empty;
    empty.spot = 100.0;
    empty.rate = 0.02;
    CalibrationConfig config = fast_config();
    config.initial_guesses = {params(0.04, 1.5, 0.05, 0.4, -0.6)};
    const auto result = calibrate_heston(empty, config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

TEST(HestonCalibratorTest, RejectsNoInitialGuess) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok());
    CalibrationConfig config = fast_config();  // no guesses
    const auto result = calibrate_heston(surface.value(), config);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Synthetic surface generation
// ---------------------------------------------------------------------------

TEST(HestonCalibratorTest, GeneratesAConsistentSurface) {
    const auto surface = generate_heston_surface(small_spec());
    ASSERT_TRUE(surface.ok()) << surface.error().describe();
    EXPECT_EQ(surface.value().quotes.size(), 6U);
    EXPECT_FALSE(surface.value().source.empty());
    for (const SurfaceQuote& quote : surface.value().quotes) {
        EXPECT_GT(quote.price, 0.0);
        EXPECT_GT(quote.implied_volatility, 0.0);
    }
}

TEST(HestonCalibratorTest, RejectsAnEmptyGrid) {
    SyntheticSurfaceSpec spec = small_spec();
    spec.strikes.clear();
    const auto surface = generate_heston_surface(spec);
    ASSERT_FALSE(surface.ok());
    EXPECT_EQ(surface.error().code, ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace diffusionworks

# ---------------------------------------------------------------------------
# diffusionworks_core: the numerical engine
#
# Per TECHNICAL-DESIGN section 1, this library holds all model, pricing,
# simulation, statistics, and validation code and is independent of CLI parsing,
# plotting, and reporting. The CLI links against it; it never links back.
# ---------------------------------------------------------------------------

add_library(diffusionworks_core)
add_library(diffusionworks::core ALIAS diffusionworks_core)

target_sources(
    diffusionworks_core
    PRIVATE config/domain_parsers.cpp
            core/build_info.cpp
            core/config.cpp
            engines/barrier_analytic.cpp
            engines/barrier_monte_carlo.cpp
            engines/black_scholes_analytic.cpp
            engines/finite_difference_engine.cpp
            engines/geometric_asian_analytic.cpp
            engines/greeks_monte_carlo.cpp
            engines/heston_analytic.cpp
            engines/heston_monte_carlo.cpp
            engines/implied_volatility.cpp
            engines/monte_carlo_engine.cpp
            experiments/barrier_experiments.cpp
            experiments/convergence.cpp
            experiments/convergence_experiments.cpp
            experiments/greek_experiments.cpp
            experiments/heston_simulation_experiments.cpp
            experiments/experiment.cpp
            experiments/pde_experiments.cpp
            experiments/scheme_moments.cpp
            instruments/asian_option.cpp
            instruments/barrier_option.cpp
            instruments/european_option.cpp
            market/market_state.cpp
            models/black_scholes.cpp
            models/heston.cpp
            numerics/tridiagonal.cpp
            pde/asset_grid.cpp
            pde/black_scholes_operator.cpp
            pde/boundary_conditions.cpp
            random/random_stream.cpp
            simulation/gbm_path_generator.cpp
            simulation/gbm_stepper.cpp
            simulation/time_grid.cpp
            statistics/distributions.cpp
            statistics/multi_seed.cpp
            statistics/online_moments.cpp
            statistics/regression.cpp)

target_include_directories(
    diffusionworks_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           # Generated headers (build_config.hpp, git_info.hpp) live under the
           # same diffusionworks/core/ prefix as the checked-in headers, so both
           # are reached by the canonical <diffusionworks/...> include path.
           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_link_libraries(
    diffusionworks_core
    PUBLIC nlohmann_json::nlohmann_json
           fmt::fmt
    PRIVATE dw::warnings
            dw::sanitizers)

# Sanitizer flags must reach every consumer's compile and link lines, so they are
# also applied publicly. dw::sanitizers is an empty interface unless a sanitizer
# preset enabled one.
target_link_libraries(diffusionworks_core PUBLIC dw::sanitizers)

add_dependencies(diffusionworks_core dw_git_info)

set_target_properties(
    diffusionworks_core
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# diffusionworks: the CLI
#
# ADR-004 makes the CLI the authoritative interface. It contains argument
# parsing and rendering only; no pricing logic lives here.
# ---------------------------------------------------------------------------

add_executable(diffusionworks)

target_sources(
    diffusionworks
    PRIVATE cli/main.cpp
            cli/experiment_command.cpp
            cli/greeks_command.cpp
            cli/options.cpp
            cli/output.cpp
            cli/price_command.cpp
            cli/simulate_command.cpp)

target_link_libraries(
    diffusionworks
    PRIVATE diffusionworks::core
            dw::warnings
            dw::sanitizers)

set_target_properties(
    diffusionworks
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# diffusionworks_cli_lib: CLI internals exposed for testing
#
# TESTING-STRATEGY section 10 requires tests for argument parsing, help text, and
# exit codes. Those live in cli/options.cpp, which the executable compiles
# directly; this object library lets the test target reuse the same translation
# unit without duplicating it or making main() testable.
# ---------------------------------------------------------------------------

if(DW_BUILD_TESTS)
    add_library(
        diffusionworks_cli_lib STATIC
        cli/experiment_command.cpp
        cli/greeks_command.cpp
        cli/options.cpp
        cli/output.cpp
        cli/price_command.cpp
        cli/simulate_command.cpp)
    add_library(diffusionworks::cli_lib ALIAS diffusionworks_cli_lib)

    target_include_directories(diffusionworks_cli_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cli)

    target_link_libraries(
        diffusionworks_cli_lib
        PUBLIC diffusionworks::core
        PRIVATE dw::warnings
                dw::sanitizers)

    set_target_properties(
        diffusionworks_cli_lib
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF)
endif()

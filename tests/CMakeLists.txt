# ---------------------------------------------------------------------------
# Test suite
#
# Tests are grouped into labelled targets so that CI and developers can run
# subsets (TESTING-STRATEGY section 15):
#
#   unit        - fast, deterministic component tests
#   statistical - stochastic tests with statistical acceptance rules
#   concurrency - tests exercised under ThreadSanitizer
#   cli         - argument parsing, exit codes, and help text
#
# add_dw_test() registers one executable per test file group and applies the
# shared configuration in one place.
# ---------------------------------------------------------------------------

function(add_dw_test target_name)
    cmake_parse_arguments(ARG "" "LABEL" "SOURCES;LIBS" ${ARGN})

    if(NOT ARG_LABEL)
        set(ARG_LABEL "unit")
    endif()

    add_executable(${target_name} ${ARG_SOURCES})

    # GTest::gtest_main already carries GTest::gtest, and the targets in ARG_LIBS
    # already carry diffusionworks::core; naming either again only produces a
    # duplicate on the link line.
    target_link_libraries(
        ${target_name}
        PRIVATE GTest::gtest_main
                dw::warnings
                dw::sanitizers
                ${ARG_LIBS})

    if(NOT ARG_LIBS)
        target_link_libraries(${target_name} PRIVATE diffusionworks::core)
    endif()

    # Reference fixtures (data/references/) are read from the source tree rather
    # than copied into the build directory, so a regenerated reference takes
    # effect without a rebuild and there is only ever one copy to audit.
    target_compile_definitions(${target_name}
                               PRIVATE DW_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

    set_target_properties(
        ${target_name}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF)

    gtest_discover_tests(
        ${target_name}
        PROPERTIES LABELS "${ARG_LABEL}"
        DISCOVERY_MODE PRE_TEST)
endfunction()

# --- Core infrastructure ----------------------------------------------------

add_dw_test(
    dw_test_core
    LABEL unit
    SOURCES core/result_test.cpp
            core/error_test.cpp
            core/config_test.cpp
            core/build_info_test.cpp)

# --- Numerics ---------------------------------------------------------------

add_dw_test(
    dw_test_numerics
    LABEL unit
    SOURCES numerics/normal_test.cpp)

# --- Randomness -------------------------------------------------------------
#
# The statistical tests here draw millions of samples so that their acceptance
# bounds sit far above the sampling noise. A flaky stochastic test is a
# test-design failure (TESTING-STRATEGY section 6), so they are labelled
# `statistical` and kept out of the fast developer loop.

add_dw_test(
    dw_test_random
    LABEL unit
    SOURCES random/philox_test.cpp
            random/reproducibility_test.cpp)

add_dw_test(
    dw_test_random_stream
    LABEL statistical
    SOURCES random/random_stream_test.cpp)

# --- Simulation -------------------------------------------------------------

add_dw_test(
    dw_test_simulation
    LABEL statistical
    SOURCES simulation/gbm_test.cpp)

# --- Statistics -------------------------------------------------------------

add_dw_test(
    dw_test_statistics
    LABEL unit
    SOURCES statistics/distributions_test.cpp
            statistics/online_moments_test.cpp
            statistics/multi_seed_test.cpp)

# --- Domain types -----------------------------------------------------------

add_dw_test(
    dw_test_domain
    LABEL unit
    SOURCES market/market_state_test.cpp
            instruments/european_option_test.cpp
            models/black_scholes_test.cpp)

# --- Pricing engines --------------------------------------------------------

add_dw_test(
    dw_test_engines
    LABEL unit
    SOURCES engines/black_scholes_reference_test.cpp
            engines/black_scholes_high_precision_test.cpp
            engines/black_scholes_invariants_test.cpp
            engines/black_scholes_greeks_test.cpp)

# The Monte Carlo suite runs hundreds of independent replications to measure
# coverage and dispersion, so it is labelled statistical and kept out of the fast
# developer loop.

add_dw_test(
    dw_test_monte_carlo
    LABEL statistical
    SOURCES engines/monte_carlo_engine_test.cpp)

# --- CLI --------------------------------------------------------------------

add_dw_test(
    dw_test_cli
    LABEL cli
    SOURCES cli/options_test.cpp
            cli/price_command_test.cpp
            cli/simulate_command_test.cpp
    LIBS diffusionworks::cli_lib)

# --- External cross-validation ----------------------------------------------
#
# QuantLib is a validation-only reference (ADR-003). It links into this target
# and nowhere else: not the core library, not the CLI, not any published result.
# The target exists only when DW_ENABLE_QUANTLIB=ON, so a clean checkout builds
# and tests fully without QuantLib installed.

if(DW_ENABLE_QUANTLIB)
    # core is named explicitly: add_dw_test only adds it by default when LIBS is
    # empty, and QuantLib::QuantLib does not carry it.
    add_dw_test(
        dw_test_quantlib
        LABEL validation
        SOURCES validation/quantlib_cross_validation_test.cpp
        LIBS diffusionworks::core QuantLib::QuantLib)

    message(STATUS "QuantLib cross-validation target enabled: dw_test_quantlib")
endif()

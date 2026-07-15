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

# --- CLI --------------------------------------------------------------------

add_dw_test(
    dw_test_cli
    LABEL cli
    SOURCES cli/options_test.cpp
    LIBS diffusionworks::cli_lib)

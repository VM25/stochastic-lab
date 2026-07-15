# Dependency acquisition.
#
# Pinned to exact tags so a clean checkout resolves identical dependency
# versions. Per ARCHITECTURE-DECISIONS ADR-001/003, none of these implement the
# project's models, pricing, simulation, Greeks, or calibration; they provide
# formatting, JSON, testing, and benchmarking support only.

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# --- fmt: console and report formatting -------------------------------------
#
# 11.2.0 or newer is required: fmt 11.0.x fails to compile under Clang 19+,
# which rejects its consteval format-string constructors as non-constant
# expressions (fmtlib/fmt#4177). Apple Clang 21 hits this.

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS NAMES fmt)

# --- nlohmann/json: configuration parsing and result serialization ----------

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS NAMES nlohmann_json)

FetchContent_MakeAvailable(fmt nlohmann_json)

# --- GoogleTest -------------------------------------------------------------

if(DW_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
        SYSTEM
        FIND_PACKAGE_ARGS NAMES GTest)

    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif()

# --- Google Benchmark -------------------------------------------------------

if(DW_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.0
        GIT_SHALLOW TRUE
        SYSTEM
        FIND_PACKAGE_ARGS NAMES benchmark)

    FetchContent_MakeAvailable(benchmark)
endif()

# --- QuantLib (optional, external validation only) --------------------------
#
# Per ADR-003 QuantLib is a validation-only reference. It is never linked into
# the core library or the CLI; only dedicated cross-validation targets use it.

if(DW_ENABLE_QUANTLIB)
    find_package(QuantLib QUIET)
    if(NOT QuantLib_FOUND)
        message(WARNING
                "DW_ENABLE_QUANTLIB=ON but QuantLib was not found. "
                "External cross-validation targets will be skipped.")
        set(DW_ENABLE_QUANTLIB OFF CACHE BOOL "" FORCE)
    else()
        message(STATUS "QuantLib found: external validation targets enabled")
    endif()
endif()

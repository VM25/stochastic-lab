cmake_minimum_required(VERSION 3.24)

project(
    diffusionworks
    VERSION 0.1.0
    DESCRIPTION "DiffusionWorks: Stochastic Derivatives Numerics"
    HOMEPAGE_URL "https://github.com/VM25/stochastic-lab"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard and global policy
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Keep floating-point semantics IEEE-conformant. The project measures numerical
# error to several significant figures and compares against analytic references;
# fast-math style reassociation would invalidate those measurements.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

option(DW_BUILD_TESTS "Build the DiffusionWorks test suite" ON)
option(DW_BUILD_BENCHMARKS "Build the DiffusionWorks benchmark suite" OFF)
option(DW_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DW_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(DW_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(DW_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(DW_ENABLE_QUANTLIB "Enable optional QuantLib external-validation targets" OFF)

include(CompilerWarnings)
include(Sanitizers)
include(BuildInfo)
include(Dependencies)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------

add_subdirectory(src)

if(DW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DW_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "DiffusionWorks ${PROJECT_VERSION}")
message(STATUS "  Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler            : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests               : ${DW_BUILD_TESTS}")
message(STATUS "  Benchmarks          : ${DW_BUILD_BENCHMARKS}")
message(STATUS "  Warnings as errors  : ${DW_WARNINGS_AS_ERRORS}")
message(STATUS "  ASan / UBSan / TSan : ${DW_ENABLE_ASAN} / ${DW_ENABLE_UBSAN} / ${DW_ENABLE_TSAN}")
message(STATUS "  QuantLib validation : ${DW_ENABLE_QUANTLIB}")
message(STATUS "")

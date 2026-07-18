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

# ---------------------------------------------------------------------------
# Floating-point semantics
#
# Disable contraction of a*b+c into a fused multiply-add. This is a correctness
# and reproducibility setting, not a performance preference, and it is applied
# globally rather than per target so that no target can quietly opt out.
#
# Why it matters here more than in most projects:
#
#   * Reproducibility across compilers. GCC defaults to -ffp-contract=fast and
#     Clang to =on, and both decide *per expression* whether to contract. The
#     same source then yields different low-order bits on different toolchains --
#     while CI builds and compares results on both. A published result that
#     depends on which compiler produced it is not reproducible (VALIDATION-PLAN
#     section 19).
#
#   * It changes convergence, not just the last bit. The incomplete beta
#     continued fraction in src/statistics/distributions.cpp is a fixed-point
#     iteration. Under contraction its residual stalls near 1e-14 and never
#     reaches the 3e-16 convergence criterion, so Student-t at large degrees of
#     freedom fails outright -- which is exactly the ordinary case of a Monte
#     Carlo run with 100,000 paths. Contraction perturbs each step's rounding,
#     and an iteration seeking the last representable digit cannot settle.
#
# The cost is that FMA is unavailable in numeric kernels. That is measured in
# Phase 13 rather than assumed, and disclosed with every benchmark: build flags
# are part of the published metadata (BENCHMARK-PLAN section 2).
#
# This deliberately does not disable FMA formation the compiler cannot observe
# (a library exp may still use it internally); the guarantee is over this
# project's own expressions.
#
# Appended to CMAKE_CXX_FLAGS rather than set with add_compile_options so that it
# reaches the published build metadata. A flag that changes the numbers must
# appear alongside them, or a reader cannot tell which semantics produced the
# result they are looking at.
if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /fp:precise")
else()
    string(APPEND CMAKE_CXX_FLAGS " -ffp-contract=off")
endif()

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

# Multithreaded Monte Carlo (Phase 12) uses std::jthread; on libstdc++/libc++ the
# standard threading facilities need the system threads library on the link line.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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

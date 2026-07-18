# Benchmark suite.
#
# BENCHMARK-PLAN section 1 requires correctness before measurement, so this suite
# lands in Phase 13 -- once Monte Carlo, PDE, Greek, and calibration paths are
# validated and there is something meaningful and correct to time. The engines timed
# here are exactly those Phase 12 validated for correctness under threading.
#
# Only built when DW_BUILD_BENCHMARKS=ON (the `benchmark` preset), which forces a
# Release build: BENCHMARK-PLAN section 1 forbids benchmarking a debug build.

# add_dw_benchmark() registers one Google Benchmark executable and applies the shared
# configuration in one place, mirroring add_dw_test() in tests/.
function(add_dw_benchmark target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    add_executable(${target_name} ${ARG_SOURCES})

    # A custom main() calls benchmark::Initialize and registers the build metadata, so
    # only the library is linked, not benchmark_main.
    target_link_libraries(
        ${target_name}
        PRIVATE diffusionworks::core
                benchmark::benchmark
                dw::warnings)

    set_target_properties(
        ${target_name}
        PROPERTIES CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF)
endfunction()

# Monte Carlo throughput and parallel scaling (BENCHMARK-PLAN sections 4 and 5): the
# single-thread and multithread baselines every optimization is measured against.
add_dw_benchmark(
    dw_bench_monte_carlo_scaling
    SOURCES monte_carlo_scaling.cpp)

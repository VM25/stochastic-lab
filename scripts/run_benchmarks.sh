#!/usr/bin/env bash
#
# Capture a Monte Carlo scaling baseline as a reproducible, machine-readable artifact.
#
# BENCHMARK-PLAN sections 2, 3, and 13: release build, repeated timed runs with
# dispersion, and raw JSON results carrying the full environment metadata. This is the
# single command a baseline (or a before/after optimization comparison) is captured
# with, so the result is reproducible from its own record.
#
# Usage:
#   scripts/run_benchmarks.sh [label]
#
#   label   optional suffix for the output file (e.g. "baseline", "after-opt").
#           Defaults to "baseline".
#
# Output:
#   results/benchmarks/monte_carlo_scaling.<label>.json
#
# IMPORTANT: run on an otherwise-idle machine. BENCHMARK-PLAN section 3 forbids
# measuring under unrelated workloads; a baseline captured under load is not a baseline.
# This script refuses to run when the 1-minute load average exceeds the core count,
# because contention would corrupt the wall-clock scaling numbers. Override only if you
# understand the measurement is contaminated: DW_ALLOW_LOADED=1.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABEL=${1:-baseline}
OUT="results/benchmarks/monte_carlo_scaling.${LABEL}.json"

# --- Refuse to measure under load ------------------------------------------

CORES=$(getconf _NPROCESSORS_ONLN 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 1)
# 1-minute load average, portable across macOS and Linux `uptime` output.
LOAD1=$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')
LOAD1_INT=${LOAD1%.*}

if [[ ${DW_ALLOW_LOADED:-0} -ne 1 && ${LOAD1_INT:-0} -gt ${CORES} ]]; then
    echo "error: 1-minute load average is ${LOAD1} on ${CORES} cores." >&2
    echo "       The machine is busy; a benchmark now would measure contention, not the" >&2
    echo "       engine (BENCHMARK-PLAN section 3). Wait for it to settle, or set" >&2
    echo "       DW_ALLOW_LOADED=1 to measure anyway (the result is contaminated)." >&2
    exit 1
fi

# --- Build (release, benchmark preset) -------------------------------------

echo "==> configuring and building the benchmark preset (release)"
cmake --preset benchmark > /dev/null
cmake --build build/benchmark --target dw_bench_monte_carlo_scaling > /dev/null

# --- Measure ---------------------------------------------------------------

mkdir -p results/benchmarks

# Repetitions give the dispersion BENCHMARK-PLAN section 3 requires; the JSON carries
# every repetition plus mean/median/stddev. min_time keeps each case long enough to
# amortise timer overhead. The metadata is registered by the benchmark's own main().
echo "==> running (10 repetitions, JSON -> ${OUT})"
./build/benchmark/benchmarks/dw_bench_monte_carlo_scaling \
    --benchmark_repetitions=10 \
    --benchmark_min_time=0.5s \
    --benchmark_display_aggregates_only=true \
    --benchmark_out="${OUT}" \
    --benchmark_out_format=json

echo ""
echo "==> wrote ${OUT}"
echo "    load average at start was ${LOAD1} (recorded in the JSON context)."
echo "    Generate the scaling chart with: python3 python/plot_benchmarks.py ${OUT}"

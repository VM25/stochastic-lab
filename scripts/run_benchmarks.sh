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
#   label   optional suffix for the output file (e.g. "baseline", "session2",
#           "after-opt"). Defaults to "baseline".
#
# Output:
#   results/benchmarks/monte_carlo_scaling.<label>.json
#
# The JSON carries, in its "context": Google Benchmark's own host fields (date,
# num_cpus, mhz_per_cpu, cpu_scaling_enabled, caches, load_avg at start) and the
# benchmark's build metadata (compiler, flags, C++ standard, commit, seed); and, injected
# by this script, the CPU topology (performance/efficiency core split on Apple Silicon),
# the power and thermal state, and the load average *after* the run -- so contention that
# arose during measurement is visible, not hidden.
#
# IMPORTANT: run on an otherwise-idle machine on AC power. BENCHMARK-PLAN section 3
# forbids measuring under unrelated workloads; a baseline captured under load or on a
# throttling battery clock is not a baseline. This script refuses when the 1-minute load
# average exceeds the core count, or when running on battery. Override only if you
# understand the measurement is contaminated: DW_ALLOW_LOADED=1.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABEL=${1:-baseline}
OUT="results/benchmarks/monte_carlo_scaling.${LABEL}.json"

# --- Environment capture (macOS-aware; degrades gracefully elsewhere) -------

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CORES=$(getconf _NPROCESSORS_ONLN 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 1)

# 1-minute load average, portable across macOS and Linux `uptime` output.
read_load1() { uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/'; }
LOAD_PRE=$(read_load1)

# Cool-down before measuring, so every session starts from a comparable thermal and load
# state. Back-to-back all-core runs heat the CPU and its sustained multi-core frequency
# drops without a pmset thermal warning, which shifts the whole session's medians -- the
# instability the session-to-session stability gate exists to catch.
#
# Two parts, decoupled so brief background blips (a Shortcuts run, an iCloud sync tick)
# do not stall it forever:
#   1. A fixed idle soak of DW_COOLDOWN_S seconds (default 240). The benchmark is the
#      machine's real heat source and it is not running during the soak, so the die cools
#      even with the occasional short blip; requiring a *perfect* quiet streak here was too
#      strict and simply never completed on this machine. Set DW_COOLDOWN_S=0 to opt out.
#   2. A capture-start gate: after the soak, do not begin timing until the 1-minute load
#      is below DW_COOLDOWN_LOAD (default 2.0, matching the controlled-capture soak gate),
#      so no unrelated work is running as the measurement starts. Give up after 10 minutes
#      -- the machine will not settle.
cool_down() {
    local soak=${DW_COOLDOWN_S:-240}
    [[ ${soak} -le 0 ]] && return 0
    local ceiling=${DW_COOLDOWN_LOAD:-2.0}
    echo "==> cool-down: ${soak}s idle thermal soak, then wait for load < ${ceiling}"
    sleep "${soak}"
    local waited=0 l1
    while :; do
        l1=$(read_load1)
        if awk -v l="${l1}" -v cap="${ceiling}" 'BEGIN { exit !(l < cap) }'; then break; fi
        sleep 15
        waited=$((waited + 15))
        if ((waited > 600)); then
            echo "error: load stayed above ${ceiling} for 10 min after the soak (${l1})." >&2
            echo "       The machine will not settle; capture on a dedicated machine instead." >&2
            exit 1
        fi
    done
    echo "    soak complete, load ${l1} < ${ceiling}; measuring."
    LOAD_PRE=$(read_load1)
}

# CPU topology. On Apple Silicon the performance and efficiency cores are not
# homogeneous, so 1/2/4/8-thread scaling must be read against this split, not as
# ordinary linear multicore scaling.
P_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2> /dev/null || echo "")
E_CORES=$(sysctl -n hw.perflevel1.physicalcpu 2> /dev/null || echo "")
TOPOLOGY="logical=${CORES}"
if [[ -n ${P_CORES} && -n ${E_CORES} ]]; then
    TOPOLOGY="${TOPOLOGY}, performance=${P_CORES}, efficiency=${E_CORES}"
fi

# Power state and thermal pressure (macOS pmset). A battery clock throttles, and a
# thermal speed limit below 100 means the CPU is being held back -- both invalidate a
# headline number.
POWER_STATE=$(pmset -g ps 2> /dev/null | grep -oE "'(AC|Battery) Power'" | tr -d "'" | head -1 || true)
POWER_STATE=${POWER_STATE:-unknown}

# pmset reports either an explicit CPU_Speed_Limit (a throttle below 100 means the CPU
# is being held back) or, when nothing is throttling, "No thermal warning level has been
# recorded". Distinguish the two: an unrecorded warning is a nominal state, not unknown.
THERM_OUT=$(pmset -g therm 2> /dev/null || true)
if grep -q "CPU_Speed_Limit" <<< "${THERM_OUT}"; then
    THERMAL=$(sed -nE 's/.*CPU_Speed_Limit *= *([0-9]+).*/CPU_Speed_Limit=\1/p' <<< "${THERM_OUT}" | head -1)
elif grep -q "No thermal warning level" <<< "${THERM_OUT}"; then
    THERMAL="nominal (no thermal warning)"
else
    THERMAL="unknown"
fi

# --- Refuse to measure under load or on battery -----------------------------

LOAD_PRE_INT=${LOAD_PRE%.*}
if [[ ${DW_ALLOW_LOADED:-0} -ne 1 && ${LOAD_PRE_INT:-0} -gt ${CORES} ]]; then
    echo "error: 1-minute load average is ${LOAD_PRE} on ${CORES} cores." >&2
    echo "       The machine is busy; a benchmark now would measure contention, not the" >&2
    echo "       engine (BENCHMARK-PLAN section 3). Wait for it to settle, or set" >&2
    echo "       DW_ALLOW_LOADED=1 to measure anyway (the result is contaminated)." >&2
    exit 1
fi
if [[ ${DW_ALLOW_LOADED:-0} -ne 1 && ${POWER_STATE} == "Battery Power" ]]; then
    echo "error: running on battery power; the clock throttles, so wall-clock timing is" >&2
    echo "       not stable. Connect power, or set DW_ALLOW_LOADED=1 to measure anyway." >&2
    exit 1
fi

# --- Build (release, benchmark preset) -------------------------------------

echo "==> configuring and building the benchmark preset (release)"
cmake --preset benchmark > /dev/null
cmake --build build/benchmark --target dw_bench_monte_carlo_scaling > /dev/null

# --- Cool-down (thermal + load reset), then measure ------------------------

cool_down

mkdir -p results/benchmarks

# Repetitions give the dispersion BENCHMARK-PLAN section 3 requires; the JSON carries
# every repetition's mean/median/stddev/cv aggregate. min_time keeps each case long
# enough to amortise timer overhead. The build metadata is registered by the
# benchmark's own main().
echo "==> running (10 repetitions, JSON -> ${OUT})"
echo "    pre-run load ${LOAD_PRE}, power ${POWER_STATE}, thermal ${THERMAL}, topology ${TOPOLOGY}"
./build/benchmark/benchmarks/dw_bench_monte_carlo_scaling \
    --benchmark_repetitions=10 \
    --benchmark_min_time=0.5s \
    --benchmark_display_aggregates_only=true \
    --benchmark_out="${OUT}" \
    --benchmark_out_format=json

LOAD_POST=$(read_load1)

# --- Inject the OS-level environment metadata into the result's context -----

python3 - "${OUT}" "${TIMESTAMP}" "${LOAD_PRE}" "${LOAD_POST}" "${TOPOLOGY}" \
    "${POWER_STATE}" "${THERMAL}" "${CORES}" << 'PY'
import json
import sys

path, ts, load_pre, load_post, topology, power, thermal, cores = sys.argv[1:9]
record = json.loads(open(path, encoding="utf-8").read())
context = record.setdefault("context", {})
context["dw_capture_timestamp_utc"] = ts
context["dw_load_avg_pre"] = load_pre
context["dw_load_avg_post"] = load_post
context["dw_cpu_topology"] = topology
context["dw_power_state"] = power
context["dw_thermal_state"] = thermal
context["dw_core_count"] = cores
with open(path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(record, indent=2))
    handle.write("\n")
PY

# --- Post-run contention check ---------------------------------------------

LOAD_POST_INT=${LOAD_POST%.*}
echo ""
echo "==> wrote ${OUT}"
echo "    load: pre ${LOAD_PRE}, post ${LOAD_POST}; power ${POWER_STATE}; thermal ${THERMAL}"
if [[ ${LOAD_POST_INT:-0} -gt ${CORES} ]]; then
    echo "    WARNING: post-run load ${LOAD_POST} exceeds the core count -- unrelated work may" >&2
    echo "             have started during the run. Treat this capture as contaminated." >&2
fi
echo "    Chart + summary: python3 python/plot_benchmarks.py ${OUT}"
echo "    Capture at least one more session (a different label) and compare medians and"
echo "    dispersion before accepting this as a baseline."

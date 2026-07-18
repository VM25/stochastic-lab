// Monte Carlo throughput and parallel-scaling baseline.
//
// BENCHMARK-PLAN sections 4 (throughput) and 5 (parallel scaling): fixed workloads
// timed across thread counts on wall-clock time, reporting paths/second, so speedup
// T_1/T_p and efficiency = speedup/p follow directly. This is the *validated baseline*
// the optimization process (section 12) is measured against -- it is run before any
// optimization, and the engines it times are exactly the ones Phase 12 validated for
// correctness under threading. No optimization is made from these numbers without
// first profiling and preserving before-and-after evidence.
//
// Real time, not CPU time: a multithreaded run accumulates CPU time across workers, so
// only wall-clock time measures the speedup a thread count actually buys. Every case
// uses UseRealTime() for that reason.

#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/engines/barrier_monte_carlo.hpp>
#include <diffusionworks/engines/greeks_monte_carlo.hpp>
#include <diffusionworks/engines/heston_monte_carlo.hpp>
#include <diffusionworks/engines/monte_carlo_engine.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace diffusionworks {
namespace {

constexpr std::uint64_t kSeed = 20260718;

// The thread counts every scaling benchmark sweeps. 1 is the single-thread baseline;
// the rest give speedup and efficiency against it.
void thread_scaling(benchmark::internal::Benchmark* bench) {
    bench->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime()->Unit(benchmark::kMillisecond);
}

// Records paths/second and echoes the thread count, so speedup and efficiency follow
// from the wall-clock time reported at each thread count. kIsIterationInvariantRate:
// the path count is the same every iteration, and the counter is divided by the
// per-iteration wall time to yield a rate.
void report(benchmark::State& state, std::int64_t paths, int threads) {
    state.SetItemsProcessed(state.iterations() * paths);
    state.counters["paths_per_second"] = benchmark::Counter(
        static_cast<double>(paths), benchmark::Counter::kIsIterationInvariantRate);
    state.counters["threads"] = threads;
}

// --- European GBM ----------------------------------------------------------

void european(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig config;
    config.paths = 1'000'000;
    config.steps = 1;
    config.seed = kSeed;
    config.threads = threads;

    for (auto _ : state) {
        auto priced = MonteCarloEngine::price(market, option, model, config);
        if (!priced.ok()) {
            const std::string message = priced.error().describe();
            state.SkipWithError(message.c_str());
            return;
        }
        benchmark::DoNotOptimize(priced.value().value);
    }
    report(state, config.paths, threads);
}

BENCHMARK(european)->Apply(thread_scaling);

// --- Arithmetic Asian, with the control-variate pilot ----------------------

void asian_control_variate(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option =
        AsianOption::create(OptionType::Call, AveragingType::Arithmetic, 100.0, 1.0, 12).value();
    const auto model = BlackScholesModel::create(0.2).value();

    MonteCarloConfig config;
    config.paths = 200'000;
    config.steps = 12;
    config.seed = kSeed;
    config.threads = threads;
    config.variance_reduction.control_variate = true;

    for (auto _ : state) {
        auto priced = MonteCarloEngine::price(market, option, model, config);
        if (!priced.ok()) {
            const std::string message = priced.error().describe();
            state.SkipWithError(message.c_str());
            return;
        }
        benchmark::DoNotOptimize(priced.value().value);
    }
    report(state, config.paths, threads);
}

BENCHMARK(asian_control_variate)->Apply(thread_scaling);

// --- Heston full-truncation Euler ------------------------------------------

void heston(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = HestonModel::create(0.04, 3.0, 0.04, 0.3, -0.6).value();

    HestonMonteCarloConfig config;
    config.paths = 100'000;
    config.steps = 50;
    config.seed = kSeed;
    config.threads = threads;

    for (auto _ : state) {
        auto priced = HestonMonteCarloEngine::price(market, option, model, config);
        if (!priced.ok()) {
            const std::string message = priced.error().describe();
            state.SkipWithError(message.c_str());
            return;
        }
        benchmark::DoNotOptimize(priced.value().value);
    }
    report(state, config.paths, threads);
}

BENCHMARK(heston)->Apply(thread_scaling);

// --- Greeks: pathwise delta ------------------------------------------------

void greeks_pathwise_delta(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = EuropeanOption::create(OptionType::Call, 100.0, 1.0).value();
    const auto model = BlackScholesModel::create(0.2).value();

    GreeksMonteCarloConfig config;
    config.paths = 1'000'000;
    config.seed = kSeed;
    config.threads = threads;

    for (auto _ : state) {
        auto estimate = GreeksMonteCarloEngine::estimate(
            market, option, model, GreekName::Delta, GreekMethod::Pathwise, config);
        if (!estimate.ok()) {
            const std::string message = estimate.error().describe();
            state.SkipWithError(message.c_str());
            return;
        }
        benchmark::DoNotOptimize(estimate.value().value);
    }
    report(state, config.paths, threads);
}

BENCHMARK(greeks_pathwise_delta)->Apply(thread_scaling);

// --- Barrier, Brownian-bridge monitoring -----------------------------------

void barrier_bridge(benchmark::State& state) {
    const int threads = static_cast<int>(state.range(0));
    const auto market = MarketState::create(100.0, 0.05, 0.0).value();
    const auto option = BarrierOption::create(OptionType::Call,
                                              BarrierType::DownAndOut,
                                              100.0,
                                              90.0,
                                              1.0,
                                              MonitoringConvention::BrownianBridge,
                                              25)
                            .value();
    const auto model = BlackScholesModel::create(0.2).value();

    BarrierMonteCarloConfig config;
    config.paths = 200'000;
    config.seed = kSeed;
    config.threads = threads;

    for (auto _ : state) {
        auto priced = BarrierMonteCarloEngine::price(market, option, model, config);
        if (!priced.ok()) {
            const std::string message = priced.error().describe();
            state.SkipWithError(message.c_str());
            return;
        }
        benchmark::DoNotOptimize(priced.value().value);
    }
    report(state, config.paths, threads);
}

BENCHMARK(barrier_bridge)->Apply(thread_scaling);

// Records the build and host metadata BENCHMARK-PLAN section 2 requires, as benchmark
// context, so every JSON result carries the compiler, flags, C++ standard, commit, CPU,
// and OS it was produced on. Google Benchmark itself records the core count and CPU
// frequency, and the seed and workloads are fixed in the cases above.
void register_metadata() {
    const BuildInfo info = collect_build_info();
    benchmark::AddCustomContext("dw_version", info.version);
    benchmark::AddCustomContext("compiler", info.compiler_id + " " + info.compiler_version);
    benchmark::AddCustomContext("build_type", info.build_type);
    benchmark::AddCustomContext("build_flags", info.build_flags);
    benchmark::AddCustomContext("cxx_standard", info.cxx_standard);
    benchmark::AddCustomContext("git_commit", info.git_commit);
    benchmark::AddCustomContext("git_branch", info.git_branch);
    benchmark::AddCustomContext("os", info.os_name + " " + info.os_version);
    benchmark::AddCustomContext("cpu_brand", info.cpu_brand);
    benchmark::AddCustomContext("seed", std::to_string(kSeed));
}

}  // namespace
}  // namespace diffusionworks

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    diffusionworks::register_metadata();
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

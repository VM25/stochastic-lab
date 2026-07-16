#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace diffusionworks {

/// Provenance of a published run.
///
/// PROJECT-SPEC "Required Output Metadata" and BENCHMARK-PLAN section 2 require
/// every published run to identify the code, the toolchain, and the machine that
/// produced it. Fields split by when they are knowable:
///
///   - compile time: version, compiler, build type, flags, standard, git commit
///   - run time:     OS, CPU, cores, host, timestamp
///
/// CPU and OS are resolved at run time because the machine that executes a
/// benchmark is not necessarily the machine that compiled the binary.
struct BuildInfo {
    // --- Compile-time provenance ---
    std::string version;
    std::string compiler_id;
    std::string compiler_version;
    std::string build_type;
    std::string build_flags;
    std::string cxx_standard;
    std::string git_commit;
    std::string git_commit_short;
    std::string git_branch;

    /// True unless the working tree was verified clean at build time.
    ///
    /// A dirty tree means the commit hash does not fully identify the code, so
    /// results carrying git_dirty=true are not reproducible from the commit
    /// alone and must not back a published claim.
    ///
    /// Note the asymmetry: this is false only when `git status` ran and reported
    /// no modifications. If git was unavailable or the query failed, the field
    /// is true, because an unverified tree must not be published as a clean one.
    bool git_dirty{true};

    // --- Run-time environment ---
    std::string os_name;
    std::string os_version;
    std::string cpu_brand;
    unsigned int logical_cores{0};
    std::string hostname;

    /// UTC, ISO-8601, second resolution.
    std::string timestamp_utc;
};

/// Collects build and environment metadata for the current process.
///
/// The compile-time half is fixed at build time; the run-time half is sampled on
/// each call, so `timestamp_utc` differs between calls.
[[nodiscard]] BuildInfo collect_build_info();

/// Serialises provenance.
///
/// Lives beside BuildInfo rather than in the CLI because provenance is part of
/// every published artifact, not a rendering concern. EXPERIMENT-CATALOG requires
/// each experiment record to carry the compiler, hardware, and commit that
/// produced it, and experiments/ must not reach into cli/ to say so.
[[nodiscard]] nlohmann::json to_json(const BuildInfo& info);

}  // namespace diffusionworks

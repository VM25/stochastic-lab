#include <diffusionworks/core/build_config.hpp>
#include <diffusionworks/core/build_info.hpp>
#include <diffusionworks/core/git_info.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fstream>
#include <sys/utsname.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace diffusionworks {
namespace {

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t as_time_t = std::chrono::system_clock::to_time_t(now);

    // A failed conversion leaves `utc` unpopulated. Formatting it anyway would
    // stamp the artifact with a timestamp built from an unset tm, so the failure
    // is surfaced instead.
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &as_time_t) != 0) {
        return "unknown";
    }
#else
    if (::gmtime_r(&as_time_t, &utc) == nullptr) {
        return "unknown";
    }
#endif

    std::array<char, 32> buffer{};
    const std::size_t written =
        std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    if (written == 0) {
        return "unknown";
    }
    return {buffer.data(), written};
}

std::string read_hostname() {
#if defined(_WIN32)
    std::array<char, 256> buffer{};
    DWORD size = static_cast<DWORD>(buffer.size());
    if (GetComputerNameA(buffer.data(), &size) != 0) {
        return std::string(buffer.data(), size);
    }
    return "unknown";
#else
    std::array<char, 256> buffer{};
    if (::gethostname(buffer.data(), buffer.size() - 1) == 0) {
        return {buffer.data()};
    }
    return "unknown";
#endif
}

#if defined(__APPLE__)

std::string sysctl_string(const char* name) {
    std::size_t size = 0;
    if (::sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return "unknown";
    }
    std::string value(size, '\0');
    if (::sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return "unknown";
    }
    // sysctl reports the size including the trailing NUL; trim it so the string
    // does not carry an embedded terminator into JSON output.
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

#endif

std::string detect_cpu_brand() {
#if defined(__APPLE__)
    return sysctl_string("machdep.cpu.brand_string");
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        // x86 exposes "model name"; aarch64 commonly exposes only "Hardware" or
        // nothing useful, in which case the caller sees "unknown".
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, colon);
        if (key.starts_with("model name") || key.starts_with("Hardware")) {
            const std::string value = line.substr(colon + 1);
            const auto first = value.find_first_not_of(" \t");
            if (first == std::string::npos) {
                continue;
            }
            const auto last = value.find_last_not_of(" \t");
            return value.substr(first, last - first + 1);
        }
    }
    return "unknown";
#elif defined(_WIN32)
    return "unknown";
#else
    return "unknown";
#endif
}

struct OsDescription {
    std::string name;
    std::string version;
};

OsDescription detect_os() {
#if defined(_WIN32)
    return {.name = "Windows", .version = "unknown"};
#else
    struct utsname info{};
    if (::uname(&info) != 0) {
        return {.name = "unknown", .version = "unknown"};
    }
    // utsname exposes fixed-size char arrays. The casts make the decay to
    // const char* explicit rather than implicit; the fields are NUL-terminated
    // by POSIX, so constructing a string from them is well defined.
    return {.name = std::string(static_cast<const char*>(info.sysname)),
            .version = std::string(static_cast<const char*>(info.release))};
#endif
}

}  // namespace

BuildInfo collect_build_info() {
    BuildInfo info;

    info.version = DW_VERSION_STRING;
    info.compiler_id = DW_COMPILER_ID;
    info.compiler_version = DW_COMPILER_VERSION;
    info.build_type = DW_BUILD_TYPE;
    info.build_flags = DW_BUILD_FLAGS;
    info.cxx_standard = DW_CXX_STANDARD;

    info.git_commit = DW_GIT_COMMIT;
    info.git_commit_short = DW_GIT_COMMIT_SHORT;
    info.git_branch = DW_GIT_BRANCH;

    // Only an explicit "false" means the tree was verified clean. "unknown"
    // (git absent, or `git status` failed) is reported as dirty: claiming a
    // clean tree that was never checked would attach a reproducibility
    // guarantee to a result that has not earned one.
    info.git_dirty = (std::strcmp(DW_GIT_DIRTY, "false") != 0);

    const OsDescription os = detect_os();
    info.os_name = os.name;
    info.os_version = os.version;
    info.cpu_brand = detect_cpu_brand();
    info.logical_cores = std::thread::hardware_concurrency();
    info.hostname = read_hostname();
    info.timestamp_utc = current_utc_timestamp();

    return info;
}

nlohmann::json to_json(const BuildInfo& info) {
    return nlohmann::json{
        {"version", info.version},
        {"compiler_id", info.compiler_id},
        {"compiler_version", info.compiler_version},
        {"build_type", info.build_type},
        {"build_flags", info.build_flags},
        {"cxx_standard", info.cxx_standard},
        {"git_commit", info.git_commit},
        {"git_commit_short", info.git_commit_short},
        {"git_branch", info.git_branch},
        {"git_dirty", info.git_dirty},
        {"os_name", info.os_name},
        {"os_version", info.os_version},
        {"cpu_brand", info.cpu_brand},
        {"logical_cores", info.logical_cores},
        {"hostname", info.hostname},
        {"timestamp_utc", info.timestamp_utc},
    };
}

}  // namespace diffusionworks

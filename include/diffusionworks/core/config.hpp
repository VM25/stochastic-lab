#pragma once

#include <diffusionworks/core/error.hpp>
#include <diffusionworks/core/result.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>

namespace diffusionworks {

/// Version of the configuration schema this build understands.
///
/// Stored in every configuration file as `schema_version`. Loading refuses a
/// document whose version this build does not know, rather than guessing at the
/// meaning of unfamiliar fields.
inline constexpr int kConfigSchemaVersion = 1;

/// A validated view over one JSON object in a configuration document.
///
/// Every accessor returns a Result carrying the *path* of the offending field
/// (for example `market.spot`), because a configuration error that does not say
/// which field was wrong is not actionable.
///
/// Accessors are strict by design (TECHNICAL-DESIGN section 17): a missing
/// required key, a wrong type, an out-of-range value, and an unknown key are all
/// rejections. Silent defaulting is what lets an unintended configuration
/// produce a plausible number, which is the failure mode this layer exists to
/// prevent.
class ConfigNode {
public:
    ConfigNode(const nlohmann::json& node, std::string path);

    /// Dotted path of this node within the document; empty for the root.
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    [[nodiscard]] bool contains(std::string_view key) const;

    // --- Required accessors -------------------------------------------------

    [[nodiscard]] Result<ConfigNode> object(std::string_view key) const;
    [[nodiscard]] Result<ConfigNode> array(std::string_view key) const;
    [[nodiscard]] Result<double> number(std::string_view key) const;
    [[nodiscard]] Result<std::int64_t> integer(std::string_view key) const;
    [[nodiscard]] Result<std::string> string(std::string_view key) const;
    [[nodiscard]] Result<bool> boolean(std::string_view key) const;

    // --- Range-checked accessors -------------------------------------------

    /// Requires lo <= value <= hi and rejects non-finite values.
    [[nodiscard]] Result<double> number_in(std::string_view key, double lo, double hi) const;

    /// Requires value > 0 and finite. Used for spot, strike, maturity,
    /// volatility, and other strictly positive model inputs.
    [[nodiscard]] Result<double> positive_number(std::string_view key) const;

    /// Requires lo <= value <= hi.
    [[nodiscard]] Result<std::int64_t>
    integer_in(std::string_view key, std::int64_t lo, std::int64_t hi) const;

    /// Requires value >= 1. Used for path counts, step counts, and grid sizes.
    [[nodiscard]] Result<std::int64_t> positive_integer(std::string_view key) const;

    // --- Optional accessors -------------------------------------------------

    [[nodiscard]] Result<double> number_or(std::string_view key, double fallback) const;
    [[nodiscard]] Result<std::int64_t> integer_or(std::string_view key,
                                                  std::int64_t fallback) const;
    [[nodiscard]] Result<std::string> string_or(std::string_view key, std::string fallback) const;
    [[nodiscard]] Result<bool> boolean_or(std::string_view key, bool fallback) const;

    // --- Array access -------------------------------------------------------

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] Result<ConfigNode> at(std::size_t index) const;
    [[nodiscard]] Result<double> number_at(std::size_t index) const;

    // --- Strictness ---------------------------------------------------------

    /// Rejects any key on this node outside `allowed`.
    ///
    /// This is what turns a typo into an error instead of a silent default. A
    /// configuration containing `volatilty` must fail loudly; otherwise the run
    /// quietly uses a different volatility than the author intended and the
    /// resulting number looks entirely reasonable.
    [[nodiscard]] Status reject_unknown_keys(std::initializer_list<std::string_view> allowed) const;

    /// Underlying JSON, for callers that must inspect structure directly.
    [[nodiscard]] const nlohmann::json& raw() const noexcept { return *node_; }

private:
    [[nodiscard]] std::string child_path(std::string_view key) const;
    [[nodiscard]] Result<const nlohmann::json*> find(std::string_view key) const;

    const nlohmann::json* node_;
    std::string path_;
};

/// A parsed, schema-checked configuration document.
class ConfigDocument {
public:
    ConfigDocument(nlohmann::json root, std::filesystem::path source);

    [[nodiscard]] ConfigNode root() const;

    [[nodiscard]] const std::filesystem::path& source() const noexcept { return source_; }

    [[nodiscard]] int schema_version() const noexcept { return schema_version_; }

    /// Original document, used to embed the exact configuration in result
    /// artifacts so a published run carries the inputs that produced it.
    [[nodiscard]] const nlohmann::json& json() const noexcept { return root_; }

private:
    nlohmann::json root_;
    std::filesystem::path source_;
    int schema_version_{0};
};

/// Parses a configuration document from text.
///
/// Rejects malformed JSON, a non-object root, a missing or non-integer
/// `schema_version`, and any `schema_version` this build does not implement.
[[nodiscard]] Result<ConfigDocument> parse_config(std::string_view text,
                                                  std::filesystem::path source = {});

/// Reads and parses a configuration file. Reports a missing or unreadable file
/// as ErrorCode::IoFailure, distinct from a malformed one.
[[nodiscard]] Result<ConfigDocument> load_config_file(const std::filesystem::path& path);

}  // namespace diffusionworks

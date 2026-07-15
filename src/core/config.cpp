#include <diffusionworks/core/config.hpp>

#include <fmt/format.h>

#include <cmath>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace diffusionworks {
namespace {

constexpr const char* kContext = "config";

/// Names the JSON type as it appears to a configuration author, so a type error
/// reads in the vocabulary of the file rather than of nlohmann's internals.
std::string_view json_type_name(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::null:
            return "null";
        case nlohmann::json::value_t::object:
            return "object";
        case nlohmann::json::value_t::array:
            return "array";
        case nlohmann::json::value_t::string:
            return "string";
        case nlohmann::json::value_t::boolean:
            return "boolean";
        case nlohmann::json::value_t::number_integer:
        case nlohmann::json::value_t::number_unsigned:
            return "integer";
        case nlohmann::json::value_t::number_float:
            return "number";
        case nlohmann::json::value_t::binary:
            return "binary";
        case nlohmann::json::value_t::discarded:
            return "invalid";
    }
    return "unknown";
}

}  // namespace

// ---------------------------------------------------------------------------
// ConfigNode
// ---------------------------------------------------------------------------

ConfigNode::ConfigNode(const nlohmann::json& node, std::string path)
    : node_(&node), path_(std::move(path)) {}

std::string ConfigNode::child_path(std::string_view key) const {
    if (path_.empty()) {
        return std::string(key);
    }
    return fmt::format("{}.{}", path_, key);
}

bool ConfigNode::contains(std::string_view key) const {
    return node_->is_object() && node_->contains(key);
}

Result<const nlohmann::json*> ConfigNode::find(std::string_view key) const {
    if (!node_->is_object()) {
        return Result<const nlohmann::json*>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("expected an object at '{}' but found {}",
                        path_.empty() ? "<root>" : path_,
                        json_type_name(*node_)),
            kContext);
    }
    const auto it = node_->find(key);
    if (it == node_->end()) {
        return Result<const nlohmann::json*>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("missing required field '{}'", child_path(key)),
            kContext);
    }
    return Result<const nlohmann::json*>::success(&(*it));
}

Result<ConfigNode> ConfigNode::object(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<ConfigNode>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    if (!value->is_object()) {
        return Result<ConfigNode>::failure(ErrorCode::InvalidConfiguration,
                                           fmt::format("field '{}' must be an object but is {}",
                                                       child_path(key),
                                                       json_type_name(*value)),
                                           kContext);
    }
    return Result<ConfigNode>::success(ConfigNode(*value, child_path(key)));
}

Result<ConfigNode> ConfigNode::array(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<ConfigNode>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    if (!value->is_array()) {
        return Result<ConfigNode>::failure(ErrorCode::InvalidConfiguration,
                                           fmt::format("field '{}' must be an array but is {}",
                                                       child_path(key),
                                                       json_type_name(*value)),
                                           kContext);
    }
    return Result<ConfigNode>::success(ConfigNode(*value, child_path(key)));
}

Result<double> ConfigNode::number(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<double>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    if (!value->is_number()) {
        return Result<double>::failure(ErrorCode::InvalidConfiguration,
                                       fmt::format("field '{}' must be a number but is {}",
                                                   child_path(key),
                                                   json_type_name(*value)),
                                       kContext);
    }
    const auto as_double = value->get<double>();
    // JSON cannot express NaN or infinity, but a value can still overflow double
    // on conversion from a very large literal.
    if (!std::isfinite(as_double)) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field '{}' is not a finite number", child_path(key)),
            kContext);
    }
    return Result<double>::success(as_double);
}

Result<std::int64_t> ConfigNode::integer(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<std::int64_t>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    // A float is rejected rather than truncated: `"paths": 1e5` is fine but
    // `"paths": 1000.5` is a mistake the author needs to see.
    if (!value->is_number_integer()) {
        if (value->is_number_float()) {
            const double as_double = value->get<double>();
            double integral_part = 0.0;
            if (std::modf(as_double, &integral_part) == 0.0 && std::isfinite(as_double)) {
                return Result<std::int64_t>::success(static_cast<std::int64_t>(integral_part));
            }
        }
        return Result<std::int64_t>::failure(ErrorCode::InvalidConfiguration,
                                             fmt::format("field '{}' must be an integer but is {}",
                                                         child_path(key),
                                                         json_type_name(*value)),
                                             kContext);
    }
    return Result<std::int64_t>::success(value->get<std::int64_t>());
}

Result<std::string> ConfigNode::string(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<std::string>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    if (!value->is_string()) {
        return Result<std::string>::failure(ErrorCode::InvalidConfiguration,
                                            fmt::format("field '{}' must be a string but is {}",
                                                        child_path(key),
                                                        json_type_name(*value)),
                                            kContext);
    }
    return Result<std::string>::success(value->get<std::string>());
}

Result<bool> ConfigNode::boolean(std::string_view key) const {
    auto found = find(key);
    if (!found) {
        return Result<bool>::failure(std::move(found).error());
    }
    const nlohmann::json* value = found.value();
    if (!value->is_boolean()) {
        return Result<bool>::failure(ErrorCode::InvalidConfiguration,
                                     fmt::format("field '{}' must be a boolean but is {}",
                                                 child_path(key),
                                                 json_type_name(*value)),
                                     kContext);
    }
    return Result<bool>::success(value->get<bool>());
}

Result<double> ConfigNode::number_in(std::string_view key, double lo, double hi) const {
    auto value = number(key);
    if (!value) {
        return value;
    }
    const double v = value.value();
    if (v < lo || v > hi) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field '{}' must lie in [{}, {}] but is {}", child_path(key), lo, hi, v),
            kContext);
    }
    return value;
}

Result<double> ConfigNode::positive_number(std::string_view key) const {
    auto value = number(key);
    if (!value) {
        return value;
    }
    const double v = value.value();
    if (v <= 0.0) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field '{}' must be strictly positive but is {}", child_path(key), v),
            kContext);
    }
    return value;
}

Result<std::int64_t>
ConfigNode::integer_in(std::string_view key, std::int64_t lo, std::int64_t hi) const {
    auto value = integer(key);
    if (!value) {
        return value;
    }
    const std::int64_t v = value.value();
    if (v < lo || v > hi) {
        return Result<std::int64_t>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field '{}' must lie in [{}, {}] but is {}", child_path(key), lo, hi, v),
            kContext);
    }
    return value;
}

Result<std::int64_t> ConfigNode::positive_integer(std::string_view key) const {
    auto value = integer(key);
    if (!value) {
        return value;
    }
    const std::int64_t v = value.value();
    if (v < 1) {
        return Result<std::int64_t>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field '{}' must be at least 1 but is {}", child_path(key), v),
            kContext);
    }
    return value;
}

Result<double> ConfigNode::number_or(std::string_view key, double fallback) const {
    if (!contains(key)) {
        return Result<double>::success(fallback);
    }
    return number(key);
}

Result<std::int64_t> ConfigNode::integer_or(std::string_view key, std::int64_t fallback) const {
    if (!contains(key)) {
        return Result<std::int64_t>::success(fallback);
    }
    return integer(key);
}

Result<std::string> ConfigNode::string_or(std::string_view key, std::string fallback) const {
    if (!contains(key)) {
        return Result<std::string>::success(std::move(fallback));
    }
    return string(key);
}

Result<bool> ConfigNode::boolean_or(std::string_view key, bool fallback) const {
    if (!contains(key)) {
        return Result<bool>::success(fallback);
    }
    return boolean(key);
}

std::size_t ConfigNode::size() const {
    return node_->size();
}

Result<ConfigNode> ConfigNode::at(std::size_t index) const {
    if (!node_->is_array()) {
        return Result<ConfigNode>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("expected an array at '{}' but found {}", path_, json_type_name(*node_)),
            kContext);
    }
    if (index >= node_->size()) {
        return Result<ConfigNode>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format(
                "index {} is out of range for '{}' of size {}", index, path_, node_->size()),
            kContext);
    }
    return Result<ConfigNode>::success(
        ConfigNode((*node_)[index], fmt::format("{}[{}]", path_, index)));
}

Result<double> ConfigNode::number_at(std::size_t index) const {
    if (!node_->is_array()) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("expected an array at '{}' but found {}", path_, json_type_name(*node_)),
            kContext);
    }
    if (index >= node_->size()) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format(
                "index {} is out of range for '{}' of size {}", index, path_, node_->size()),
            kContext);
    }
    const nlohmann::json& value = (*node_)[index];
    if (!value.is_number()) {
        return Result<double>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("'{}[{}]' must be a number but is {}", path_, index, json_type_name(value)),
            kContext);
    }
    const auto as_double = value.get<double>();
    if (!std::isfinite(as_double)) {
        return Result<double>::failure(ErrorCode::InvalidConfiguration,
                                       fmt::format("'{}[{}]' is not a finite number", path_, index),
                                       kContext);
    }
    return Result<double>::success(as_double);
}

Status ConfigNode::reject_unknown_keys(std::initializer_list<std::string_view> allowed) const {
    if (!node_->is_object()) {
        return Status::failure(ErrorCode::InvalidConfiguration,
                               fmt::format("expected an object at '{}' but found {}",
                                           path_.empty() ? "<root>" : path_,
                                           json_type_name(*node_)),
                               kContext);
    }

    std::vector<std::string> unknown;
    for (const auto& entry : node_->items()) {
        bool permitted = false;
        for (const std::string_view candidate : allowed) {
            if (entry.key() == candidate) {
                permitted = true;
                break;
            }
        }
        if (!permitted) {
            unknown.push_back(entry.key());
        }
    }

    if (unknown.empty()) {
        return Status::success();
    }

    // Listing the accepted keys alongside the rejected one turns a typo into a
    // one-look fix.
    std::string allowed_list;
    for (const std::string_view candidate : allowed) {
        if (!allowed_list.empty()) {
            allowed_list += ", ";
        }
        allowed_list += candidate;
    }

    std::string unknown_list;
    for (const std::string& key : unknown) {
        if (!unknown_list.empty()) {
            unknown_list += ", ";
        }
        unknown_list += fmt::format("'{}'", child_path(key));
    }

    return Status::failure(ErrorCode::InvalidConfiguration,
                           fmt::format("unknown field(s) {}; accepted fields for '{}' are: {}",
                                       unknown_list,
                                       path_.empty() ? "<root>" : path_,
                                       allowed_list),
                           kContext);
}

// ---------------------------------------------------------------------------
// ConfigDocument
// ---------------------------------------------------------------------------

ConfigDocument::ConfigDocument(nlohmann::json root, std::filesystem::path source)
    : root_(std::move(root)), source_(std::move(source)) {
    if (root_.is_object() && root_.contains("schema_version") &&
        root_["schema_version"].is_number_integer()) {
        schema_version_ = root_["schema_version"].get<int>();
    }
}

ConfigNode ConfigDocument::root() const {
    return {root_, ""};
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

Result<ConfigDocument> parse_config(std::string_view text, std::filesystem::path source) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(text);
    } catch (const nlohmann::json::parse_error& e) {
        return Result<ConfigDocument>::failure(
            ErrorCode::ParseFailure,
            fmt::format("malformed JSON at byte {}: {}", e.byte, e.what()),
            kContext);
    }

    if (!root.is_object()) {
        return Result<ConfigDocument>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("configuration root must be an object but is {}", json_type_name(root)),
            kContext);
    }

    if (!root.contains("schema_version")) {
        return Result<ConfigDocument>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("missing required field 'schema_version' (this build expects {})",
                        kConfigSchemaVersion),
            kContext);
    }

    if (!root["schema_version"].is_number_integer()) {
        return Result<ConfigDocument>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("field 'schema_version' must be an integer but is {}",
                        json_type_name(root["schema_version"])),
            kContext);
    }

    const auto version = root["schema_version"].get<int>();
    if (version != kConfigSchemaVersion) {
        // Refusing an unknown version is deliberate: reinterpreting fields whose
        // meaning may have changed would silently produce a run the author did
        // not describe.
        return Result<ConfigDocument>::failure(
            ErrorCode::InvalidConfiguration,
            fmt::format("unsupported schema_version {} (this build implements {})",
                        version,
                        kConfigSchemaVersion),
            kContext);
    }

    return Result<ConfigDocument>::success(ConfigDocument(std::move(root), std::move(source)));
}

Result<ConfigDocument> load_config_file(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return Result<ConfigDocument>::failure(
            ErrorCode::IoFailure,
            fmt::format("configuration file not found: {}", path.string()),
            kContext);
    }
    if (std::filesystem::is_directory(path, ec)) {
        return Result<ConfigDocument>::failure(
            ErrorCode::IoFailure,
            fmt::format("configuration path is a directory, not a file: {}", path.string()),
            kContext);
    }

    // Every member used below happens to be const-qualified, but the stream is
    // being consumed. Declaring it const would compile while telling the reader
    // the opposite of what the code does.
    // NOLINTNEXTLINE(misc-const-correctness)
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Result<ConfigDocument>::failure(
            ErrorCode::IoFailure,
            fmt::format("cannot open configuration file: {}", path.string()),
            kContext);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad()) {
        return Result<ConfigDocument>::failure(
            ErrorCode::IoFailure,
            fmt::format("error while reading configuration file: {}", path.string()),
            kContext);
    }

    return parse_config(buffer.str(), path);
}

}  // namespace diffusionworks

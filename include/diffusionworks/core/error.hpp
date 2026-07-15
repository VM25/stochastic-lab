#pragma once

#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace diffusionworks {

/// Classification of an expected, recoverable failure.
///
/// TECHNICAL-DESIGN section 20 distinguishes two failure classes: unrecoverable
/// boundary-level failures (exceptions) and expected numerical failures
/// (explicit result states). This enum names the second class. Every code here
/// describes a condition the engine anticipates and must report rather than
/// convert into a zero, a NaN, or a plausible-looking value.
enum class ErrorCode {
    /// Caller supplied a value outside the domain of the operation.
    InvalidArgument,

    /// A configuration file is structurally valid but semantically rejected.
    InvalidConfiguration,

    /// Configuration could not be parsed at all.
    ParseFailure,

    /// The requested model/instrument/method triple is not supported.
    UnsupportedCombination,

    /// A computed quantity was NaN or infinite.
    NonFiniteValue,

    /// An iterative method exhausted its budget without meeting its tolerance.
    ConvergenceFailure,

    /// A root-finding problem has no root in the supplied bracket, or the
    /// target lies outside admissible bounds.
    RootNotBracketed,

    /// Numerical quadrature failed to reach its requested tolerance.
    IntegrationFailure,

    /// A discretization scheme was run in a configuration known to be unstable.
    UnstableScheme,

    /// A simulated path reached an invalid state (non-finite, or otherwise
    /// outside the model's admissible domain).
    PathFailure,

    /// Reading or writing an artifact failed.
    IoFailure,

    /// Reached a code path that is declared but not yet implemented. This is a
    /// developer error and must never appear in a released result.
    NotImplemented,
};

/// Stable machine-readable spelling of an error code, used in JSON artifacts.
[[nodiscard]] constexpr std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::InvalidArgument:
            return "invalid_argument";
        case ErrorCode::InvalidConfiguration:
            return "invalid_configuration";
        case ErrorCode::ParseFailure:
            return "parse_failure";
        case ErrorCode::UnsupportedCombination:
            return "unsupported_combination";
        case ErrorCode::NonFiniteValue:
            return "non_finite_value";
        case ErrorCode::ConvergenceFailure:
            return "convergence_failure";
        case ErrorCode::RootNotBracketed:
            return "root_not_bracketed";
        case ErrorCode::IntegrationFailure:
            return "integration_failure";
        case ErrorCode::UnstableScheme:
            return "unstable_scheme";
        case ErrorCode::PathFailure:
            return "path_failure";
        case ErrorCode::IoFailure:
            return "io_failure";
        case ErrorCode::NotImplemented:
            return "not_implemented";
    }
    return "unknown";
}

/// An expected failure, carrying enough detail to diagnose it without a
/// debugger.
///
/// `context` names the component that detected the failure (for example
/// "BlackScholesEngine::price"). It is kept separate from `message` so that
/// artifacts can group failures by origin.
struct Error {
    ErrorCode code{ErrorCode::InvalidArgument};
    std::string message;
    std::string context;

    Error() = default;

    Error(ErrorCode error_code, std::string error_message)
        : code(error_code), message(std::move(error_message)) {}

    Error(ErrorCode error_code, std::string error_message, std::string error_context)
        : code(error_code), message(std::move(error_message)), context(std::move(error_context)) {}

    /// Human-readable single-line rendering: "context: message [code]".
    [[nodiscard]] std::string describe() const {
        std::string out;
        if (!context.empty()) {
            out += context;
            out += ": ";
        }
        out += message;
        out += " [";
        out += to_string(code);
        out += "]";
        return out;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Error& error) {
    return os << error.describe();
}

/// Thrown for unrecoverable boundary-level failures, and by Result<T>::value()
/// when a caller reads a value that does not exist.
///
/// Numerical failures inside engines must not use this type; they belong in a
/// Result so the caller is forced to handle them.
class DiffusionWorksError : public std::runtime_error {
public:
    explicit DiffusionWorksError(Error error)
        : std::runtime_error(error.describe()), error_(std::move(error)) {}

    [[nodiscard]] const Error& error() const noexcept { return error_; }

private:
    Error error_;
};

}  // namespace diffusionworks

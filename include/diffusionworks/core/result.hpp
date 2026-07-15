#pragma once

#include <diffusionworks/core/error.hpp>

#include <type_traits>
#include <utility>
#include <variant>

namespace diffusionworks {

/// A value or an Error, never both.
///
/// The project standard (ADR-009) is that fallible operations return a
/// structured result rather than a bare value, so that an expected numerical
/// failure cannot be mistaken for a successful computation. Result<T> is the
/// carrier for that rule.
///
/// This is a deliberately small stand-in for std::expected, which is C++23; the
/// project targets C++20 for GCC/Clang portability. Only the operations the
/// engine actually needs are provided.
///
/// Reading value() on a failed Result throws DiffusionWorksError. That is a
/// programming error, not a numerical one: callers are expected to branch on
/// ok() or use value_or().
template<typename T>
class [[nodiscard]] Result {
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Error>,
                  "Result<Error> is ambiguous; use Result<void> or a distinct type");

public:
    using value_type = T;

    Result(T value) : storage_(std::in_place_index<0>, std::move(value)) {}  // NOLINT

    Result(Error error) : storage_(std::in_place_index<1>, std::move(error)) {}  // NOLINT

    [[nodiscard]] static Result success(T value) { return Result(std::move(value)); }

    [[nodiscard]] static Result failure(Error error) { return Result(std::move(error)); }

    [[nodiscard]] static Result
    failure(ErrorCode code, std::string message, std::string context = {}) {
        return Result(Error{code, std::move(message), std::move(context)});
    }

    [[nodiscard]] bool ok() const noexcept { return storage_.index() == 0; }

    explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] const T& value() const& {
        ensure_value();
        return std::get<0>(storage_);
    }

    [[nodiscard]] T& value() & {
        ensure_value();
        return std::get<0>(storage_);
    }

    [[nodiscard]] T&& value() && {
        ensure_value();
        return std::get<0>(std::move(storage_));
    }

    template<typename U>
    [[nodiscard]] T value_or(U&& fallback) const& {
        return ok() ? std::get<0>(storage_) : static_cast<T>(std::forward<U>(fallback));
    }

    [[nodiscard]] const Error& error() const& {
        ensure_error();
        return std::get<1>(storage_);
    }

    [[nodiscard]] Error&& error() && {
        ensure_error();
        return std::get<1>(std::move(storage_));
    }

    const T* operator->() const { return &value(); }

    T* operator->() { return &value(); }

    const T& operator*() const& { return value(); }

    T& operator*() & { return value(); }

private:
    void ensure_value() const {
        if (!ok()) {
            throw DiffusionWorksError(std::get<1>(storage_));
        }
    }

    void ensure_error() const {
        if (ok()) {
            throw DiffusionWorksError(Error{
                ErrorCode::InvalidArgument, "Result holds a value, not an error", "Result::error"});
        }
    }

    std::variant<T, Error> storage_;
};

/// Specialization for operations that either succeed or fail without producing
/// a value (validation passes, writes, in-place mutations).
template<>
class [[nodiscard]] Result<void> {
public:
    using value_type = void;

    Result() = default;

    Result(Error error) : error_(std::move(error)), has_error_(true) {}  // NOLINT

    [[nodiscard]] static Result success() { return Result(); }

    [[nodiscard]] static Result failure(Error error) { return Result(std::move(error)); }

    [[nodiscard]] static Result
    failure(ErrorCode code, std::string message, std::string context = {}) {
        return Result(Error{code, std::move(message), std::move(context)});
    }

    [[nodiscard]] bool ok() const noexcept { return !has_error_; }

    explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] const Error& error() const& {
        if (ok()) {
            throw DiffusionWorksError(Error{
                ErrorCode::InvalidArgument, "Result holds a value, not an error", "Result::error"});
        }
        return error_;
    }

private:
    Error error_{};
    bool has_error_{false};
};

using Status = Result<void>;

}  // namespace diffusionworks

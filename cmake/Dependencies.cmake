# Dependency acquisition.
#
# Pinned to exact tags so a clean checkout resolves identical dependency
# versions. Per ARCHITECTURE-DECISIONS ADR-001/003, none of these implement the
# project's models, pricing, simulation, Greeks, or calibration; they provide
# formatting, JSON, and testing support only.

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# --- fmt: console and report formatting -------------------------------------
#
# 11.2.0 or newer is required: fmt 11.0.x fails to compile under Clang 19+,
# which rejects its consteval format-string constructors as non-constant
# expressions (fmtlib/fmt#4177). Apple Clang 21 hits this.

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS NAMES fmt)

# --- nlohmann/json: configuration parsing and result serialization ----------

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS NAMES nlohmann_json)

FetchContent_MakeAvailable(fmt nlohmann_json)

# --- GoogleTest -------------------------------------------------------------

if(DW_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
        SYSTEM
        FIND_PACKAGE_ARGS NAMES GTest)

    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
endif()

# --- QuantLib (optional, external validation only) --------------------------
#
# Per ADR-003 QuantLib is a validation-only reference. It is never linked into
# the core library or the CLI; only dedicated cross-validation targets use it.

if(DW_ENABLE_QUANTLIB)
    # Discovery has to cope with two packagings. QuantLib built with CMake
    # installs QuantLibConfig.cmake, but Homebrew's bottle ships neither a CMake
    # config nor a pkg-config file -- only the legacy quantlib-config script. So
    # try the config package first and fall back to locating the header and
    # library directly.
    find_package(QuantLib QUIET CONFIG)

    if(NOT QuantLib_FOUND)
        set(_dw_quantlib_hints "")
        find_program(DW_BREW_EXECUTABLE brew)
        if(DW_BREW_EXECUTABLE)
            execute_process(
                COMMAND "${DW_BREW_EXECUTABLE}" --prefix quantlib
                OUTPUT_VARIABLE _dw_brew_quantlib_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _dw_brew_result)
            if(_dw_brew_result EQUAL 0 AND _dw_brew_quantlib_prefix)
                list(APPEND _dw_quantlib_hints "${_dw_brew_quantlib_prefix}")
            endif()
        endif()

        find_path(
            DW_QUANTLIB_INCLUDE_DIR ql/quantlib.hpp
            HINTS ${_dw_quantlib_hints}
            PATH_SUFFIXES include)

        find_library(
            DW_QUANTLIB_LIBRARY
            NAMES QuantLib
            HINTS ${_dw_quantlib_hints}
            PATH_SUFFIXES lib)

        # ql/qldefines.hpp includes boost/config.hpp, so QuantLib's headers do
        # not stand alone even when its shared_ptr maps to std::shared_ptr.
        # A CMake config package would carry this dependency; discovered by
        # hand, it has to be supplied here.
        set(_dw_boost_hints "")
        if(DW_BREW_EXECUTABLE)
            execute_process(
                COMMAND "${DW_BREW_EXECUTABLE}" --prefix boost
                OUTPUT_VARIABLE _dw_brew_boost_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _dw_brew_boost_result)
            if(_dw_brew_boost_result EQUAL 0 AND _dw_brew_boost_prefix)
                list(APPEND _dw_boost_hints "${_dw_brew_boost_prefix}")
            endif()
        endif()

        find_path(
            DW_BOOST_INCLUDE_DIR boost/config.hpp
            HINTS ${_dw_boost_hints}
            PATH_SUFFIXES include)

        if(DW_QUANTLIB_INCLUDE_DIR AND DW_QUANTLIB_LIBRARY AND DW_BOOST_INCLUDE_DIR)
            add_library(QuantLib::QuantLib UNKNOWN IMPORTED)
            set_target_properties(
                QuantLib::QuantLib
                PROPERTIES IMPORTED_LOCATION "${DW_QUANTLIB_LIBRARY}"
                           # SYSTEM so QuantLib's and Boost's headers are not
                           # compiled under this project's -Werror settings.
                           INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                           "${DW_QUANTLIB_INCLUDE_DIR};${DW_BOOST_INCLUDE_DIR}"
                           INTERFACE_INCLUDE_DIRECTORIES
                           "${DW_QUANTLIB_INCLUDE_DIR};${DW_BOOST_INCLUDE_DIR}")
            set(QuantLib_FOUND TRUE)
            message(STATUS "QuantLib found via direct discovery: ${DW_QUANTLIB_LIBRARY}")
            message(STATUS "  Boost headers: ${DW_BOOST_INCLUDE_DIR}")
        elseif(DW_QUANTLIB_INCLUDE_DIR AND DW_QUANTLIB_LIBRARY AND NOT DW_BOOST_INCLUDE_DIR)
            message(WARNING "QuantLib was found but its Boost headers were not; "
                            "install Boost or set DW_BOOST_INCLUDE_DIR.")
        endif()
    else()
        message(STATUS "QuantLib found via CMake config package")
    endif()

    if(NOT QuantLib_FOUND)
        # A warning rather than an error: QuantLib is an optional reference, and
        # a clean checkout must build and test fully without it.
        message(WARNING
                "DW_ENABLE_QUANTLIB=ON but QuantLib was not found. "
                "External cross-validation targets will be skipped. "
                "Install with: brew install quantlib | apt-get install libquantlib0-dev")
        set(DW_ENABLE_QUANTLIB OFF CACHE BOOL "" FORCE)
    endif()
endif()

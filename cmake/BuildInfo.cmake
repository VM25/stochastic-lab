# Build metadata capture.
#
# PROJECT-SPEC requires every published run to record compiler, build type,
# flags, OS, CPU, and git commit. Metadata splits by when it is knowable:
#
#   compile time - compiler id/version, build type, flags, C++ standard, commit
#   run time     - OS, CPU brand, core count, host, timestamp, thread count
#
# CPU and OS are resolved at run time because the machine that executes a
# benchmark is not necessarily the machine that compiled it.

set(DW_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/diffusionworks/core")
file(MAKE_DIRECTORY "${DW_GENERATED_DIR}")

# --- Git metadata, refreshed on every build ---------------------------------

set(DW_GIT_INFO_HEADER "${DW_GENERATED_DIR}/git_info.hpp")

add_custom_target(
    dw_git_info ALL
    COMMAND "${CMAKE_COMMAND}"
            -DDW_GIT_OUTPUT=${DW_GIT_INFO_HEADER}
            -DDW_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P "${CMAKE_SOURCE_DIR}/cmake/GenerateGitInfo.cmake"
    BYPRODUCTS "${DW_GIT_INFO_HEADER}"
    COMMENT "Capturing git commit metadata"
    VERBATIM)

# Generate once at configure time so the first build has a header to include.
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -DDW_GIT_OUTPUT=${DW_GIT_INFO_HEADER}
            -DDW_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P "${CMAKE_SOURCE_DIR}/cmake/GenerateGitInfo.cmake")

# --- Compiler and build metadata --------------------------------------------

# Flags are the union of the global flags and the per-configuration flags, which
# is what actually reaches the compiler for the selected build type.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _dw_build_type_upper)
set(_dw_config_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_dw_build_type_upper}}")
string(STRIP "${_dw_config_flags}" _dw_config_flags)

set(DW_BUILD_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
set(DW_BUILD_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
set(DW_BUILD_TYPE_STR "${CMAKE_BUILD_TYPE}")
set(DW_BUILD_FLAGS "${_dw_config_flags}")
set(DW_BUILD_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
set(DW_BUILD_VERSION "${PROJECT_VERSION}")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/build_config.hpp.in"
    "${DW_GENERATED_DIR}/build_config.hpp"
    @ONLY)

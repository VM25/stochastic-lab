# ---------------------------------------------------------------------------
# diffusionworks_core: the numerical engine
#
# Per TECHNICAL-DESIGN section 1, this library holds all model, pricing,
# simulation, statistics, and validation code and is independent of CLI parsing,
# plotting, and reporting. The CLI links against it; it never links back.
# ---------------------------------------------------------------------------

add_library(diffusionworks_core)
add_library(diffusionworks::core ALIAS diffusionworks_core)

target_sources(
    diffusionworks_core
    PRIVATE core/build_info.cpp
            core/config.cpp)

target_include_directories(
    diffusionworks_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           # Generated headers (build_config.hpp, git_info.hpp) live under the
           # same diffusionworks/core/ prefix as the checked-in headers, so both
           # are reached by the canonical <diffusionworks/...> include path.
           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>)

target_link_libraries(
    diffusionworks_core
    PUBLIC nlohmann_json::nlohmann_json
           fmt::fmt
    PRIVATE dw::warnings
            dw::sanitizers)

# Sanitizer flags must reach every consumer's compile and link lines, so they are
# also applied publicly. dw::sanitizers is an empty interface unless a sanitizer
# preset enabled one.
target_link_libraries(diffusionworks_core PUBLIC dw::sanitizers)

add_dependencies(diffusionworks_core dw_git_info)

set_target_properties(
    diffusionworks_core
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               POSITION_INDEPENDENT_CODE ON)

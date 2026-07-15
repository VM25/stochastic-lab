# Strict warning configuration.
#
# Applied through an INTERFACE target rather than global flags so that
# FetchContent dependencies are not compiled under project warning settings.

add_library(dw_warnings INTERFACE)
add_library(dw::warnings ALIAS dw_warnings)

set(_dw_gcc_clang_warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wnull-dereference)

set(_dw_gcc_only_warnings
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast)

set(_dw_msvc_warnings /W4 /permissive-)

if(MSVC)
    target_compile_options(dw_warnings INTERFACE ${_dw_msvc_warnings})
    if(DW_WARNINGS_AS_ERRORS)
        target_compile_options(dw_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(dw_warnings INTERFACE ${_dw_gcc_clang_warnings})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(dw_warnings INTERFACE ${_dw_gcc_only_warnings})
    endif()
    if(DW_WARNINGS_AS_ERRORS)
        target_compile_options(dw_warnings INTERFACE -Werror)
    endif()
endif()

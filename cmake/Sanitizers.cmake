# Sanitizer configuration exposed as an INTERFACE target.
#
# ASan and UBSan compose. TSan is mutually exclusive with ASan and is therefore
# configured as a separate preset rather than combined.

add_library(dw_sanitizers INTERFACE)
add_library(dw::sanitizers ALIAS dw_sanitizers)

set(_dw_sanitizer_list "")

if(DW_ENABLE_ASAN AND DW_ENABLE_TSAN)
    message(FATAL_ERROR
            "AddressSanitizer and ThreadSanitizer cannot be enabled simultaneously. "
            "Use the 'sanitize' preset for ASan+UBSan or the 'tsan' preset for ThreadSanitizer.")
endif()

if(MSVC)
    if(DW_ENABLE_ASAN)
        target_compile_options(dw_sanitizers INTERFACE /fsanitize=address)
    endif()
else()
    if(DW_ENABLE_ASAN)
        list(APPEND _dw_sanitizer_list address)
    endif()
    if(DW_ENABLE_UBSAN)
        list(APPEND _dw_sanitizer_list undefined)
    endif()
    if(DW_ENABLE_TSAN)
        list(APPEND _dw_sanitizer_list thread)
    endif()

    if(_dw_sanitizer_list)
        list(JOIN _dw_sanitizer_list "," _dw_sanitizer_arg)
        target_compile_options(dw_sanitizers INTERFACE
                               -fsanitize=${_dw_sanitizer_arg}
                               -fno-omit-frame-pointer
                               -fno-sanitize-recover=all
                               -g)
        target_link_options(dw_sanitizers INTERFACE -fsanitize=${_dw_sanitizer_arg})
        message(STATUS "Sanitizers enabled: ${_dw_sanitizer_arg}")
    endif()
endif()

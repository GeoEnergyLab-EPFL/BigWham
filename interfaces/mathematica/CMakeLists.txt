## Make directories
add_custom_target(copy_math_root ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/LibraryResources/Source
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/Kernel
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/PacletInfo.m ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/"BigWhamLink/LibraryResources/Source/HMatExpr.h"
                                                ${CMAKE_CURRENT_SOURCE_DIR}/"BigWhamLink/LibraryResources/Source/Manager.h"
                                                ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/LibraryResources/Source
)

set(_il_compile_definitions)
foreach(_flags ${IL_COMPILE_DEFINITIONS})
  list(APPEND _il_compile_definitions "-D${_flags}")
endforeach()

set(_il_link_options)
set(_il_link_dirs)
foreach(_flag ${IL_LINK_OPTIONS})
  if(NOT ${_flag} MATCHES "-Wl")
    if(EXISTS ${_flag})
      get_filename_component(_blas_library_dir "${_flag}" DIRECTORY)
      get_filename_component(_blas_library "${_flag}" NAME_WE)
      string(REPLACE ${CMAKE_SHARED_LIBRARY_SUFFIX} "${CMAKE_STATIC_LIBRARY_SUFFIX}" _blas_library_flag "${_flag}")
      #message("${_blas_library} ${_blas_library_flag}")
      list(APPEND _il_link_dirs ${_blas_library_dir})
      list(APPEND _il_link_options "${_blas_library_flag}")
    else()
      list(APPEND _il_link_options "${_flag}")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES _il_link_dirs)


quote_arguments(MMA_COMPILER_OPTIONS "-std=c++17" ${IL_COMPILE_OPTIONS} ${_il_compile_definitions})
quote_arguments(MMA_LINKER_OPTIONS ${_il_link_options})
quote_arguments(MMA_LIBRARY_DIRS ${_il_link_dirs})
quote_arguments(MMA_INCLUDE_DIRS ${IL_INCLUDE_DIRS} ${BIGWHAM_INCLUDE_DIRS})

#message("LTEMPLATE_DIR: ${LTEMPLATE_DIR}")
#message("MMA_COMPILER_OPTIONS: ${MMA_COMPILER_OPTIONS}")
#message("MMA_LINKER_OPTIONS: ${MMA_LINKER_OPTIONS}")
#message("MMA_INCLUDE_DIRS: ${MMA_INCLUDE_DIRS}")
#message("MMA_LIBRARY_DIRS: ${MMA_LIBRARY_DIRS}")
# This is a bit hacky, since I cannot retreive link informations from the

configure_file(BigWhamLink/Kernel/BuildSettings.m
  ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/Kernel/BuildSettings.m
  @ONLY)
configure_file(BigWhamLink/Kernel/BigWhamLink.m
  ${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/Kernel/BigWhamLink.m
  @ONLY)
configure_file(compile.m
  ${CMAKE_CURRENT_BINARY_DIR}/compile.m
  @ONLY)

find_file(WOLFRAMSCRIPT_EXE wolframscript REQUIRED)

mark_as_advanced(WOLFRAMSCRIPT_EXE)

set(MMA_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BigWhamLink/LibraryResources/${CMAKE_HOST_SYSTEM_NAME}-x86-64/HmatExpressions${CMAKE_SHARED_LIBRARY_SUFFIX}")
# message("  OUTPUT ${MMA_OUTPUT}")
add_custom_command(
  OUTPUT "${MMA_OUTPUT}"
  COMMAND ${WOLFRAMSCRIPT_EXE} -print all -v -f compile.m
  DEPENDS BigWhamStatic copy_math_root
  COMMENT "Building kthe Matematica Interface"
  )
add_custom_target(MathematicaInterface ALL
  DEPENDS ${MMA_OUTPUT})

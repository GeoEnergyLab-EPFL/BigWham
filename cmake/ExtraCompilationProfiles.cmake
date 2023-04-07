#===============================================================================
# @author Nicolas Richart <nicolas.richart@epfl.ch>
#
# @section LICENSE
#
# Copyright (©) 2016-2021 EPFL (Ecole Polytechnique Fédérale de Lausanne)
# Laboratory (LSMS - Laboratoire de Simulation en Mécanique des Solides)
#
# Akantu is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# Akantu is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
# 
# You should have received a copy of the GNU Lesser General Public License along
# with Akantu. If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================

option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)
mark_as_advanced(FORCE_COLORED_OUTPUT)
if(FORCE_COLORED_OUTPUT)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_flags(cxx "-fcolor-diagnostics")
  else()
    add_flags(cxx "-fdiagnostics-color=always")
  endif()
endif()



set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG"
  CACHE STRING "Flags used by the compiler during release builds" FORCE)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} -ggdb3"
    CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT} -ggdb3"
    CACHE STRING "Flags used by the compiler during debug builds" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} -march=native"
    CACHE STRING "Flags used by the compiler during release builds" FORCE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} -xHost"
    CACHE STRING "Flags used by the compiler during release builds" FORCE)
endif()

function(declare_compilation_profile name)
  include(CMakeParseArguments)

  cmake_parse_arguments(_args
    "" "COMPILER;LINKER;DOC" "" ${ARGN})

  string(TOUPPER "${name}" _u_name)

  if(NOT _args_DOC)
    string(TOLOWER "${name}" _args_DOC)
  endif()

  if(NOT _args_COMPILER)
    message(FATAL_ERROR "declare_compilation_profile: you should at least give COMPILER flags")
  endif()

  if(NOT _args_LINKER)
    set(_args_LINKER ${_args_COMPILER})
  endif()

  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  foreach(_flag ${_enabled_languages} SHARED_LINKER EXE_LINKER)
    set(_stage "compiler")
    set(_flags ${_args_COMPILER})
    if(_stage MATCHES ".*LINKER")
      set(_stage "linker")
      set(_flags ${_args_LINKER})
    endif()
    set(CMAKE_${_flag}_FLAGS_${_u_name} ${_flags}
      CACHE STRING "Flags used by the ${_stage} during ${name} builds" FORCE)
    mark_as_advanced(CMAKE_${_flag}_FLAGS_${_u_name})
  endforeach()
endfunction()


# Profiling
declare_compilation_profile(PROFILING
  COMPILER "-g -ggdb3 -pg -DNDEBUG -O3")

# Valgrind
declare_compilation_profile(VALGRIND
  COMPILER "-g -ggdb3 -O3")

# Coverage
declare_compilation_profile(COVERAGE
  COMPILER "-g -ggdb3 -DNDEBUG -O2 --coverage")

# Sanitize the code
if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.2") OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_blacklist " -fsanitize-blacklist=${PROJECT_SOURCE_DIR}/cmake/sanitize-blacklist.txt")
  endif()

  declare_compilation_profile(SANITIZE
    COMPILER "-g -ggdb3 -DNDEBUG -O2 -fsanitize=address -fsanitize=leak -fsanitize=undefined -fno-omit-frame-pointer${_blacklist}")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  declare_compilation_profile(SANITIZEMEMORY
    COMPILER "-g -ggdb3 -O2 -fPIE -fsanitize=memory -fsanitize-memory-track-origins -fsanitize-recover=all -fno-omit-frame-pointer${_blacklist}"
    DOC "\"sanitize memory\"")
endif()

if (CMAKE_BUILD_TYPE MATCHES "[Vv][Aa][Ll][Gg][Rr][Ii][Nn][Dd]")
  find_program(VALGRIND_EXECUTABLE valgrind)
endif()

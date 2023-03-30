download_external_project(nanobind
  URL "https://github.com/wjakob/nanobind.git"
  TAG "v${_nanobind_version}"
  BACKEND GIT
  THIRD_PARTY_SRC_DIR ${_nanobind_external_dir}
  NO_UPDATE
  )

set(NANOBIND_LTO_CXX_FLAGS "" CACHE INTERNAL "")
set(NANOBIND_LTO_LINKER_FLAGS "" CACHE INTERNAL "")
set(NANOBIND_PYTHON_VERSION 3.8 CACHE INTERNAL "")

add_subdirectory(${_nanobind_external_dir}/nanobind)

include_directories(SYSTEM ${NANOBIND_INCLUDE_DIR})

mark_as_advanced_prefix(NANOBIND)
mark_as_advanced(USE_PYTHON_INCLUDE_DIR)

set(${package}_FOUND TRUE CACHE INTERNAL "To avoid cyclic search" FORCE)
set(${package}_FOUND_EXTERNAL TRUE CACHE INTERNAL "" FORCE)

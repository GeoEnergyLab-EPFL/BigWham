#####################################
# BIGWHAM                           #
# compilation configuration         #
#####################################
cmake_minimum_required(VERSION 3.5.1)

if(CMAKE_VERSION VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(BigWham VERSION 1.0
        DESCRIPTION "This is the library containing the wheels"
        LANGUAGES CXX)

# This is only for pybind11 to behave properly
set(CMAKE_CXX_STANDARD 17)


# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "CMAKE_BUILD_TYPE :  ${CMAKE_BUILD_TYPE}")

#if(NOT CMAKE_BUILD_TYPE)
#    message(STATUS "Build Type not set, defaulting to Release..." )
#    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type")
#endif()

# set options: False = 0, True = 1  #
#####################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BIGWHAM_PYTHON_INTERFACE "Build Pybind11 binding" ON)
option(BIGWHAM_MATHEMATICA_INTERFACE "make mathematica interface" OFF)  # --> to add static option to CMAKE_CXX_FLAGS
option(BIGWHAM_TESTING "Build google tests" OFF)
option(BIGWHAM_HDF5 "Use HDF5 for checkpoint/restart" OFF)
option(BIGWHAM_DOCUMENTATION "Build documentation using doxygen/breathe/sphinx" OFF)

#option(USE_STATIC "use static" OFF)  # --> to add static option to CMAKE_CXX_FLAGS

#####################################
# optimization options              #
#####################################
set(_optimization_flags "-O3 -DNDEBUG")
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(_optimization_flags "${_optimization_flags} -xHost")
else()
    set(_optimization_flags "${_optimization_flags} -march=native")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${_optimization_flags}" CACHE STRING "Flags used by the compiler during release builds" FORCE)


#if(USE_STATIC)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
#endif()

################################################################################
# set compiler: clang, gcc, intel etc.
################################################################################
message(STATUS "Detecting the compiler:")
message(STATUS "  The compiler is:  ${CMAKE_CXX_COMPILER}")

if(BIGWHAM_MATHEMATICA_INTERFACE)
  set(IL_MATH_VENDOR "MKL_sequential" CACHE INTERNAL "" FORCE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(IL_OPENMP OFF CACHE INTERNAL "" FORCE)
  endif()
endif()

add_subdirectory(il)

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CBLAS_LIBRARY_DIRS}")
#set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CBLAS_LIBRARY_DIRS}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)


################################################################################
# file definitions
################################################################################
    # # 2D source code
    #     src/elasticity/FsIsoSp3dSegment/ElasticS3DP0_element.cpp
    #     src/elasticity/FsIso2dSegment/Elastic2DP1_element.cpp
    #     src/elasticity/oldies/PostProcessDDM_2d.cpp
    #     src/elasticity/oldies/PostProcessDDM_3d.cpp
    # #  - T6 element
    # src/elasticity/3d/Elastic3DT6_element.cpp
    # src/elasticity/3d/h_potential_3DT6.cpp
    # src/elasticity/3d/tensor_utilities_3DT6.cpp
    # src/elasticity/3d/ElasticHMatrix3DT6.h
    # #  - T0 element
    # src/elasticity/3d/Elastic3DT0_element.cpp
    # src/elasticity/3d/ElasticHMatrix3DT0.h
    # src/elasticity/3d/ElasticHMatrix3DT0_modes2and3.h
    # #  - R0 element
    # src/elasticity/3d/Elastic3DR0_common.cpp
    # src/elasticity/3d/Elastic3DR0_element.cpp
    # src/elasticity/3d/Elastic3DR0_mode1Cartesian_element.cpp
    # src/elasticity/3d/ElasticHMatrix3DR0.h
    # src/elasticity/3d/ElasticHMatrix3DR0displ.h
    # src/elasticity/3d/ElasticHMatrix3DR0_mode1Cartesian.h
    # src/elasticity/3d/ElasticHMatrix3DR0_modes2and3Cartesian.h
    # #new
    #     src/elasticity/BieElastostatic.h
    #     src/elasticity/FsIso2dSegment/BIE_elastostatic_segment_0_impls.h
    #     src/elasticity/FsIso2dSegment/BIE_elastostatic_segment_1_impls.h
    # src/elasticity/3d/BIE_elastostatic_triangle_0_impls.h
    #     src/elasticity/FsIso2dSegment/Elastic2D_segment.cpp
        # src/elasticity/FsIsoSp3dSegment/BieElastostaticSp3d.h
      # )


# Source files are added by src subdirectories
set(SOURCE_FILES "")
add_subdirectory(src)

# message(STATUS ${SOURCE_FILES})


add_library(BigWham OBJECT ${SOURCE_FILES})
target_compile_features(BigWham PUBLIC cxx_std_17)
target_link_libraries(BigWham PUBLIC InsideLoop)
set_target_properties(BigWham PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(BigWham PUBLIC src il)

add_library(BigWhamShared SHARED $<TARGET_OBJECTS:BigWham>)
add_library(BigWhamStatic STATIC $<TARGET_OBJECTS:BigWham>)
foreach(tgt BigWhamShared BigWhamStatic)
    target_link_libraries(${tgt} PUBLIC BigWham)
    set_target_properties(${tgt} PROPERTIES OUTPUT_NAME BigWham)
endforeach()

set(BIGWHAM_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src;${PROJECT_SOURCE_DIR}/il"
    CACHE PATH "BigWham include dirs")
mark_as_advanced(BIGWHAM_INCLUDE_DIRS)

add_executable(BigWham_exe  main.cpp)
target_link_libraries(BigWham_exe BigWhamShared)
include(BigWhamTools)

if(BIGWHAM_HDF5)
    enable_language(C)
    find_package(HDF5 REQUIRED COMPONENTS C)
    target_compile_definitions(BigWham PUBLIC BIGWHAM_HDF5)
    target_link_libraries(BigWham PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(BigWham PUBLIC ${HDF5_INCLUDE_DIRS})


    if(HDF5_IS_PARALLEL)
        find_package(MPI REQUIRED COMPONENTS C)
        target_compile_options(BigWham PUBLIC ${MPI_C_COMPILE_OPTIONS})
        target_link_libraries(BigWham PUBLIC ${MPI_C_LIBRARIES})
        target_include_directories(BigWham PUBLIC ${MPI_C_INCLUDE_DIRS})
    endif()
endif()

################################################################
# Pybind --->use with PYTHON version >=3.7
################################################################
message(STATUS "Building python interface: ${BIGWHAM_PYTHON_INTERFACE}")
if(BIGWHAM_PYTHON_INTERFACE)
    find_package (Python3 COMPONENTS Interpreter Development)
    message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python Version: ${Python3_VERSION}")
    message(STATUS "Python Include Dir: ${Python3_INCLUDE_DIRS}")
    set(PYBIND11_PYTHON_VERSION 3)
    add_external_package(pybind11 VERSION 2.8.0 CONFIG)
    add_subdirectory(interfaces/python)
endif()

################################################################################
# Mathematica interface
################################################################################
message(STATUS "Building the mathematica interface: ${BIGWHAM_MATHEMATICA_INTERFACE}")
if(BIGWHAM_MATHEMATICA_INTERFACE)
    add_external_package(ltemplate VERSION 0.5.4)
    add_subdirectory(interfaces/mathematica)
endif()

################################################################################
# For Unit Tests
################################################################################
if(BIGWHAM_TESTING)
    # set(IL_FOLDER "${CMAKE_SOURCE_DIR}")
    set(TESTED_SOURCE_FILES ${SOURCE_FILES})
    set(UNIT_TEST_FILES
        il/gtest/src/gtest-all.cc
        src/_test/penny_shaped_crack_analytical_sol.cpp
        src/core/_test/FaceData_tests.cpp
        src/elasticity/_test/elastic3DR0_element_tests.cpp
        # src/_test/elastic3DR0_element_benchmark.cpp
        src/_test/penny_shaped_crack_analytical_sol_benchmark.cpp
        src/core/_test/segment_test.cc
        src/elasticity/_test/segment0_sp3d_test.cc
        src/core/_test/bemesh_test.cc
        src/elasticity/_test/elastic_square_matrix_generator_test.cc
        src/core/_test/hierarchical_representation_test.cc
        src/hmat/hmatrix/_test/Hmat_test_gaussian_mat.cpp
        src/core/_test/triangle_test.cc
        src/core/_test/rectangle_test.cc
        src/_test/bigwhamio_test_2d.cc
        src/elasticity/_test/triangle0_test.cc
        src/elasticity/fullspace_iso_2d_segment/_test/plane_elasticity_segment_test.cc
        src/elasticity/_test/segment0_kernels_test.cc
      )

    add_executable(BigWhamElastUnitTest ${UNIT_TEST_FILES} test.cpp)
    target_link_libraries(BigWhamElastUnitTest PUBLIC BigWhamShared)

    target_include_directories(BigWhamElastUnitTest
        PUBLIC ${CMAKE_SOURCE_DIR}/il/gtest
        )

    # For unit tests: The precondition of our fonctions are checked with assert
    # macros that terminate the program in debug mode. In order to test those macros
    # in our unit tests, the flag IL_UNIT_TEST turns them into exceptions that can
    # be caught and checked.
    target_compile_definitions(BigWhamElastUnitTest
        PRIVATE IL_UNIT_TEST
                IL_FOLDER='\"${IL_FOLDER}\"')
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(BigWhamElastUnitTest PRIVATE -Wno-terminate -Wno-return-type)
    endif()
endif()

################################################################################
# For running examples
################################################################################
# add_library(bigwham_shared SHARED  ${SOURCE_FILES})
add_subdirectory(src/examples)


################################################################################
# For BigWham Documentation
################################################################################
# message(STATUS "Building documentation: ${BIGWHAM_DOCUMENTATION}")
# if(BIGWHAM_DOCUMENTATION)
#   add_subdirectory(docs)
# endif()

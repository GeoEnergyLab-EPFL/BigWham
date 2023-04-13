#####################################
# BIGWHAM                           #
# compilation configuration         #
#####################################
cmake_minimum_required(VERSION 3.7...3.19)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(BigWham VERSION 1.0
        DESCRIPTION "This is the library containing the wheels"
        LANGUAGES CXX)

# This is only for pybind11 to behave properly
set(CMAKE_CXX_STANDARD 17)

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
option(BIGWHAM_PROFILING "Add extra tool for profiling" OFF)
option(BIGWHAM_HDF5 "Use HDF5 for checkpoint/restart" OFF)


#####################################
# optimization options              #
#####################################
include(ExtraCompilationProfiles)

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
  set(IL_OPENMP OFF CACHE INTERNAL "" FORCE)
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
set(SOURCE_FILES
    src/hmat/cluster/cluster.h
    src/hmat/cluster/cluster.cpp
    # 2D source code
    src/core/Mesh2D.cpp src/core/Mesh2D.h
    src/core/SegmentData.h
    src/core/SquareMatrixGenerator.h
    src/core/BEMesh.h
    src/elasticity/2d/ElasticS3DP0_element.cpp
    src/elasticity/2d/Elastic2DP1_element.cpp
    src/elasticity/PostProcessDDM_2d.cpp
    src/elasticity/PostProcessDDM_3d.cpp

    # 3D source code
    src/core/Mesh3D.cpp src/core/Mesh3D.h
    src/core/FaceData.cpp
    #  - T6 element
    src/core/element_utilities.cpp
    src/elasticity/3d/Elastic3DT6_element.cpp
    src/elasticity/3d/h_potential_3DT6.cpp
    src/elasticity/3d/tensor_utilities_3DT6.cpp
    src/elasticity/3d/ElasticHMatrix3DT6.h
    #  - T0 element
    src/elasticity/3d/Elastic3DT0_element.cpp
    src/elasticity/3d/ElasticHMatrix3DT0.h
    src/elasticity/3d/ElasticHMatrix3DT0_modes2and3.h
    #  - R0 element
    src/elasticity/3d/Elastic3DR0_common.cpp
    src/elasticity/3d/Elastic3DR0_element.cpp
    src/elasticity/3d/Elastic3DR0_mode1Cartesian_element.cpp
    src/elasticity/3d/ElasticHMatrix3DR0.h
    src/elasticity/3d/ElasticHMatrix3DR0displ.h
    src/elasticity/3d/ElasticHMatrix3DR0_mode1Cartesian.h
    src/elasticity/3d/ElasticHMatrix3DR0_modes2and3Cartesian.h
    # BigWham Interface
    src/BigWham.h
    #new
    src/core/BIE_Kernel.h
    src/core/BEMesh.h
    src/core/BoundaryElement.h
    src/elasticity/BIE_elastostatic.h

    src/core/hierarchical_representation.h
    src/hmat/hmatrix/toHPattern.cpp
    src/core/elements/Segment.h
    src/core/elements/Triangle.h
    src/core/elements/Rectangle.h
    src/elasticity/2d/BIE_elastostatic_segment_0_impls.h
    src/elasticity/2d/BIE_elastostatic_segment_1_impls.h
    src/elasticity/3d/BIE_elastostatic_triangle_0_impls.h
    )

add_library(BigWham OBJECT ${SOURCE_FILES})
target_compile_features(BigWham PUBLIC cxx_std_17)
target_link_libraries(BigWham PUBLIC InsideLoop)
set_target_properties(BigWham PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(BigWham PUBLIC src src/hmat)

add_library(BigWhamShared SHARED $<TARGET_OBJECTS:BigWham>)
add_library(BigWhamStatic STATIC $<TARGET_OBJECTS:BigWham>)
foreach(tgt BigWhamShared BigWhamStatic)
    target_link_libraries(${tgt} PUBLIC BigWham)
    set_target_properties(${tgt} PROPERTIES OUTPUT_NAME BigWham)
endforeach()

set(BIGWHAM_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src;${PROJECT_SOURCE_DIR}/src/hmat"
    CACHE PATH "BigWham include dirs")
mark_as_advanced(BIGWHAM_INCLUDE_DIRS)

add_executable(BigWham_exe  main.cpp)
target_link_libraries(BigWham_exe BigWhamShared)

if (BIGWHAM_PROFILING)
    find_package(ITT REQUIRED)
    target_compile_definitions(BigWham PUBLIC HAS_ITT)
    target_link_libraries(BigWham PUBLIC ${ITT_LIBRARIES})
    target_include_directories(BigWham PUBLIC ${ITT_INCLUDE_DIRS})
endif()


if(BIGWHAM_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    target_compile_definitions(BigWham PUBLIC BIGWHAM_HDF5)
    target_link_libraries(BigWham PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(BigWham PUBLIC ${HDF5_INCLUDE_DIRS})
endif()

################################################################
# Pybind --->use with PYTHON version >=3.7
################################################################
include(BigWhamTools)

message(STATUS "Building python interface: ${BIGWHAM_PYTHON_INTERFACE}")
if(BIGWHAM_PYTHON_INTERFACE)
    set(PYBIND11_PYTHON_VERSION 3)
    add_external_package(pybind11 VERSION 2.2.4 CONFIG)
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
        # src/elasticity/_test/elasticity_tests.cpp
        src/_test/penny_shaped_crack_analytical_sol.h
        src/_test/penny_shaped_crack_analytical_sol.cpp
        src/core/_test/FaceData_tests.cpp
        src/elasticity/_test/elastic3DR0_element_tests.cpp
        #src/_test/elastic3DR0_element_benchmark.cpp
        src/_test/penny_shaped_crack_analytical_sol_benchmark.cpp
        src/core/_test/segment_test.cpp
        src/elasticity/BIE_elastostatic.h
        src/elasticity/_test/segment0_sp3d_tests.cpp
        src/core/_test/BEmesh_tests.cpp
        src/elasticity/_test/elastic_square_matrix_generator_tests.cpp
        src/core/_test/hierarchical_representation_tests.cpp
        src/hmat/hmatrix/_test/Hmat_test_gaussian_mat.cpp
        src/core/_test/triangle_tests.cpp
        src/core/_test/rectangle_tests.cpp
        src/_test/bigwhamio_tests_2d.cpp
        src/elasticity/_test/triangle0_tests.cpp)

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

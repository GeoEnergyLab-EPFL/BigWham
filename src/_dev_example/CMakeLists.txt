find_package(ZLIB REQUIRED)

option(BIGWHAM_OPENMP "OPENMP" ON) # MUST BE ON for openMP

add_library(cnpy SHARED cnpy.cpp)
target_include_directories(cnpy PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cnpy PRIVATE ${ZLIB_LIBRARIES})

add_executable(rep_test  rep_test.cpp)
target_link_libraries(rep_test PRIVATE BigWhamShared cnpy)

add_executable(rep_test_rowhmat  rep_test_rowhmat.cpp)
target_link_libraries(rep_test_rowhmat PRIVATE BigWhamShared cnpy)

add_executable(build_hmat build_hmat.cc)
target_link_libraries(build_hmat PRIVATE BigWhamShared cnpy)

add_executable(mat_vec mat_vec.cc)
target_link_libraries(mat_vec PRIVATE BigWhamShared cnpy)


configure_file(generate_penny_mesh.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(mesh_conn.npy ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(mesh_coords.npy ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(rep_test.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

find_package(ZLIB REQUIRED)
add_library(cnpy SHARED cnpy.cpp)
target_include_directories(cnpy PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cnpy PRIVATE ${ZLIB_LIBRARIES})
configure_file(generate_penny_mesh.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

add_executable(rep_test  rep_test.cpp)
target_link_libraries(rep_test PRIVATE BigWhamShared cnpy)

add_executable(build_hmat build_hmat.cpp)
target_link_libraries(build_hmat PRIVATE BigWhamShared cnpy)

add_executable(mat_vec mat_vec.cpp)
target_link_libraries(mat_vec PRIVATE BigWhamShared cnpy)

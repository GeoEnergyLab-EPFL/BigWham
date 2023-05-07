## All subdirectories editing SOURCE_FILES should named before
set(SOURCE_FILES ${SOURCE_FILES}
  ## 3D Tri 0 Element
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_3d/elastic_3dT0_element.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_3d/bie_elastostatic_triangle_0_impls.cc

  ## 3D Rect 0 Element
  ${CMAKE_CURRENT_SOURCE_DIR}/3d/Elastic3DR0_common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/3d/Elastic3DR0_element.cpp

  ## 2D Segment 0
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_2d_segment/bie_elastostatic_segment_0_impls.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_2d_segment/elastic_2dP0_segment.cc

  ## 2D Segment 1
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_2d_segment/bie_elastostatic_segment_1_impls.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_2d_segment/elastic_2dP1_segment.cc

  ## 3D Special Segment 0
   ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_sp3d_segment/elastic_sp3dP0_element.cc
  
  ## AxisSymm Ring 
  ${CMAKE_CURRENT_SOURCE_DIR}/fullspace_iso_axisymm_flat_ring_unidirectional/elliptic_integral.cc


  PARENT_SCOPE)

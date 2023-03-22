//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 22.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <iostream>

#include <gtest/gtest.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/math.h>

#include "elements/segment.h"
/* -------------------------------------------------------------------------- */

TEST(Segment, test_n_vert) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1.0;
  bie::Segment<0> seg;
  seg.set_element(xy);
  int n_vert = seg.get_number_vertices();
  ASSERT_TRUE(n_vert == 2);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_centroid) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1.0;

  bie::Segment<0> seg;
  seg.set_element(xy);

  auto center = seg.get_centroid();
  std::cout << "center " << center[0] << "\n";
  ASSERT_TRUE(center[0] == 0.5);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_size_1) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 6.40;

  bie::Segment<0> seg;
  seg.set_element(xy);
  double size = seg.get_size();

  ASSERT_TRUE(size == 6.40);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_n_s_1) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 6.40;
  bie::Segment<0> seg;
  seg.set_element(xy);
  auto normal = seg.get_normal();
  auto tang = seg.get_tangent_1();
  ASSERT_TRUE(normal[0] == 0.0 && normal[1] == 1.0 && tang[0] == 1. &&
              tang[1] == 0.);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_size_2) { // 45 deg element
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1;
  xy(1, 1) = 1;
  bie::Segment<0> seg;
  seg.set_element(xy);
  double size = seg.get_size();

  ASSERT_TRUE(size == sqrt(2.0));
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_n_s_2) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1;
  xy(1, 1) = 1;
  bie::Segment<0> seg;
  seg.set_element(xy);
  auto normal = seg.get_normal();
  auto tang = seg.get_tangent_1();
  ASSERT_TRUE(normal[0] == -1. / sqrt(2) && normal[1] == 1. / sqrt(2) &&
              tang[0] == 1. / sqrt(2) && tang[1] == 1. / sqrt(2)); //
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_rotation_a) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1;
  xy(1, 1) = 1;
  bie::Segment<0> seg;
  seg.set_element(xy);
  auto normal = seg.get_normal();
  auto tang = seg.get_tangent_1();
  auto rot = seg.get_rotation_matrix_T();

  // std::cout << rot(0, 0) << " " << rot(0, 1) << " " << rot(1, 0) << " "
  //           << rot(1, 1) << "\n";

  ASSERT_TRUE(rot(0, 0) == 1. / sqrt(2) && rot(1, 0) == 1. / sqrt(2) &&
              rot(0, 1) == -1. / sqrt(2) && rot(1, 1) == 1. / sqrt(2)); //
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_rotation_b) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1;
  xy(1, 1) = 1;
  bie::Segment<0> seg;
  seg.set_element(xy);
  auto tang = seg.get_tangent_1();
  auto rot = seg.get_rotation_matrix_T();
  double theta = std::atan2(tang[1], tang[0]);

  ASSERT_TRUE((abs(rot(0, 0) - std::cos(theta)) < 1.e-12) &&
              (abs(rot(0, 1) + std::sin(theta)) < 1.e-12) &&
              (abs(rot(1, 0) - std::sin(theta)) < 1.e-12) &&
              (abs(rot(1, 1) - std::cos(theta)) < 1.e-12)); //
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_rotation_c) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 1;
  xy(1, 1) = 1;
  bie::Segment<0> seg;
  seg.set_element(xy);
  auto tang = seg.get_tangent_1();
  auto rot = seg.get_rotation_matrix();
  double theta = std::atan2(tang[1], tang[0]);

  // std::cout << rot(0, 0) << " " << rot(0, 1) << " " << rot(1, 0) << " "
  //           << rot(1, 1) << "\n";
  ASSERT_TRUE((abs(rot(0, 0) - std::cos(theta)) < 1.e-12) &&
              (abs(rot(0, 1) - std::sin(theta)) < 1.e-12) &&
              (abs(rot(1, 0) + std::sin(theta)) < 1.e-12) &&
              (abs(rot(1, 1) - std::cos(theta)) < 1.e-12)); //
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_collocation0_1) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 4.;
  bie::Segment<0> seg;
  seg.set_element(xy);
  double size = seg.get_size();
  auto mycol = seg.get_collocation_points();
  // std::cout << mycol(0, 0) << "-" << mycol(0, 1) << "\n";
  ASSERT_TRUE(mycol(0, 0) == 2. && mycol(0, 1) == 0.);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_collocation0_2) {
  il::Array2D<double> xy{2, 2, 5.};
  xy(1, 0) = 5. + 4.;
  bie::Segment<0> seg;
  seg.set_element(xy);
  double size = seg.get_size();
  auto mycol = seg.get_collocation_points();
  // std::cout << mycol(0, 0) << "-" << mycol(0, 1) << "\n";
  ASSERT_TRUE(mycol(0, 0) = 2. + 5. && mycol(0, 1) == 5.);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_collocation1_1) {
  il::Array2D<double> xy{2, 2, 0.};
  xy(1, 0) = 2.;
  bie::Segment<1> seg;
  seg.set_element(xy);
  double size = seg.get_size();
  auto mycol = seg.get_collocation_points();
  // std::cout << mycol(0, 0) << " " << mycol(0, 1) << " " << mycol(1, 0) << " "
  //           << mycol(1, 1) << "\n";
  ASSERT_TRUE(abs(mycol(0, 0) - (1. - 1 / sqrt(2))) < 1.e-12 &&
              mycol(0, 1) == 0. && mycol(1, 1) == 0. &&
              abs(mycol(1, 0) - (1. + 1 / sqrt(2))) < 1.e-12); //
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_dim_0) {
  il::Array2D<double> xy{2, 2, 5.};
  xy(1, 0) = 5. + 4.;
  bie::Segment<0> seg;
  seg.set_element(xy);
  int mydim = seg.get_spatial_dimension();
  ASSERT_TRUE(mydim == 2);
}
/* -------------------------------------------------------------------------- */

TEST(Segment, test_dim_1) {
  il::Array2D<double> xy{2, 2, 5.};
  xy(1, 0) = 5. + 4.;
  bie::Segment<1> seg;
  seg.set_element(xy);
  int mydim = seg.get_spatial_dimension();
  ASSERT_TRUE(mydim == 2);
}
/* -------------------------------------------------------------------------- */

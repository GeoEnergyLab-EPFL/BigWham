//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 31.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
#include <gtest/gtest.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/math.h>

#include "hmat/hmatrix/Hmat.h"
#include "bigwham_io.h"
#include "io/bigwham_io_gen.h"

#include "core/be_mesh.h"
#include "elements/boundary_element.h"
#include "hmat/square_matrix_generator.h"
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_2d, Sp3D0_1_1) {
  // create a simple mesh for a griffith crack -
  // use the bigwhamio interface.
  ///  simple mesh
  int n_elts = 1200;
  std::vector<double> coor(2 * (n_elts + 1), 0.);
  double L = 1.;
  double h = 2. * L / n_elts;
  int k = 0;
  for (int i = 0; i < n_elts + 1; i++) {
    coor[k] = i * h - L;
    k = k + 2;
  }
  std::vector<int> conn(n_elts * 2, 0.);
  k = 0;
  for (int i = 0; i < n_elts; i++) {
    conn[k] = i;
    conn[k + 1] = i + 1;
    k = k + 2;
  }

  Bigwhamio my_io;
  std::vector<double> properties{1., 0., 100};
  my_io.set(coor, conn, "S3DP0", properties, 32, 2, 1.e-3);

  ASSERT_TRUE(abs(my_io.getCompressionRatio() - 0.12664) < 1e-4);
}
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_2d, Sp3D0_1_2) {
  // create a simple mesh for a griffith crack -
  // use the bigwhamio interface.
  ///  simple mesh
  int n_elts = 1200;
  std::vector<double> coor(2 * (n_elts + 1), 0.);
  double L = 1.;
  double h = 2. * L / n_elts;
  int k = 0;
  for (int i = 0; i < n_elts + 1; i++) {
    coor[k] = i * h - L;
    k = k + 2;
  }
  std::vector<int> conn(n_elts * 2, 0.);
  k = 0;
  for (int i = 0; i < n_elts; i++) {
    conn[k] = i;
    conn[k + 1] = i + 1;
    k = k + 2;
  }

  Bigwhamio my_io;
  std::vector<double> properties{1., 0., 100};
  my_io.set(coor, conn, "S3DP0", properties, 32, 2, 1.e-3);
  ASSERT_TRUE(my_io.getProblemDimension() == 2 &&
              my_io.getSpatialDimension() == 2); // h_.isBuilt()
}
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_2d, Sp3D0_1_3) {
  // create a simple mesh for a griffith crack -
  // use the bigwhamio interface.
  ///  simple mesh
  // check diag
  int n_elts = 1200;
  std::vector<double> coor(2 * (n_elts + 1), 0.);
  double L = 1.;
  double h = 2. * L / n_elts;
  int k = 0;
  for (int i = 0; i < n_elts + 1; i++) {
    coor[k] = i * h - L;
    k = k + 2;
  }
  std::vector<int> conn(n_elts * 2, 0.);
  k = 0;
  for (int i = 0; i < n_elts; i++) {
    conn[k] = i;
    conn[k + 1] = i + 1;
    k = k + 2;
  }

  Bigwhamio my_io;
  std::vector<double> properties{1., 0., 100};
  my_io.set(coor, conn, "S3DP0", properties, 32, 2, 1.e-3);

  std::vector<double> x(my_io.matrixSize(1), 0.);
  for (il::int_t i = 0; i < n_elts; i++) {
    x[2 * i + 1] = 4.0 * sqrt(L * L - coor[2 * i] * coor[2 * i]);
  }
  auto y = my_io.matvect(x);
  std::vector<double> the_diag(n_elts * 2, 0.);
  my_io.getDiagonal(the_diag);

  il::Array<double> rel_err{n_elts, 0.};
  for (il::int_t i = 0; i < n_elts; i++) {
    rel_err[i] =
        sqrt((the_diag[2 * i + 1] - 190.985) * (the_diag[2 * i + 1] - 190.985));
  }
  std::cout << "Mean rel error (using biwghamio) " << il::mean(rel_err) << "\n";
  ASSERT_TRUE(il::mean(rel_err) < 0.05); // h_.isBuilt()
}
/* -------------------------------------------------------------------------- */

// 2DP0
TEST(bigwham_io_2d, 2DP0_1) {
  // create a simple mesh for a griffith crack -
  // use the bigwhamio interface.
  ///  simple mesh
  // check diag
  int n_elts = 1200;
  std::vector<double> coor(2 * (n_elts + 1), 0.);
  double L = 1.;
  double h = 2. * L / n_elts;
  int k = 0;
  for (int i = 0; i < n_elts + 1; i++) {
    coor[k] = i * h - L;
    k = k + 2;
  }
  std::vector<int> conn(n_elts * 2, 0.);
  k = 0;
  for (int i = 0; i < n_elts; i++) {
    conn[k] = i;
    conn[k + 1] = i + 1;
    k = k + 2;
  }

  Bigwhamio my_io;
  std::vector<double> properties{1., 0.};
  my_io.set(coor, conn, "2DP0", properties, 32, 2, 1.e-3);

  std::vector<double> x(my_io.matrixSize(1), 0.);
  for (il::int_t i = 0; i < n_elts; i++) {
    x[2 * i + 1] = 4.0 * sqrt(L * L - coor[2 * i] * coor[2 * i]);
  }
  auto y = my_io.matvect(x);
  std::vector<double> the_diag(n_elts * 2, 0.);
  my_io.getDiagonal(the_diag);

  il::Array<double> rel_err{n_elts, 0.};
  for (il::int_t i = 0; i < n_elts; i++) {
    rel_err[i] =
        sqrt((the_diag[2 * i + 1] - 190.985) * (the_diag[2 * i + 1] - 190.985));
  }
  std::cout << "Mean rel error (using biwghamio) " << il::mean(rel_err) << "\n";
  ASSERT_TRUE(il::mean(rel_err) < 0.05); // h_.isBuilt()
}
/* -------------------------------------------------------------------------- */

// 2DP0 Bigwhamio General
TEST(bigwham_io_gen_2d, 2DP0_1) {
  // create a simple mesh for a griffith crack -
  // use the bigwhamio interface.
  ///  simple mesh
  // check diag
  int n_elts = 1200;
  std::vector<double> coor(2 * (n_elts + 1), 0.);
  double L = 1.;
  double h = 2. * L / n_elts;
  int k = 0;
  for (int i = 0; i < n_elts + 1; i++) {
    coor[k] = i * h - L;
    k = k + 2;
  }
  std::vector<int> conn(n_elts * 2, 0.);
  k = 0;
  for (int i = 0; i < n_elts; i++) {
    conn[k] = i;
    conn[k + 1] = i + 1;
    k = k + 2;
  }

  BigWhamIOGen my_io;
  std::vector<double> properties{1., 0.};
  my_io.Set(coor, conn, "2DP0", properties, 32, 2, 1.e-3);

  std::vector<double> x(my_io.MatrixSize(1), 0.);
  for (il::int_t i = 0; i < n_elts; i++) {
    x[2 * i + 1] = 4.0 * sqrt(L * L - coor[2 * i] * coor[2 * i]);
  }
  auto y = my_io.MatVec(x);
  std::vector<double> the_diag(n_elts * 2, 0.);
  my_io.GetDiagonal(the_diag);

  il::Array<double> rel_err{n_elts, 0.};
  for (il::int_t i = 0; i < n_elts; i++) {
    rel_err[i] =
        sqrt((the_diag[2 * i + 1] - 190.985) * (the_diag[2 * i + 1] - 190.985));
  }
  std::cout << "Mean rel error (using biwghamio) " << il::mean(rel_err) << "\n";
  ASSERT_TRUE(il::mean(rel_err) < 0.05); // h_.isBuilt()
}
/* -------------------------------------------------------------------------- */

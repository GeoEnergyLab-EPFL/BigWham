//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 31.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2025.  All rights reserved. See the LICENSE
// file for more details.
//
#include <gtest/gtest.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/math.h>

#include "hmat/hmatrix/hmat.h"
#include "io/bigwham_io.h"

#include "core/be_mesh.h"
#include "elements/boundary_element.h"
#include "hmat/square_matrix_generator.h"
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_gen_2d, Sp3S0_1_1) {
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

  std::vector<double> properties{1., 0., 100};
  BigWhamIO my_io{coor, conn, "S3DS0-H", properties};
  my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);

  ASSERT_TRUE(abs(my_io.GetCompressionRatio() - 0.12664) < 1e-4);
}
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_gen_2d, Sp3DS0_1_2) {
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

  std::vector<double> properties{1., 0., 100};
  BigWhamIO my_io{coor, conn, "S3DS0-H", properties};

  my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);
  ASSERT_TRUE(my_io.dof_dimension() == 2 &&
              my_io.spatial_dimension() == 2); // h_.isBuilt()
}
/* -------------------------------------------------------------------------- */

TEST(bigwham_io_gen_2d, Sp3DS0_1_3) {
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

  std::vector<double> properties{1., 0., 100};
  BigWhamIO my_io{coor, conn, "S3DS0-H", properties};
  my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);

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

// 2DP0
TEST(bigwham_io_gen_2d, 2DS0_1) {
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

  std::vector<double> properties{1., 0.};
  BigWhamIO my_io{coor, conn, "2DS0-H", properties};
  my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);

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

// 2DP0
TEST(bigwham_io_gen_2d, 2DS0_2) {
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

    std::vector<double> properties{1., 0.};
    BigWhamIO my_io{coor, conn, "2DS0-H", properties};
    my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);

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

TEST(bigwham_io_gen_2d, 2DS0_3) {
    // create a simple mesh for a griffith crack -
    // use the bigwhamio interface.
    ///  simple mesh
    // check displacement
    int n_elts = 4;
    std::vector<double> coor(2 * (n_elts + 1), 0.);
    double L = 1.;
    double h = 2. * L / n_elts;
    int k = 0;
    for (il::int_t i = 0; i < n_elts + 1; i++) {
        coor[k] = i * h - L;
        k = k + 2;
    }
    std::vector<int> conn(n_elts * 2, 0.);
    k = 0;
    for (il::int_t  i = 0; i < n_elts; i++) {
        conn[k] = i;
        conn[k + 1] = i + 1;
        k = k + 2;
    }

    std::vector<double> properties{1., 0.};
    BigWhamIO my_io{coor, conn, "2DS0-H", properties};
    my_io.BuildHierarchicalMatrix(32, 2, 1.e-3);

    il::Array<double> x(my_io.MatrixSize(1),0.);
    for(il::int_t i=0;i<n_elts ;i++){
        x[2*i+1]=4.0*sqrt(L*L-0.25*(coor[2*i]+coor[2*(i+1)])*(coor[2*i]+coor[2*(i+1)]));
    }
    // obs mesh ....
    int n_obs=3;
    std::vector<double> obs_coor(2*n_obs,0.);
    k=0;
    for (il::int_t  i=0;i<n_obs;i++) {
        obs_coor[k+1]=i*h+h; //  x=0, y-axis
        k=k+2;
    }
    for (il::int_t  i=0;i<n_obs;i++){
        std::cout << "x " << obs_coor[2*i] << " y " << obs_coor[2*i+1] << "\n";
    }
    il::Array<double> test_disp= my_io.ComputeDisplacements(obs_coor, x.view());

    bool tt = true;
    for (il::int_t  i=0;i<n_obs;i++){
        std::cout << "u_x " << test_disp[i*2+0] <<"\n";
        tt =tt && (il::abs(test_disp[i*2+0])<1.e-12) ;
        std::cout << "u_y " << test_disp[i*2+1] <<"\n";
    }

    il::Array<double> test_stress= my_io.ComputeStresses(obs_coor, x.view());
    for (il::int_t  i=0;i<n_obs;i++){
        std::cout << "s_xx " << test_stress[i*3] <<"\n";
        std::cout << "s_yy " << test_stress[i*3+1] <<"\n";
        std::cout << "s_xy " << test_stress[i*3+2] <<"\n";
        tt =tt && (il::abs(test_stress[i*3+2])<1.e-12) ;
    }

    ASSERT_TRUE(tt); // h_.isBuilt()
}
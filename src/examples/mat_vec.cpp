///=============================================================================
///
/// \file        rep_test.cpp
///
/// \author      Ankit
///
/// \copyright   Copyright (©) 2018 EPFL (Ecole Polytechnique Fédérale
///              de Lausanne)\n
///              Geo-Energy lab
///
/// \brief       Test for Penny shape crack for profiling bigwham
///
///=============================================================================

/* -------------------------------------------------------------------------- */
#include "cnpy.h"
#include "npy_tools.h"
/* -------------------------------------------------------------------------- */
#include "core/BEMesh.h"
#include "core/ElasticProperties.h"
#include "core/SquareMatrixGenerator.h"
#include "core/elements/Triangle.h"
#include "core/hierarchical_representation.h"
#include "elasticity/3d/BIE_elastostatic_triangle_0_impls.h"
#include "elasticity/BIE_elastostatic.h"
#include "hmat/hmatrix/Hmat.h"
/* -------------------------------------------------------------------------- */
#include <algorithm>
#include <cmath>
#include <il/Array2D.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <omp.h>
/* -------------------------------------------------------------------------- */

using Real1D = il::Array<double>;
using Real2D = il::Array2D<double>;
using Int1D = il::Array<il::int_t>;
using Int2D = il::Array2D<il::int_t>;
using Tri0 = bie::Triangle<0>;
using Mesh = bie::BEMesh<Tri0>;
using MatProp = bie::ElasticProperties;
using KernelH = bie::BIE_elastostatic<Tri0, Tri0, bie::ElasticKernelType::H>;
using MatixGenerator = bie::SquareMatrixGenerator<double, Tri0, KernelH>;

int main(int argc, char *argv[]) {

  std::string f_coord = "mesh_coords.npy";
  std::string f_conn = "mesh_conn.npy";

  auto coord_npy = cnpy::npy_load(f_coord);
  auto conn_npy = cnpy::npy_load(f_conn);

  // Penby shape geometery
  uint dim = 3;
  double radius = 1.0;
  double pressure = 1.0;

  // Elastic properties
  double G = 1.0;
  double nu = 0.25;
  double E = (2 * G) * (1 + nu);
  double pi = M_PI;

  // H-mat parameters
  il::int_t max_leaf_size = 16;
  double eta = 3.0;
  double eps_aca = 1.e-4;

  auto && coord = copy_array2D<double>(coord_npy);
  auto && conn = copy_array2D<il::int_t>(conn_npy);

  Mesh my_mesh(coord, conn);
  Real2D xcol = my_mesh.getCollocationPoints();
  MatProp elas(E, nu);
  KernelH ker(elas, coord.size(1));
  Real1D prop{1, 1000.};
  ker.setKernelProperties(prop);

  std::cout << "Number of Collocation points  = " << xcol.size(0) << " X "
            << xcol.size(1) << std::endl;

  bie::HRepresentation hr =
    bie::h_representation_square_matrix(my_mesh, max_leaf_size, eta);

  MatixGenerator M(my_mesh, ker, hr.permutation_0_);

  bie::Hmat<double> h_("hmat.h5");

  Real1D dd{M.size(1), 0.0};
  Real1D dd_perm{M.size(1), 0.0};
  Real1D trac{M.size(0), 0.0};
  Real1D trac_perm{M.size(0), 0.0};

  double pre_fac = (8 * (1 - nu * nu)) / (pi * E);

  double rsq;
  for (il::int_t i = 0; i < M.sizeAsBlocks(1); i++) {
    rsq = xcol(i, 0) * xcol(i, 0) + xcol(i, 1) * xcol(i, 1);
    dd[dim * i + 2] = pre_fac * std::sqrt(radius * radius - rsq);
  }

  for (il::int_t i = 0; i < M.sizeAsBlocks(1); i++) {
    il::int_t j = hr.permutation_1_[i];
    for (uint d = 0; d < dim; d++) {
      dd_perm[dim * i + d] = dd[dim * j + d];
    }
  }

  double y0 = 0.;
  auto start = omp_get_wtime();
  //for (il::int_t i = 0; i < 1000; ++i) {
  trac_perm = h_.matvec(dd_perm);
  y0 += trac_perm[0];
  //}
  auto end = omp_get_wtime();
  std::cout << "Hmat matvec: " << (end - start) / 1000 << "s - y0: " << y0 << std::endl;

  for (il::int_t i = 0; i < M.sizeAsBlocks(0); i++) {
    il::int_t j = hr.permutation_0_[i];
    for (uint d = 0; d < dim; d++) {
      trac[dim * j + d] = trac_perm[dim * i + d];
    }
  }

  Real1D rel_err{M.sizeAsBlocks(0), 0.};
  for (il::int_t i = 0; i < M.sizeAsBlocks(0); i++) {
    rel_err[i] += trac[dim * i + 0] * trac[dim * i + 0];
    rel_err[i] += trac[dim * i + 1] * trac[dim * i + 1];
    rel_err[i] +=
        (trac[dim * i + 2] - pressure) * (trac[dim * i + 2] - pressure);
    rel_err[i] = std::sqrt(rel_err[i]);
  }

  std::cout << "Mean rel error " << il::mean(rel_err) << std::endl;

  return 0;
}

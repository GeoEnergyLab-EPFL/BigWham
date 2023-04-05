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
using Tri0 = bie::Triangle<0>;
using Mesh = bie::BEMesh<Tri0>;
using MatProp = bie::ElasticProperties;
using KernelH = bie::BIE_elastostatic<Tri0, Tri0, bie::ElasticKernelType::H>;
using MatixGenerator = bie::SquareMatrixGenerator<double, Tri0, KernelH>;

int main(int argc, char *argv[]) {

  std::string f_coord{"mesh_coords.npy"};
  std::string f_conn{"mesh_conn.npy"};

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
  MatProp elas(E, nu);
  KernelH ker(elas, coord.size(1));
  il::Array<double> prop{1, 1000.};
  auto xcol = my_mesh.getCollocationPoints();
  ker.setKernelProperties(prop);

  std::cout << "Number of Collocation points  = " << xcol.size(0) << " X "
            << xcol.size(1) << std::endl;

  bie::HRepresentation hr =
    bie::h_representation_square_matrix(my_mesh, max_leaf_size, eta);

  MatixGenerator M(my_mesh, ker, hr.permutation_0_);

  bie::Hmat<double> h_(M, hr, eps_aca);
  h_.writeToFile("hmat.h5");

  return 0;
}

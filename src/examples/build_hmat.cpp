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
#include <algorithm>
#include <cmath>
#include <il/Array2D.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
#include "core/be_mesh.h"
#include "core/elastic_properties.h"
#include "elements/triangle.h"
#include "hmat/hierarchical_representation.h"

#include "elasticity/bie_elastostatic.h"
#include "hmat/hmatrix/Hmat.h"
#include "hmat/square_matrix_generator.h"

/* -------------------------------------------------------------------------- */
#include "cnpy.h"
#include "npy_tools.h"

/* -------------------------------------------------------------------------- */
using namespace bie;

/* -------------------------------------------------------------------------- */
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

  auto &&coord = copy_array2D<double>(coord_npy);
  auto &&conn = copy_array2D<il::int_t>(conn_npy);

  auto my_mesh = std::make_shared<BEMesh<Triangle<0>>>(coord, conn);
  my_mesh->ConstructMesh();
  il::Array2D<double> xcol = my_mesh->collocation_points();
  std::cout << "Number of Collocation points  = " << xcol.size(0) << " X "
            << xcol.size(1) << std::endl;

  ElasticProperties elas(E, nu);
  auto ker = std::make_shared<
      BieElastostatic<Triangle<0>, Triangle<0>, ElasticKernelType::H>>(
      elas, coord.size(1));


  auto hr = HRepresentationSquareMatrix(my_mesh, max_leaf_size, eta);

  SquareMatrixGenerator<double> M(my_mesh, ker, hr);
  Hmat<double> hmat(M, eps_aca);
  hmat.writeToFile("hmat.h5");

  return 0;
}

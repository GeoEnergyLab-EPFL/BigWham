//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 01.02.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <il/blas.h>
#include <il/StaticArray.h>
#include <il/StaticArray2D.h>

#include "elements/triangle.h"
#include "elasticity/bie_elastostatic.h"

#include "elastic_3dT0_element.h"


namespace bie {

template <>
std::vector<double>
BieElastostatic<Triangle<0>, Triangle<0>, ElasticKernelType::H>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {
  //  return tractions - Hypersingular elastic kernel - Triangle 0 element
  // source_elt : element object of the source element
  // i_s : integert for the source collocation number (0,1)
  // receiver_elt  : element object of the receiver element
  // i_r : integer for the collocation number where to compute the normal and
  // shear stress in the receiver element outputs: column-major (fortran order)
  // vector for the displacement to tractions influence matrix

  // get constitutive parameters
  double G = this->elas_.shear_modulus();
  double nu = this->elas_.poisson_ratio();

  // get coordinates receiver cp
  auto el_cp_r = receiver_elt.collocation_points();

  // randomly perturb collocation point
  il::StaticArray<double, 3> receiver_coor{0};
  for (int i = 0; i < 3; i++) {
    receiver_coor[i] =
        el_cp_r(i_r, i) + sqrt(std::numeric_limits<double>::epsilon());
  }

  // get coordinates vertices of triangular source element
  auto el_vertices_s = source_elt.vertices();

  il::StaticArray2D<double, 3, 3> el_vertices_s_static{0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      el_vertices_s_static(i, j) = el_vertices_s(i, j);
    }
  }

  // get stress influence coefficients - in the local coordinate system of the
  // source element
  auto stress = StressesKernelT0(receiver_coor, el_vertices_s_static, G, nu);

  // normal vector at the receiver location in the reference system of the
  // source element
  auto nr = source_elt.ConvertToLocal(receiver_elt.normal());

  // compute traction vectors at receiver element cp due to (DD1,DD2,DD3) source
  // element in the reference system of the source element
  il::Array2D<double> DDs_to_traction_local{3, 3, 0.0}; // traction vectors
  il::Array<double> traction_temp{3, 0.0};   // temporary traction vector
  il::Array2D<double> sigma_temp{3, 3, 0.0}; // temporary stress tensor
  for (int i = 0; i < 3; ++i) { // loop over the rows of Stress, i.e., over each
                                // DD component effect
    // definition of temporary stress tensor
    // i = 0 : D1
    // i = 1 : D2
    // i = 2 : D3

    sigma_temp(0, 0) = stress(i, 0); // S11
    sigma_temp(0, 1) = stress(i, 3); // S12
    sigma_temp(0, 2) = stress(i, 4); // S13
    sigma_temp(1, 0) = stress(i, 3); // S21
    sigma_temp(1, 1) = stress(i, 1); // S22
    sigma_temp(1, 2) = stress(i, 5); // S23
    sigma_temp(2, 0) = stress(i, 4); // S31
    sigma_temp(2, 1) = stress(i, 5); // S32
    sigma_temp(2, 2) = stress(i, 2); // S33

    // compute temporary traction vector
    traction_temp = il::dot(sigma_temp, nr);
    // convert to global from local src element coordinate system
    auto traction_temp_global = source_elt.ConvertToGlobal(traction_temp);
    // convert to local rec element coordinate system
    auto traction_temp_local_rec =
        receiver_elt.ConvertToLocal(traction_temp_global);

    for (int j = 0; j < 3; ++j) {
      // fill the traction vectors at receiver element cp due to (DD1,DD2,DD3)
      // source element
      DDs_to_traction_local(j, i) = traction_temp_local_rec[j];
      // | t1/D1   t1/D2  t1/D3 |
      // | t2/D1   t2/D2  t2/D3 |
      // | t3/D1   t3/D2  t3/D3 |
      // local DD & local traction
      // both in the reference system of the source element
    }
  }

  // column major format
  std::vector<double> stnl(9, 0.);
  int k = 0;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      stnl[k] = DDs_to_traction_local(i, j);
      k++;
    }
  }
  return stnl;
}

template class BieElastostatic<Triangle<0>, Triangle<0>, ElasticKernelType::H>;
} // namespace bie

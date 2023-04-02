//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 17.03.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_BIEELASTOSTATICSP3D_H
#define BIGWHAM_BIEELASTOSTATICSP3D_H

#include "core/elastic_properties.h"
#include "elasticity/bie_elastostatic.h"
#include "elements/segment.h"

#include "elastic_sp3dP0_element.h"
#include "il/container/1d/Array.h"

namespace bie {

// a dummy derived class for simplified 3D P0 kernel....
template <class Es, class Er, ElasticKernelType k>
class BieElastostaticSp3d : public BieElastostatic<Es, Er, k> {
  using BieElastostatic<Es, Er, k>::BieElastostatic;

public:
  BieElastostaticSp3d() : BieElastostatic<Es, Er, k>(){};

  BieElastostaticSp3d(bie::ElasticProperties &elas, il::int_t dim)
      : BieElastostatic<Es, Er, k>() {
    IL_EXPECT_FAST(dim == 2);
    this->elas_ = elas;
    this->dof_dimension_ = dim;
    this->spatial_dimension_ = dim;
    this->kernel_properties_ = il::Array<double>(1, 10000.0);
  };

  BieElastostaticSp3d(bie::ElasticProperties &elas, il::int_t dim,
                      bool local_unknowns, bool local_co_variables)
      : BieElastostatic<Es, Er, k>() {
    IL_EXPECT_FAST(dim == 2);
    this->elas_ = elas;
    this->dof_dimension_ = dim;
    this->spatial_dimension_ = dim;
    this->kernel_properties_ = il::Array<double>(1, 10000.0);
    this->local_unknowns_ = local_unknowns;
    this->local_co_variables_ = local_co_variables;
  };

  std::vector<double> influence(const BoundaryElement &, il::int_t,
                                const BoundaryElement &,
                                il::int_t) const override;
};

//////////////////////////////////////////////////////////////////////////////////////////////
template <>
inline std::vector<double>
BieElastostaticSp3d<Segment<0>, Segment<0>, ElasticKernelType::H>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {
  //  return tractions - Hypersingular elastic kernel - Segment 0 element - note
  //  use of simplified3D kernel here
  // source_elt : element object of the source element
  // i_s : integert for the source collocation number (0,1)
  // receiver_elt  : element object of the receiver element
  // i_r : integer for the collocation number where to compute the normal and
  // shear stress in the receiver element outputs: column-major (fortran order)
  // vector for the displacement to traction influence matrix

  // switch to the frame of the source element....

  il::Array<double> xe{2, 0.};
  auto Xmid = source_elt.centroid();
  auto r_col = receiver_elt.collocation_points();
  for (int i = 0; i < 2; ++i) {
    xe[i] = r_col(i_r, i) - Xmid[i];
  }
  xe = source_elt.ConvertToLocal(xe.view());


  double h = source_elt.size();
  double ker_options = this->kernel_properties_[0];

  // std::cout << "before"
  //           << "\n";
  il::StaticArray2D<double, 2, 3> stress_l = stresses_kernel_s3d_p0_dd(
      h / 2., ker_options / 2., this->elas_.shear_modulus(),
      this->elas_.poisson_ratio(), xe[0], xe[1]);
  // std::cout << "here"
  //           << "\n";

  auto n = source_elt.ConvertToLocal(receiver_elt.normal().view());
  auto s = source_elt.ConvertToLocal(receiver_elt.tangent1().view());

  double n1n1 = n[0] * n[0];
  double n2n2 = n[1] * n[1];
  double n1n2 = n[0] * n[1];
  double n1s1 = n[0] * s[0];
  double n2s2 = n[1] * s[1];

  double n1s2pn2s1 = n[0] * s[1] + n[1] * s[0];

  std::vector<double> stnl(4, 0.);

  // shear stress
  //  effect of shear dd
  //        St(0, 0)
  stnl[0] = n1s1 * stress_l(0, 0) + n1s2pn2s1 * stress_l(0, 1) +
            n2s2 * stress_l(0, 2);
  //  effect of normal dd
  // St(0, 1)
  stnl[2] = n1s1 * stress_l(1, 0) + n1s2pn2s1 * stress_l(1, 1) +
            n2s2 * stress_l(1, 2);

  // normal stress
  //  effect of shear dd
  // St(1, 0)
  stnl[1] = n1n1 * stress_l(0, 0) + 2. * n1n2 * stress_l(0, 1) +
            n2n2 * stress_l(0, 2);
  //  effect of normal dd
  // St(1, 1)
  stnl[3] = n1n1 * stress_l(1, 0) + 2. * n1n2 * stress_l(1, 1) +
            n2n2 * stress_l(1, 2);

  return stnl;
}

} // namespace bie

#endif // BIGWHAM_BIEELASTOSTATICSP3D_H

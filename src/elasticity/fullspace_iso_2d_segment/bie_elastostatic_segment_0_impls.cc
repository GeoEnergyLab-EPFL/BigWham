//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 23.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <il/StaticArray.h>
#include <tuple>

#include "elasticity/bie_elastostatic.h"
#include "elements/boundary_element.h"
#include "elements/segment.h"
#include "elements/point.h"

#include "elastic_2dP0_segment.h"

namespace bie {
/* -------------------------------------------------------------------------- */

//  U kernel - pt force displacement
template <>
std::vector<double>
BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::U>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {
  //  return  - U elastic kernel - Segment 0 element - note use of simplified3D
  //  kernel here
  // source_elt : element object of the source element
  // i_s : integert for the source collocation number (0,1)
  // receiver_elt  : element object of the receiver element
  // i_r : integer for the collocation number where to compute the normal and
  // shear stress in the receiver element outputs: column-major (fortran order)
  // vector for the displacement to traction influence matrix

  // switch to the frame of the source element....
  // il::StaticArray2D<double, 2, 2> Rt = source_elt.rotationMatrix_T();; //
  // transpose rotation matrix
  il::Array<double> xe{2, 0.0};
  auto Xmid = source_elt.centroid();
  auto r_col = receiver_elt.collocation_points();
  for (int i = 0; i < 2; ++i) {
    xe[i] = r_col(i_r, i) - Xmid[i];
  }
  auto xe_local = source_elt.ConvertToLocal(xe);

  double h = source_elt.size();

  auto uik =
      Ue_segment_0(h, this->elas_.shear_modulus(), this->elas_.poisson_ratio(),
                   xe_local[0], xe_local[1]);
  // displacement u_k due to tractions t_i (on the source element) -

  il::Array<double> uk_1{2, 0.};
  uk_1[0] = uik(0, 0);
  uk_1[1] = uik(1, 0);
  il::Array<double> uk_2{2, 0.};
  uk_2[0] = uik(0, 1);
  uk_2[1] = uik(1, 1);

  auto uk_1_gl = source_elt.ConvertToGlobal(uk_1);
  auto uk_2_gl = source_elt.ConvertToGlobal(uk_2);

  // switch back to the receiver element coor
  auto uk_1_local = receiver_elt.ConvertToLocal(uk_1_gl);
  auto uk_2_local = receiver_elt.ConvertToLocal(uk_2_gl);

  std::vector<double> stnl(4, 0.); // column major storage..
  stnl[0] = uk_1_local[0];
  stnl[1] = uk_1_local[1];
  stnl[2] = uk_2_local[0];
  stnl[3] = uk_2_local[1];

  return stnl;
}
/* -------------------------------------------------------------------------- */

//       Singular Kernel   T
template <>
std::vector<double>
BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::T>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {
  //  return tractions - singular elastic kernel - Segment 0 element
  // source_elt : element object of the source element
  // i_s : integert for the source collocation number (0,1)
  // receiver_elt  : element object of the receiver element
  // i_r : integer for the collocation number where to compute the normal and
  // shear stress in the receiver element outputs: column-major (fortran order)
  // vector for the displacement to traction influence matrix

  // switch to the frame of the source element....
  il::Array<double> xe{2, 0.0};
  auto Xmid = source_elt.centroid();
  auto r_col = receiver_elt.collocation_points();
  for (int i = 0; i < 2; ++i) {
    xe[i] = r_col(i_r, i) - Xmid[i];
  }
  xe = source_elt.ConvertToLocal(xe);

  double h = source_elt.size();

  il::StaticArray2D<double, 2, 3> stress_l =
      Se_segment_0(h, this->elas_.shear_modulus(), this->elas_.poisson_ratio(),
                   xe[0], xe[1]);

  il::Array2D<double> stress_shear{2, 2, 0.}, stress_norm{2, 2, 0.};
  stress_shear(0, 0) = stress_l(0, 0);
  stress_shear(0, 1) = stress_l(0, 1);
  stress_shear(1, 1) = stress_l(0, 2);
  stress_shear(1, 0) = stress_shear(0, 1);

  stress_norm(0, 0) = stress_l(1, 0);
  stress_norm(0, 1) = stress_l(1, 1);
  stress_norm(1, 1) = stress_l(1, 2);
  stress_norm(1, 0) = stress_norm(0, 1);

  // receiver normal and tangent vector in the source local coordinates system
  auto n = source_elt.ConvertToLocal(receiver_elt.normal());

  //  in the coord sys of the source elt
  auto tt_s = il::dot(stress_shear, n);
  auto tt_n = il::dot(stress_norm, n);

  // in local coord sys of the receiver element
  auto tt_s_local =
      receiver_elt.ConvertToLocal(source_elt.ConvertToGlobal(tt_s));
  auto tt_n_local =
      receiver_elt.ConvertToLocal(source_elt.ConvertToGlobal(tt_n));

  std::vector<double> stnl(4, 0.);

  // shear stress in the receiver coor sys.
  //  effect of shear
  //        St(0, 0)
  stnl[0] = tt_s_local[0];
  //  effect of normal
  // St(0, 1)
  stnl[2] = tt_n_local[0];

  // normal stress in the receiver coor sys.
  //  effect of shear
  // St(1, 0)
  stnl[1] = tt_s_local[1];
  //  effect of normal
  // St(1, 1)
  stnl[3] = tt_n_local[1];

  return stnl;
}
/* -------------------------------------------------------------------------- */

//      HyperSingular Kernel   H
template <>
std::vector<double>
BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::H>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {
  //  return tractions - Hypersingular elastic kernel - Segment 0 element
  // source_elt : element object of the source element
  // i_s : integert for the source collocation number (0,1)
  // receiver_elt  : element object of the receiver element
  // i_r : integer for the collocation number where to compute the normal and
  // shear stress in the receiver element outputs: column-major (fortran order)
  // vector for the displacement to traction influence matrix

  // switch to the frame of the source element....
  il::Array<double> xe{2, 0.0};
  auto Xmid = source_elt.centroid();
  auto r_col = receiver_elt.collocation_points();
  for (int i = 0; i < 2; ++i) {
    xe[i] = r_col(i_r, i) - Xmid[i];
  }
  auto xe_local = source_elt.ConvertToLocal(xe);

  double h = source_elt.size();

  il::StaticArray2D<double, 2, 3> stress_l =
      We_segment_0(h, this->elas_.shear_modulus(), this->elas_.poisson_ratio(),
                   xe_local[0], xe_local[1]);

  // receiver normal and tangent vector in the source local coordinates system
  auto n = source_elt.ConvertToLocal(receiver_elt.normal());
  auto s = source_elt.ConvertToLocal(receiver_elt.tangent1());

  double n1n1 = n[0] * n[0];
  double n2n2 = n[1] * n[1];
  double n1n2 = n[0] * n[1];
  double n1s1 = n[0] * s[0];
  double n2s2 = n[1] * s[1];
  double n1s2pn2s1 = n[0] * s[1] + n[1] * s[0];

  std::vector<double> stnl(4, 0.);

  // shear stress in the receiver coor sys.
  //  effect of shear dd
  //        St(0, 0)
  stnl[0] = n1s1 * stress_l(0, 0) + n1s2pn2s1 * stress_l(0, 1) +
            n2s2 * stress_l(0, 2);
  //  effect of normal dd
  // St(0, 1)
  stnl[2] = n1s1 * stress_l(1, 0) + n1s2pn2s1 * stress_l(1, 1) +
            n2s2 * stress_l(1, 2);

  // normal stress in the receiver coor sys.
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
/* -------------------------------------------------------------------------- */

// Singular Kernel   T : Rectangular Kernel
// Displacement at rec due to displacment discontnuity at src, delta(src)
//                               and normal at src, n(src)
template <>
std::vector<double>
BieElastostatic<Segment<0>, Point<2>, ElasticKernelType::T>::influence(
    const BoundaryElement &source_elt, il::int_t i_s,
    const BoundaryElement &receiver_elt, il::int_t i_r) const {

  /*
   Traction due to point force at src, f(src)
   t(rec) = S(src, rec) x n(rec) x f(src) 
   T(src, rec) =  S(src, rec) x n(rec) 

   Displacement due to displacment discontnuity at src, delta(src)
                               and normal at src, n(src)
   u(rec) = - S (rec, src) x n(src) x delta(src)
   T(src, rec) =  - S(rec, src) x n(src) 
  */

  il::Array<double> xe{2, 0.0};
  auto Xmid = source_elt.centroid();
  auto r_col = receiver_elt.collocation_points();

  // xe = rec - src
  for (int i = 0; i < 2; ++i) {
    xe[i] = r_col(i_r, i) - Xmid[i];
  }
  xe = source_elt.ConvertToLocal(xe);

  double h = source_elt.size();


  // The below method will give us S(rec - src) 
  // but we need S(src - rec), therefore -xe is taken in input 
  il::StaticArray2D<double, 2, 3> stress_l =
      Se_segment_0(h, this->elas_.shear_modulus(), this->elas_.poisson_ratio(),
                    -xe[0], -xe[1]); /*  careful about (-) sign */

  // Further we need - S(src - rec) /* see the sign */
  il::Array2D<double> stress_shear{2, 2, 0.}, stress_norm{2, 2, 0.};
  stress_shear(0, 0) = -stress_l(0, 0); /* careful about (-) sign */
  stress_shear(0, 1) = -stress_l(0, 1);
  stress_shear(1, 1) = -stress_l(0, 2);
  stress_shear(1, 0) = stress_shear(0, 1);

  stress_norm(0, 0) = -stress_l(1, 0); /* careful about (-) sign */
  stress_norm(0, 1) = -stress_l(1, 1);
  stress_norm(1, 1) = -stress_l(1, 2);
  stress_norm(1, 0) = stress_norm(0, 1);

  // source norrmal vector in local
  auto n = source_elt.ConvertToLocal(source_elt.normal());

  //  in the local coord sys of the source elt
  auto tt_s = il::dot(stress_shear, n);
  auto tt_n = il::dot(stress_norm, n);

  // in local coord sys of the receiver element
  auto tt_s_local =
      receiver_elt.ConvertToLocal(source_elt.ConvertToGlobal(tt_s));
  auto tt_n_local =
      receiver_elt.ConvertToLocal(source_elt.ConvertToGlobal(tt_n));

  std::vector<double> stnl(4, 0.);

  // St(0, 0)
  // rec displacement in 0 direction due to DD in 0 direction (ModeII)
  stnl[0] = tt_s_local[0];
  // St(0, 1)
  // rec displacement in 0 direction due to DD in 1 direction (ModeI)
  stnl[2] = tt_n_local[0];
  // St(1, 0)
  // rec displacement in 1 direction due to DD in 0 direction (ModeII)
  stnl[1] = tt_s_local[1];
  // St(1, 1)
  // rec displacement in 1 direction due to DD in 1 direction (ModeI)
  stnl[3] = tt_n_local[1];

  return stnl;
}
/* -------------------------------------------------------------------------- */

template class BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::H>;
template class BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::U>;
template class BieElastostatic<Segment<0>, Segment<0>, ElasticKernelType::T>;
template class BieElastostatic<Segment<0>, Point<2>, ElasticKernelType::T>;
} // namespace bie

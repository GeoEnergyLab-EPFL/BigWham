//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 08.02.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <iostream>

#include <il/Array2D.h>
#include <il/math.h>

#include "elastic_2dP0_segment.h"

namespace bie {

// Function for the
// integral of the fundamental displacement due to point force over a segment of
// size h centered on the origin and lying on the axis e_1 (with normal e_2_ on
// a point (x1,x2)  for a uniform point force over that segment Ue_ij(x1,x2)
// =\int_e U_ij(x1-xi,x2) dxi where U_ij is the ith displacement component due
// to unit force along e_j
il::StaticArray2D<double, 2, 2> Ue_segment_0(double h, double G, double nu,
                                             double x1_o, double y1_o) {
  double overhalfh = 2. / h;
  double x1 = x1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  double x2 = y1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  double x1_2 = x1 * x1;
  double x2_2 = x2 * x2;
  // coef
  double beta_u = overhalfh / (8. * il::pi * G * (1 - nu));

  double three_4_nu = 3 - 4 * nu;
  double x1m1 = x1 - 1.;
  double x1p1 = x1 + 1.;
  il::StaticArray2D<double, 2, 2> Ue{0.};

  // note that Ue_12 = U2_21 ....
  if (x2 == 0.) { // on the segment case
    Ue(0, 0) = beta_u * (2 - (three_4_nu * (-4 - (-x1m1) * log(pow(x1m1, 2)) +
                                            (1 + x1) * log(pow(1 + x1, 2)))) /
                                 2.);
    Ue(1, 1) =
        beta_u * (-0.5 * (three_4_nu * (-4 - (-x1m1) * log(pow(x1m1, 2)) +
                                        (1 + x1) * log(pow(1 + x1, 2)))));
  } else {
    // case x2!=0
    Ue(0, 0) =
        beta_u *
        (2 + x2 * atan(x1m1 / x2) - x2 * atan(x1p1 / x2) -
         three_4_nu * (-2 + x2 * atan((-x1m1) / x2) + x2 * atan((x1p1) / x2) -
                       (x1m1 * log(1 - 2 * x1 + x1_2 + x2_2)) / 2. +
                       (x1p1 * log(1 + 2 * x1 + x1_2 + x2_2)) / 2.));
    Ue(0, 1) =
        beta_u *
        (x2 * (-log(pow(-1 + x1, 2) + x2_2) + log(pow(1 + x1, 2) + x2_2))) / 2.;
    Ue(1, 0) = Ue(0, 1);
    Ue(1, 1) =
        beta_u * (2 * (-1 + 2 * nu) * x2 * atan((1 - x1) / x2) +
                  2 * (-1 + 2 * nu) * x2 * atan((x1p1) / x2) -
                  ((-three_4_nu) * (4 + x1m1 * log(1 - 2 * x1 + x1_2 + x2_2) -
                                    x1p1 * log(1 + 2 * x1 + x1_2 + x2_2))) /
                      2.);
  }
  return Ue;
}

// integration of the stress solution due a point force uniform on the segment
// -h/2,h/2
//   for operators S and T=S.n
il::StaticArray2D<double, 2, 3> Se_segment_0(double h, double G, double nu,
                                             double x1_o, double y1_o) {
  double overhalfh = 2. / h;
  double x1 = x1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  double x2 = y1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  // if (x2 == 0.) {
  //   x2 -= 1e-5;
  // }

  double x1_2 = x1 * x1;
  double x2_2 = x2 * x2;
  // coef
  double beta_s = overhalfh / (4. * il::pi * (1 - nu));

  double x1m1 = x1 - 1.;
  double x1p1 = x1 + 1.;
  // Symmetry !   Se_ijk = Se_jik   !  such that we only need to store 3*2 = 6
  // entries ! and we output a 3*2 array

  il::StaticArray2D<double, 2, 3> Se{0.};
  // Se_111, Se_121, Se_221
  // Se_112,Se_122, Se_222
  // in-plane case
  // check Eq. 5.2.8 Crouch Starfield BEM book
  // (Dx / 2) * h
  if (il::abs(x2) <= 1e-7) {
    // by convention we take the limit from-below !
    if (il::abs(x1) <= 1.0) {
      Se(0, 1) = 0.5 * overhalfh;
      Se(1, 2) = 0.5 * overhalfh;
    } else {
      double tmpfac = beta_s * (1 - 2 * nu) *
        log(il::abs((x1 - 1) / (x1 + 1)));
      Se(0, 2) = -tmpfac;
      Se(1, 1) = tmpfac;
    }
    // std::cout << "Dont come here \n" << std::endl;
  } else {

    double aux22 = pow(1 + x2_2, 2);
    double atan1p12 = atan(x1p1 / x2);
    double atan1m12 = atan((-x1m1) / x2);
    // Se_111
    Se(0, 0) = beta_s * ((4 * x1 * x2_2) /
                             (x1_2 * x1_2 + 2 * x1_2 * (-1 + x2_2) + aux22) +
                         (1.5 - nu) * log(x1m1 * x1m1 + x2_2) +
                         (-1.5 + nu) * log(x1p1 * x1p1 + x2_2));
    // Se_121
    Se(0, 1) =
        beta_s * (2 * ((x2 * (1 - x1_2 + x2_2)) / ((1 - 2 * x1 + x1_2 + x2_2) *
                                                   (1 + 2 * x1 + x1_2 + x2_2)) +
                       (-1 + nu) * atan1m12 + (-1 + nu) * atan1p12));
    // Se_221
    Se(0, 2) =
        beta_s *
        ((-4 * x1 * x2_2) / (x1_2 * x1_2 + 2 * x1_2 * (-1 + x2_2) + aux22) +
         ((-1 + 2 * nu) * (log(x1m1 * x1m1 + x2_2) - log(x1p1 * x1p1 + x2_2))) /
             2.);
    // Se_112
    Se(1, 0) = beta_s *
               ((2 * x2 * (1 - x1_2 + x2_2)) /
                    ((1 - 2 * x1 + x1_2 + x2_2) * (1 + 2 * x1 + x1_2 + x2_2)) -
                2 * nu * atan1m12 - 2 * nu * atan1p12);
    // Se_122
    Se(1, 1) =
        beta_s *
        ((-4 * x1 * x2_2) / (x1_2 * x1_2 + 2 * x1_2 * (-1 + x2_2) + aux22) -
         ((-1 + 2 * nu) * (log(x1m1 * x1m1 + x2_2) - log(x1p1 * x1p1 + x2_2))) /
             2.);
    // Se_222
    Se(1, 2) = beta_s *
               (2 * ((x2 * (-1 + x1_2 - x2_2)) / ((1 - 2 * x1 + x1_2 + x2_2) *
                                                  (1 + 2 * x1 + x1_2 + x2_2)) +
                     (-1 + nu) * atan1m12 + (-1 + nu) * atan1p12));
  }
  return Se;
}

/// Integration of  Vijkl.nl on the unit segment !
// integration of the stress solution due a point edge dislocation on the
// segment -h/2,h/2
//   for operators W and H=W.n
il::StaticArray2D<double, 2, 3> We_segment_0(double h, double G, double nu,
                                             double x1_o, double y1_o) {
  double overhalfh = 2. / h;
  double x1 = x1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  double x2 = y1_o * overhalfh; // change of variable to perform the computation
  // for a unit segment [-1,1]
  double x1_2 = x1 * x1;
  double x2_2 = x2 * x2;
  // coef
  double beta_v = G * overhalfh / (2. * il::pi * (1 - nu));

  double x12m1 = (-1 + x1_2) * (-1 + x1_2);
  double x1p1 = x1 + 1.;

  double auxdenom =
      pow(2 * x1_2 * (-1 + x2_2) + x1_2 * x1_2 + (1 + x2_2) * (1 + x2_2), 2);
  double aux1 = (-1 + x1_2 - 2 * x1 * x2 - x2_2);
  double aux2 = (-1 + x1_2 + 2 * x1 * x2 - x2_2);

  il::StaticArray2D<double, 2, 3> We{0.};

  // Wijk = Vijk2
  // W11_1 = V1112
  We(0, 0) =
      beta_v * (4 * ((1 - nu) / (-1 + nu)) * x1 * x2 *
                (-2 * (1 + x1_2) * x2_2 - 3 * x12m1 + x2_2 * x2_2) / auxdenom);
  // W12_1=V1212
  We(0, 1) = beta_v * (2 * ((1 - nu) / (-1 + nu)) * aux1 * aux2 *
                       (-1 + x1_2 + x2_2) / auxdenom);
  // W22_1=V2212
  We(0, 2) =
      beta_v * (4 * ((1 - nu) / (-1 + nu)) * x1 * x2 *
                (-2 * (1 + x1_2) * x2_2 + x12m1 - 3 * x2_2 * x2_2) / auxdenom);
  // W11_2=V1122
  We(1, 0) = beta_v * (2 * ((1 - nu) / (-1 + nu)) * aux1 * aux2 *
                       (-1 + x1_2 + x2_2) / auxdenom);

  // W12_2 = V1222
  We(1, 1) =
      beta_v * (4 * ((1 - nu) / (-1 + nu)) * x1 * x2 *
                (-2 * (1 + x1_2) * x2_2 + x12m1 - 3 * x2_2 * x2_2) / auxdenom);
  // W22_2 = V2222
  We(1, 2) = beta_v *
             (2 * ((1 - nu) / (-1 + nu)) *
              ((-1 + x1_2) * x12m1 + x2_2 * (-5 - 2 * x1_2 + 7 * x1_2 * x1_2) +
               (-7 + 3 * x1_2) * x2_2 * x2_2 - 3 * pow(x2, 6)) /
              auxdenom);

  return We;
}

} // namespace bie

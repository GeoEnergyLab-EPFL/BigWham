//
// This file part of BigWham
//
// Created by nikolski on 6/29/2017.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2017.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//

#ifndef BIE_CONSTANTS_H
#define BIE_CONSTANTS_H

#include <complex>

// todo :: I don t like such a file - it is used only by the the 3D code function and have little content

namespace bie {

 //   const double pi = il::pi; //3.1415926535897932385;   // is this needed as il::pi does it -> change everywhere
    const std::complex<double> ii = std::complex<double>{0.0, 1.0};
    // tolerance parameters
    const double h_tol = 1.0e-010; // 1.0e-013 2.221e-016
    const double a_tol = 1.0e-008; // 3.162e-007 1.825e-008

}

#endif //BIE_CONSTANTS_H

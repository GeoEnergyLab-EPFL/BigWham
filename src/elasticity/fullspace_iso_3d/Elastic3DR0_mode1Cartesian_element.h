//
// This file is part of HFPx3D.
//
// Created by Carlo Peruzzo on 01.03.21.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//



#include <il/StaticArray2D.h>
#include "core/oldies/FaceData.h"
#include "core/ElasticProperties.h"


namespace bie{
double StressesKernelR0opening(
    double& xx, double& yy, double& zz, double& a, double& b, double& G,
    double& nu) ;

bool is_stress_singular_at_given_location(double&x, double& y, double& z, double& a, double& b, bool verbose = true) ;


double traction_influence_3DR0opening(
            bie::FaceData &elem_data_s, // source element
            bie::FaceData &elem_data_r, // receiver element
            bie::ElasticProperties const &elas_ // elastic properties
            ) ;


// second order derivatives of I(x,y,z,xi,eta)

double ip33(double& x, double& y, double& z, double& xi, double& eta) ;

//// third order derivatives of I(x,y,z,xi,eta)

double ip333(double& x, double& y, double& z, double& xi, double& eta) ;

// special cases:

double Ip33_lim_z_to_0_and_x_to_a(double& x, double& y, double& a, double& b);

double Ip33_lim_z_to_0_and_y_to_b(double& x, double& y, double& a, double& b);

}

//
// This file part of BigWham
//
// Created by Brice Lecampion on 04.02.19.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//


#include <il/StaticArray.h>

namespace bie{

typedef double (*vFunctionCall)(double& x, double& y, double& z, double& xi, double& eta);

double rectangular_integration(double& x, double& y, double& z, double& a, double& b, vFunctionCall Func);

bool is_stress_singular_at_given_location(double& x, double& y, double& z, double& a, double& b, bool verbose);

double get_Ip3 (double & x,double &y,double &z,double &a,double &b, double& Ip3_out_plane_z_EQ_0, bool verbose = true);

il::Array2D<double> transpose(il::Array2D<double> M);

il::Array2D<double> change_ref_system (const il::Array2D<double>& linearApplication,il::int_t change_domain, il::int_t change_codomain, const il::Array2D<double>& RglobalTOlocal_domain, const il::Array2D<double>& RglobalTOlocal_codomain);

il::StaticArray<double,2> get_a_and_b(il::Array2D <double>xv, double NoV);


// first order derivatives of I(x,y,z,xi,eta)

double ip1(double& x, double& y, double& z, double& xi, double& eta) ;

double ip2(double& x, double& y, double& z, double& xi, double& eta) ;

double ip3(double& x, double& y, double& z, double& xi, double& eta) ;

// second order derivatives of I(x,y,z,xi,eta)

double ip11(double& x, double& y, double& z, double& xi, double& eta) ;

double ip12(double& x, double& y, double& z, double& xi, double& eta) ;

double ip13(double& x, double& y, double& z, double& xi, double& eta) ;

double ip22(double& x, double& y, double& z, double& xi, double& eta) ;

double ip23(double& x, double& y, double& z, double& xi, double& eta) ;

double ip33(double& x, double& y, double& z, double& xi, double& eta) ;

//// third order derivatives of I(x,y,z,xi,eta)

double ip111(double& x, double& y, double& z, double& xi, double& eta) ;

double ip112(double& x, double& y, double& z, double& xi, double& eta) ;

double ip113(double& x, double& y, double& z, double& xi, double& eta) ;

double ip122(double& x, double& y, double& z, double& xi, double& eta) ;

double ip123(double& x, double& y, double& z, double& xi, double& eta) ;

double ip133(double& x, double& y, double& z, double& xi, double& eta) ;

double ip222(double& x, double& y, double& z, double& xi, double& eta) ;

double ip223(double& x, double& y, double& z, double& xi, double& eta) ;

double ip233(double& x, double& y, double& z, double& xi, double& eta) ;

double ip333(double& x, double& y, double& z, double& xi, double& eta) ;

// special cases:

double Ip33_lim_z_to_0_and_x_to_a(double& x, double& y, double& a, double& b);

double Ip33_lim_z_to_0_and_y_to_b(double& x, double& y, double& a, double& b);

}

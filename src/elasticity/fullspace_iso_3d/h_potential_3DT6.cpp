//
// This file part of BigWham
//
// Created by D. Nikolski on 1/24/2017.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2017.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//

// Integration of the hypersingular kernel of the elasticity equation
// over a part of a polygonal element (a sector associated with one edge)
// with 2nd order polynomial approximating (shape) functions.
//
// To be contracted (via right multiplication) with the vector of
// constituing functions defined in SijK.cpp
// and (via left multiplication) with the vector of
// shape function coefficients associated with each node of the element
//
// Stress components (vs local Cartesian coordinate system of the element)
// combined as S11+S22, S11-S22+2*I*S12, S13+I*S23, S33

#include <elasticity/3d/constants.h>
#include <elasticity/3d/h_potential_3DT6.h>
#include <il/StaticArray.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/math.h>
#include <complex>

namespace bie {

// General case (h!=0, collocation point projected into or outside the element)

    il::StaticArray4D<std::complex<double>, 6, 4, 3, 9> s_ij_gen_h(double nu, std::complex<double> eix,double h, std::complex<double> d) {
        // const std::complex<double> I(0.0, 1.0);

        double c_1_nu = 1.0 + nu;
        double c_1_2nu = 1.0 + 2.0 * nu;
        double c_2_nu = 2.0 + nu;
        double c_3_nu = 3.0 + nu;
        double c_3_2nu = 3.0 + 2.0 * nu;
        double c_4_nu = 4.0 + nu;
        double c_5_4nu = 5.0 + 4.0 * nu;
        double c_6_nu = 6.0 + nu;
        double c_7_2nu = 7.0 + 2.0 * nu;
        double c_7_5nu = 7.0 + 5.0 * nu;
        double c_7_6nu = 7.0 + 6.0 * nu;
        double c_11_4nu = 11.0 + 4.0 * nu;
        double c_11_5nu = 11.0 + 5.0 * nu;
        double c_12_nu = 12.0 + nu;
        double c_13_2nu = 13.0 + 2.0 * nu;
        double c_13_10nu = 13.0 + 10.0 * nu;

        double c_1_mnu = 1.0 - nu;
        double c_1_m2nu = 1.0 - 2.0 * nu;
        double c_2_mnu = 2.0 - nu;
        double c_3_mnu = 3.0 - nu;
        double c_3_m4nu = 3.0 - 4.0 * nu;
        double c_5_mnu = 5.0 - nu;
        double c_5_m2nu = 5.0 - 2.0 * nu;
        double c_5_m4nu = 5.0 - 4.0 * nu;
        double c_6_m5nu = 6.0 - 5.0 * nu;
        double c_7_m2nu = 7.0 - 2.0 * nu;
        double c_8_m5nu = 8.0 - 5.0 * nu;
        double c_9_m2nu = 9.0 - 2.0 * nu;
        double c_9_m4nu = 9.0 - 4.0 * nu;
        double c_9_m8nu = 9.0 - 8.0 * nu;
        double c_13_m2nu = 13.0 - 2.0 * nu;
        double c_15_m4nu = 15.0 - 4.0 * nu;
        double c_15_m8nu = 15.0 - 8.0 * nu;
        double c_115_m38nu_80 = 1.4375 - 0.475 * nu;

        double cos_x = std::real(eix);
        double tan_x = std::imag(eix) / cos_x;
        std::complex<double> tcos_x = cos_x * eix;
        std::complex<double> tcos_c = std::conj(tcos_x);
        std::complex<double> c_tcos_m1 = tcos_x - 1.0;
        std::complex<double> c_3_4tcos = 3.0 + 4.0 * tcos_x;
        std::complex<double> c_5_8tcos = 5.0 + 8.0 * tcos_x;
        std::complex<double> e2x = eix * eix;
        std::complex<double> c_tcos_n1 = c_tcos_m1 * tcos_x;
        std::complex<double> w_c_tcos_n2 =(13.0 + e2x - 10.0 * tcos_x) * tcos_x;
        std::complex<double> c_8_3i_tan{8.0, 3.0 * tan_x};

        double h2 = h * h;
        double h3 = h2 * h;
        double h4 = h2 * h2;
        double sgh = ((h < 0) ? -1.0 : double((h > 0))); // sign(h)
        double abh = std::fabs(h);

        double d_1 = std::abs(d);
        double d_2 = d_1 * d_1;
        double d_4 = d_2 * d_2;
        std::complex<double> d2h2 = d_2 * h2;
        std::complex<double> d_c = std::conj(d);
        double d_cos_p = std::real(d);
        double d_sin_p = std::imag(d);
        std::complex<double> e = std::polar(1.0, std::arg(d)); //  = d/d_1
        std::complex<double> e_c = std::conj(e);
        double cos_p = std::real(e);
        double sin_p = std::imag(e);
        std::complex<double> e_2 = e * e; //  = d^2/d_1^2
        std::complex<double> e_2_c = std::conj(e_2);
        std::complex<double> e_3 = e * e_2; //  = d^3/d_1^3
        std::complex<double> e_4 = e_2 * e_2; //  = d^4/d_1^4

        std::complex<double> c_d_h = d_2 + h2;
        std::complex<double> c_d_3h = d_2 + 3.0 * h2;
        std::complex<double> c_d_m3h = d_2 - 3.0 * h2;
        std::complex<double> c_3d_h = 3.0 * d_2 + h2;

        std::complex<double> p0, p1, p2;

        il::StaticArray4D<std::complex<double>, 6, 4, 3, 9> c_array{0.0};

        // S_11 + S_22
        
        c_array(0, 0, 0, 2) = h * d_sin_p;
        c_array(0, 0, 1, 2) = -h * d_cos_p;
        p0 = 0.1875 * h;
        p1 = 3.0 * h2;
        p2 = d_2 * tan_x;
        c_array(0, 0, 0, 3) = -p0 * (p1 * d_sin_p + p2 * d_cos_p);
        c_array(0, 0, 1, 3) = p0 * (p1 * d_cos_p - p2 * d_sin_p);
        p0 = c_7_2nu * h;
        c_array(0, 0, 0, 6) = p0 * cos_p;
        c_array(0, 0, 1, 6) = p0 * sin_p;
        c_array(0, 0, 2, 6) = -c_1_2nu * d_1;
        p0 = 3.0 * h * c_d_3h;
        c_array(0, 0, 0, 7) = p0 * cos_p;
        c_array(0, 0, 1, 7) = p0 * sin_p;
        c_array(0, 0, 2, 7) = -2.0 * h2 * d_1;
        p0 = -0.5 * h * c_d_m3h * c_d_h;
        c_array(0, 0, 0, 8) = p0 * cos_p;
        c_array(0, 0, 1, 8) = p0 * sin_p;

        c_array(1, 0, 1, 1) = 0.2 * c_11_5nu * h * e_2 * tcos_x;
        c_array(1, 0, 0, 1) = ii * c_array(1, 0, 1, 1);
        p1 = 0.1 * (d_2 * (7.0 + 2.0 * ii * tan_x) + 16.0 * h2 * tcos_x) * e_2;
        p2 = c_7_5nu / 60.0 * d_2 * tan_x;
        c_array(1, 0, 0, 2) = -(ii * p1 + p2) * h;
        c_array(1, 0, 1, 2) = -(p1 + ii * p2) * h;
        p1 = (d_2 * h2 * (0.4 + 0.11875 * ii * tan_x) +0.09375 * ii * d_4 * tan_x + 0.4 * h4 * tcos_x) * e_2;
        p2 = (-0.05625 * d_2 + 0.11875 * h2) * d_2 * tan_x;
        c_array(1, 0, 0, 3) = (ii * p1 + p2) * h;
        c_array(1, 0, 1, 3) = (p1 + ii * p2) * h;
        c_array(1, 0, 0, 4) = -c_1_nu * sgh;
        c_array(1, 0, 1, 4) = -ii * c_1_nu * sgh;
        c_array(1, 0, 2, 5) = 0.5 * c_1_2nu * e;
        p1 = 0.5 * c_7_2nu * d * e;
        p2 = d_1 * (0.3 + 4.0 / 3.0 * c_2_nu);
        c_array(1, 0, 0, 6) = (p1 + p2) * h;
        c_array(1, 0, 1, 6) = ii * (-p1 + p2) * h;
        c_array(1, 0, 2, 6) = 2.0 * c_2_nu * h2 * e;
        p1 = 1.5 * d * e * c_d_3h;
        p2 = d_1 * (1.0 / 6.0 * c_1_2nu * d_2 +
                h2 * (43.0 / 30.0 + c_2_nu / 3.0));
        c_array(1, 0, 0, 7) = (p1 + p2) * h;
        c_array(1, 0, 1, 7) = ii * (-p1 + p2) * h;
        c_array(1, 0, 2, 7) = 2.0 * h4 * e;
        p0 = h * c_d_h;
        p1 = 0.15 * d_1 * d_2 - 1.9 / 6.0 * d_1 * h2;
        p2 = 0.25 * d * e * c_d_m3h;
        c_array(1, 0, 0, 8) = -p0 * (p1 + p2);
        c_array(1, 0, 1, 8) = ii * p0 * (-p1 + p2);

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(2, 0, j, k) = std::conj(c_array(1, 0, j, k));
            }
        }

        c_array(3, 0, 2, 0) = ii * c_1_2nu * e_2 * tcos_x;
        p0 = d * h;
        p1 = 0.0625 * c_13_10nu;
        p2 = e_2 * (0.0625 * c_13_10nu + 0.5 * c_3_2nu * tcos_x);
        c_array(3, 0, 0, 1) = -ii * p0 * (p1 - p2);
        c_array(3, 0, 1, 1) = p0 * (p1 + p2);
        c_array(3, 0, 2, 1) = 2.0 * ii * c_2_nu * e_2 * h2 * tcos_x;
        //p1 = ; p2 = ;
        c_array(3, 0, 0, 2) = d * h *
                (0.09375 * ii * c_7_2nu * h2 -
                 0.03125 * c_3_2nu * d_2 * tan_x +
                        e_2 * (d_2 * (-1.0 / 12.0 * ii * c_7_6nu +
                                0.09375 * c_3_2nu * tan_x) -
                        ii * h2 * (0.09375 * c_7_2nu +
                                        0.25 * c_4_nu * tcos_x)));
        c_array(3, 0, 1, 2) = d * h *
                (-0.09375 * c_7_2nu * h2 -
                 ii * 0.03125 * c_3_2nu * d_2 * tan_x -
                        e_2 * (d_2 * (1.0 / 12.0 * c_7_6nu +
                                0.09375 * ii * c_3_2nu * tan_x) +
                        h2 * (0.09375 * c_7_2nu +
                                        0.25 * c_4_nu * tcos_x)));
        c_array(3, 0, 2, 2) = -ii * e_2 * h4 * tcos_x;
        //p0 = ; p1 = ; p2 = ;
        c_array(3, 0, 0, 3) = d * h * 
                (d_2 * tan_x * ((0.09375 - 0.28125 * e_2) * h2 - 
                        (0.046875 + 0.109375 * e_2) * d_2) + 
                        ii * h2 * (0.625 * e_2 * d_2 - 0.234375 * h2 +
                                e_2 * h2 * (0.234375 + 0.3125 * tcos_x)));
        c_array(3, 0, 1, 3) = d * h * 
                (ii * d_2 * tan_x * ((0.09375 + 0.28125 * e_2) * h2 +
                        (-0.046875 + 0.109375 * e_2) * d_2) + 
                        h2 * (0.625 * e_2 * d_2 + 0.234375 * h2 + 
                                e_2 * h2 * (0.234375 + 0.3125 * tcos_x)));
        c_array(3, 0, 0, 5) = c_3_2nu * h * e * (0.25 * e_2 - 0.75);
        c_array(3, 0, 1, 5) = -ii * c_3_2nu * h * e * (0.25 * e_2 + 0.75);
        c_array(3, 0, 2, 5) = 0.5 * c_1_2nu * d * e;
        p0 = h * e;
        p1 = e_2 * (0.75 * c_5_4nu * d_2 + 0.25 * c_11_4nu * h2);
        p2 = 0.25 * c_5_4nu * d_2 + 0.75 * c_11_4nu * h2;
        c_array(3, 0, 0, 6) = h * e * (p1 - p2);
        c_array(3, 0, 1, 6) = -ii * h * e * (p1 + p2);
        c_array(3, 0, 2, 6) = 2.0 * c_2_nu * h2 * d * e;
        //p1 = ; p2 = ;
        c_array(3, 0, 0, 7) = h * e * 
                (0.125 * c_1_2nu * d_4 - 0.25 * c_7_2nu * d_2 * h2 - 
                        0.375 * c_13_2nu * h4 + 
                        e_2 * (0.625 * c_1_2nu * d_4 + 
                                0.75 * c_7_2nu * d_2 * h2 + 
                                0.125 * c_13_2nu * h4));
        c_array(3, 0, 1, 7) = ii * h * e *
                (0.125 * c_1_2nu * d_4 - 0.25 * c_7_2nu * d_2 * h2 - 
                        0.375 * c_13_2nu * h4 - 
                        e_2 * (0.625 * c_1_2nu * d_4 +
                                 0.75 * c_7_2nu * d_2 * h2 +
                                 0.125 * c_13_2nu * h4));
        c_array(3, 0, 2, 7) = 2.0 * h4 * d * e;
        p0 = h * e * c_d_h;
        p1 = e_2 * (7.0 / 24.0 * d_4 -
                      11.0 / 12.0 * d_2 * h2 - 5.0 / 24.0 * h4);
        p2 = 0.125 * d_4 - 0.25 * d_2 * h2 + 0.625 * h4;
        c_array(3, 0, 0, 8) = -p0 * (p1 + p2);
        c_array(3, 0, 1, 8) = ii * p0 * (p1 - p2);

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(4, 0, j, k) = std::conj(c_array(3, 0, j, k));
            }
        }

        p0 = 0.125 * c_13_10nu * h;
        c_array(5, 0, 0, 1) = p0 * d_sin_p;
        c_array(5, 0, 1, 1) = -p0 * d_cos_p;
        c_array(5, 0, 2, 1) = 1.0 / 12.0 * c_1_2nu * d_2 * tan_x;
        p1 = 0.0625 * c_3_2nu * d_2 * tan_x;
        p2 = 0.1875 * c_7_2nu * h2;
        c_array(5, 0, 0, 2) = -h * (p1 * d_cos_p + p2 * d_sin_p);
        c_array(5, 0, 1, 2) = -h * (p1 * d_sin_p - p2 * d_cos_p);
        c_array(5, 0, 2, 2) = -1.0 / 6.0 * c_2_nu * d_2 * h2 * tan_x;
        p1 = 0.09375 * d_2 * tan_x * (2 * h2 - d_2);
        p2 = 0.46875 * h4;
        c_array(5, 0, 0, 3) = h * (p1 * d_cos_p + p2 * d_sin_p);
        c_array(5, 0, 1, 3) = h * (p1 * d_sin_p - p2 * d_cos_p);
        c_array(5, 0, 2, 3) = 0.25 * d_2 * h4 * tan_x;
        c_array(5, 0, 2, 4) = -4.0 * c_1_nu * abh;
        p0 = -1.5 * c_3_2nu * h;
        c_array(5, 0, 0, 5) = p0 * cos_p;
        c_array(5, 0, 1, 5) = p0 * sin_p;
        c_array(5, 0, 2, 5) = -0.5 * c_1_2nu * d_1;
        p0 = -h * (0.5 * c_5_4nu * d_2 + 1.5 * c_11_4nu * h2);
        c_array(5, 0, 0, 6) = p0 * cos_p;
        c_array(5, 0, 1, 6) = p0 * sin_p;
        c_array(5, 0, 2, 6) = (1.0 / 6.0 * c_1_2nu * d_2 + 1.5 * h2 * (5 + 2 * nu)) * d_1;
        p0 = h * (0.25 * c_1_2nu * d_4 - 0.5 * c_7_2nu * d_2 * h2 - 0.75 * c_13_2nu * h4);
        c_array(5, 0, 0, 7) = p0 * cos_p;
        c_array(5, 0, 1, 7) = p0 * sin_p;
        c_array(5, 0, 2, 7) = 2.0 / 3.0 * h2 * (c_2_nu * d_2 +h2 * (7 + nu)) * d_1;
        p0 = -h * c_d_h * (0.25 * d_4 - 0.5 * d_2 * h2 + 1.25 * h4);
        c_array(5, 0, 0, 8) = p0 * cos_p;
        c_array(5, 0, 1, 8) = p0 * sin_p;
        c_array(5, 0, 2, 8) = 2.0 / 3.0 * h4 * c_d_h * d_1;

        
        // S_11 - S_22 + 2 * I * S_12
        
        c_array(0, 1, 2, 1) = -ii * c_1_m2nu * e_2 * tcos_x;
        c_array(0, 1, 0, 2) = ii * d * h * (-0.5 + e_2 * (0.5 + 0.75 * tcos_x));
        c_array(0, 1, 1, 2) = d * h * (0.5 + e_2 * (0.5 + 0.75 * tcos_x));
        c_array(0, 1, 2, 2) = ii * h2 * e_2 * tcos_x;
        //p0 = d*h; p1 = ; p2 = ;
        c_array(0, 1, 0, 3) = d * h * 
                (0.28125 * ii * h2 - 0.09375 * d_2 * tan_x +
                        e_2 * (d_2 * (-0.75 * ii + 0.28125 * tan_x) -
                                 ii * h2 * (0.28125 + 0.375 * tcos_x)));
        c_array(0, 1, 1, 3) = -d * h * 
                (0.28125 * h2 + 0.09375 * ii * d_2 * tan_x +
                        e_2 * (d_2 * (0.75 + 0.28125 * ii * tan_x) +
                                 h2 * (0.28125 + 0.375 * tcos_x)));
        p0 = 0.5 * e * h;
        p1 = 3.0 * e_2;
        c_array(0, 1, 0, 6) = p0 * (-p1 + c_9_m4nu);
        c_array(0, 1, 1, 6) = ii * p0 * (p1 + c_9_m4nu);
        c_array(0, 1, 2, 6) = -c_1_m2nu * e * d;
        p0 = 1.5 * h * e;
        p2 = c_3d_h * e_2;
        c_array(0, 1, 0, 7) = p0 * (c_d_3h - p2);
        c_array(0, 1, 1, 7) = ii * p0 * (c_d_3h + p2);
        c_array(0, 1, 2, 7) = -2.0 * h2 * e * d;
        p0 = 0.25 * h * e * c_d_h;
        p2 = e_2 * (5.0 * d_2 + h2);
        c_array(0, 1, 0, 8) = -p0 * (c_d_m3h + p2);
        c_array(0, 1, 1, 8) = ii * p0 * (-c_d_m3h + p2);

        p0 = 0.4 * e_2 * h * tcos_x;
        p2 = 8.0 * e_2 * (-1.0 + tcos_x);
        c_array(1, 1, 0, 1) = ii * p0 * (c_8_m5nu + p2);
        c_array(1, 1, 1, 1) = p0 * (-c_8_m5nu + p2);
        c_array(1, 1, 2, 1) = -ii * c_1_m2nu * d * e_2 * (0.625 + tcos_x);
        p0 = e_2 * h; //p1 = ; p2 = ;
        c_array(1, 1, 0, 2) = p0 * (d_2 * (-0.7 * ii + 0.2 * tan_x) -
                1.6 * ii * h2 * tcos_x -
                2.0 * e_2 * (0.8 * ii * h2 * c_tcos_n1 +
                        d_2 * (0.1 * tan_x - ii * (0.7 + 0.4 * tcos_x))));
        c_array(1, 1, 1, 2) = p0 * (d_2 * (0.7 + 0.2 * ii * tan_x) +
                1.6 * h2 * tcos_x + 
                2.0 * e_2 * (-0.8 * h2 * c_tcos_n1 + 
                        d_2 * (0.7 + 0.1 * ii * tan_x + 0.4 * tcos_x)));
        c_array(1, 1, 2, 2) = d * e_2 * 
                (-d_2 * c_1_m2nu * (-0.5 * ii + 0.1875 * tan_x) +
                        ii * h2 * (1.1875 + 1.75 * tcos_x -
                                0.125 * nu * c_3_4tcos));
        //p1 = ; p2 = ;
        c_array(1, 1, 0, 3) = p0 * (d2h2 * (0.4 * ii - 0.11875 * tan_x) -
                0.09375 * d_4 * tan_x + 0.4 * ii * h4 * tcos_x +
                e_2 * (3.0 * d_4 * (-0.4 * ii + 0.18125 * tan_x) +
                        0.4 * ii * h4 * c_tcos_n1 +
                        d2h2 * (0.11875 * tan_x - 0.4 * ii * (2.0 + tcos_x))));
        c_array(1, 1, 1, 3) = -p0 * (d2h2 * (0.4 + 0.11875 * ii * tan_x) +
                0.09375 * ii * d_4 * tan_x + 0.4 * h4 * tcos_x +
                e_2 * (3.0 * d_4 * (0.4 + 0.18125 * ii * tan_x) -
                        0.4 * h4 * c_tcos_n1 + 
                        d2h2 * (0.11875 * ii * tan_x + 0.4 * (2.0 + tcos_x))));
        c_array(1, 1, 2, 3) = -ii * d * e_2 * h2 *
                (d_2 * (1.5 + 0.5625 * ii * tan_x) +
                        0.1875 * h2 * c_3_4tcos);
        c_array(1, 1, 2, 5) = -0.5 * c_1_m2nu * e_3;
        p0 = d * e * h;
        c_array(1, 1, 0, 6) = -p0 * (4.5 * e_2 - 0.5 * c_9_m4nu);
        c_array(1, 1, 1, 6) = ii * p0 * (4.5 * e_2 + 0.5 * c_9_m4nu);
        c_array(1, 1, 2, 6) = -e_3 * (2.0 * c_2_mnu * h2 + 
                3.0 * c_1_m2nu * d_2);
        p1 = 1.5 * d_2 + 4.5 * h2;
        p2 = e_2 * (7.5 * d_2 + 4.5 * h2);
        c_array(1, 1, 0, 7) = p0 * (p1 - p2);
        c_array(1, 1, 1, 7) = ii * p0 * (p1 + p2);
        c_array(1, 1, 2, 7) = -0.25 * e_3 * 
                (5.0 * d_4 * c_1_m2nu + 
                6.0 * d2h2 * c_7_m2nu + h4 * c_13_m2nu);
        p0 = 0.25 * p0 * c_d_h;
        p1 = d_2 - 3.0 * h2;
        p2 = e_2 * (7.0 * d_2 + 3.0 * h2);
        c_array(1, 1, 0, 8) = -p0 * (p1 + p2);
        c_array(1, 1, 1, 8) = ii * p0 * (-p1 + p2);
        c_array(1, 1, 2, 8) = -0.5 * e_3 * h2 * c_d_h * (5.0 * d_2 + h2);

        c_array(2, 1, 0, 1) = 3.2 * ii * h * e_2 * tcos_x;
        c_array(2, 1, 1, 1) = 3.2 * h * e_2 * tcos_x;
        c_array(2, 1, 2, 1) = 0.625 * ii * c_1_m2nu * d;
        p1 = 0.1 * e_2 * 
                (d_2 * (7.0 + 2.0 * ii * tan_x) + 16.0 * h2 * tcos_x);
        p2 = ii * c_6_m5nu / 30.0 * d_2 * tan_x;
        c_array(2, 1, 0, 2) = -ii * h * (p1 - p2);
        c_array(2, 1, 1, 2) = -h * (p1 + p2);
        c_array(2, 1, 2, 2) = d * 
                (0.0625 * c_1_m2nu * d_2 * tan_x - 
                        ii * h2 * (1.1875 - 0.375 * nu));
        p1 = d_2 * tan_x * (-0.05625 * d_2 + 0.11875 * h2);
        p2 = e_2 * (d_2 * tan_x * (0.11875 * h2 + 0.09375 * d_2) -
                    0.4 * ii * h2 * (d_2 + h2 * tcos_x));
        c_array(2, 1, 0, 3) = h * (p1 - p2);
        c_array(2, 1, 1, 3) = ii * h * (p1 + p2);
        c_array(2, 1, 2, 3) = d * h2 * 
                (-0.1875 * d_2 * tan_x + 0.5625 * ii * h2);
        c_array(2, 1, 0, 4) = -2.0 * c_1_mnu * sgh;
        c_array(2, 1, 1, 4) = ii * c_array(2, 1, 0, 4);
        c_array(2, 1, 2, 5) = 1.5 * c_1_m2nu * e;
        p1 = 4.5 * d * e * h;
        p2 = d_1 * h * (4.3 - 8.0 / 3.0 * nu);
        c_array(2, 1, 0, 6) = p1 + p2;
        c_array(2, 1, 1, 6) = ii * (-p1 + p2);
        c_array(2, 1, 2, 6) = e * (6.0 * c_2_mnu * h2 + c_1_m2nu * d_2);
        p1 = 1.5 * d * e * h * c_d_3h;
        p2 = 1.0 / 3.0 * d_1 * h * 
                ((2.0 * c_2_mnu + 3.3) * h2 + 0.5 * c_3_m4nu * d_2);
        c_array(2, 1, 0, 7) = p1 + p2;
        c_array(2, 1, 1, 7) = ii * (-p1 + p2);
        c_array(2, 1, 2, 7) = e * 
                (3.0 * h2 * c_d_3h - 0.25 * c_1_m2nu * c_d_m3h * c_d_h);
        p0 = h * d_1 * c_d_h;
        p1 = 0.15 * d_2 - 0.95 / 3.0 * h2;
        p2 = 0.25 * e_2 * c_d_m3h;
        c_array(2, 1, 0, 8) = -p0 * (p1 + p2);
        c_array(2, 1, 1, 8) = ii * p0 * (-p1 + p2);
        c_array(2, 1, 2, 8) = -0.5 * e * h2 * c_d_m3h * c_d_h;

        c_array(3, 1, 2, 0) = 6.4 / 3.0 * ii * e_4 * c_1_m2nu * c_tcos_n1;
        p0 = e_2 * d * h;
        p1 = e_2 * (1.4625 + 0.375 * w_c_tcos_n2);
        p2 = 1.25 * (0.15 + c_1_mnu) + 0.5 * c_5_m4nu * tcos_x;
        c_array(3, 1, 0, 1) = -ii * p0 * (p1 - p2);
        c_array(3, 1, 1, 1) = -p0 * (p1 + p2);
        c_array(3, 1, 2, 1) = ii * e_4 *
                (12.8 / 3.0 * c_2_mnu * c_tcos_n1 * h2 - 
                        c_1_m2nu / 3.0 * (5.2 + 0.725 * ii * tan_x +
                                3.2 * tcos_x) * d_2);
        //p1 = ; p2 = ;
        c_array(3, 1, 0, 2) = ii * p0 * (e_2 * (d_2 * (3.275 +
                tan_x * (0.78125 * ii + 0.025 * tan_x) + tcos_x) +
                h2 * (0.86875 + 0.1875 * w_c_tcos_n2)) - 
                d_2 * (c_1_mnu + 0.25 / 3.0 + 
                        0.09375 * ii * c_5_m4nu * tan_x) -
                h2 * (0.0625 * c_5_m2nu * c_3_4tcos - 0.09375));
        c_array(3, 1, 1, 2) = p0 * (e_2 * (d_2 * (3.275 + 
                tan_x * (0.78125 * ii + 0.025 * tan_x) + tcos_x) +
                h2 * (0.86875 + 0.1875 * w_c_tcos_n2)) + 
                d_2 * (c_1_mnu + 0.25 / 3.0 + 
                        0.09375 * ii * c_5_m4nu * tan_x) +
                h2 * (0.0625 * c_5_m2nu * c_3_4tcos - 0.09375));
        c_array(3, 1, 2, 2) = e_4 * 
                (-d_4 * c_1_m2nu * (-0.8 * ii + 0.3625 * tan_x) -
                        0.8 / 3.0 * ii * h4 * c_13_m2nu * c_tcos_n1 +
                        d2h2 / 3.0 * (-c_115_m38nu_80 * tan_x + 
                                0.4 * ii * (1.0 + 8.0 * c_3_mnu +
                                        2.0 * c_7_m2nu * tcos_x)));
        //p1 = ; p2 = ;
        c_array(3, 1, 0, 3) = p0 * ((d2h2 * (0.625 * ii - 0.28125 * tan_x) -
                0.109375 * d_4 * tan_x + 
                0.078125 * ii * h4 * c_3_4tcos) +
                e_2 * (d_4 * (-2.0 * ii + 1.015625 * tan_x) -
                         ii * h4 * (0.234375 + 0.046875 * w_c_tcos_n2) +
                         d2h2 * (0.46875 * tan_x - 
                                 0.125 * ii * (15.0 + 4.0 * tcos_x))));
        c_array(3, 1, 1, 3) = -p0 * ((d2h2 * (0.625 + 0.28125 * ii * tan_x) +
                0.109375 * ii * d_4 * tan_x + 0.078125 * h4 * c_3_4tcos) +
                e_2 * (d_4 * (2.0 + 1.015625 * ii * tan_x) +
                        h4 * (0.234375 + 0.046875 * w_c_tcos_n2) + 
                        d2h2 * (0.46875 * ii * tan_x +
                                0.125 * (15.0 + 4.0 * tcos_x))));
        c_array(3, 1, 2, 3) = e_4 * h2 * 
                (3.0 * d_4 * (-0.8 * ii + 0.3625 * tan_x) +
                        0.8 * ii * h4 * c_tcos_n1 +
                        d2h2 * (0.2375 * tan_x - 0.8 * ii * (2.0 + tcos_x)));
        p0 = 0.25 * h * e_3;
        c_array(3, 1, 0, 5) = p0 * (-3.0 * e_2 + c_5_m4nu);
        c_array(3, 1, 1, 5) = ii * p0 * (3.0 * e_2 + c_5_m4nu);
        c_array(3, 1, 2, 5) = -1.5 * d * e_3 * c_1_m2nu;
        p1 = 3.0 * d_2 * c_9_m8nu + h2 * c_15_m8nu;
        p2 = 9.0 * e_2 * (5.0 * d_2 + h2);
        c_array(3, 1, 0, 6) = p0 * (p1 - p2);
        c_array(3, 1, 1, 6) = ii * p0 * (p1 + p2);
        c_array(3, 1, 2, 6) = -d * e_3 * 
                (6.0 * h2 * c_2_mnu + 5.0 * d_2 * c_1_m2nu);
        p0 = 0.5 * p0;
        p1 = 5.0 * d_4 * c_3_m4nu + 6.0 * d2h2 * c_9_m4nu + h4 * c_15_m4nu;
        p2 = 3.0 * e_2 * (35.0 * d_4 + 30.0 * d2h2 + 3.0 * h4);
        c_array(3, 1, 0, 7) = p0 * (p1 - p2);
        c_array(3, 1, 1, 7) = ii * p0 * (p1 + p2);
        c_array(3, 1, 2, 7) = -d * e_3 *
                     (0.75 * h4 * c_13_m2nu + 2.5 * d2h2 * c_7_m2nu + 
                             1.75 * d_4 * c_1_m2nu);
        p0 = p0 * c_d_h;
        p1 = (-7.0 * d_4 + 22.0 * d2h2 + 5.0 * h4) / 3.0;
        p2 = e_2 * (21.0 * d_4 + 14.0 * d2h2 + h4);
        c_array(3, 1, 0, 8) = p0 * (p1 - p2);
        c_array(3, 1, 1, 8) = ii * p0 * (p1 + p2);
        c_array(3, 1, 2, 8) = -d * e_3 * h2 * c_d_h * (3.5 * d_2 + 1.5 * h2);

        p1 = 1.25 * nu * d_c;
        c_array(4, 1, 0, 1) = h * (2.875 * d_sin_p - ii * p1);
        c_array(4, 1, 1, 1) = h * (p1 - 2.875 * d_cos_p);
        c_array(4, 1, 2, 1) = 0.725 / 3.0 * c_1_m2nu * d_2 * tan_x;
        p0 = 0.0625 * h;
        p1 = 6.0 * nu * ii * h2 - c_5_m2nu * d_2 * tan_x;
        p2 = ii * (3.0 * c_9_m2nu * ii * h2 - 2.0 * nu * d_2 * tan_x);
        c_array(4, 1, 0, 2) = p0 * (p1 * d_cos_p + p2 * d_sin_p);
        c_array(4, 1, 1, 2) = p0 * (p1 * d_sin_p - p2 * d_cos_p);
        c_array(4, 1, 2, 2) = d_2 * tan_x * 
                (0.0375 * c_1_m2nu * d_2 - c_115_m38nu_80 / 3.0 * h2);
        p0 = 0.09375 * h;
        p1 = 5.0 * h4;
        p2 = (-d_4 + 2.0 * d2h2) * tan_x;
        c_array(4, 1, 0, 3) = p0 * (p1 * d_sin_p + p2 * d_cos_p);
        c_array(4, 1, 1, 3) = p0 * (-p1 * d_cos_p + p2 * d_sin_p);
        c_array(4, 1, 2, 3) = d_2 * h2 * tan_x * 
                (0.2375 * h2 - 0.1125 * d_2);
        c_array(4, 1, 2, 4) = -8.0 * c_1_mnu * abh;
        p0 = 1.5 * h;
        p2 = 2.0 * ii * nu;
        c_array(4, 1, 0, 5) = p0 * 
                (-c_5_m2nu * cos_p - p2 * sin_p);
        c_array(4, 1, 1, 5) = p0 * 
                (-c_5_m2nu * sin_p + p2 * cos_p);
        c_array(4, 1, 2, 5) = -1.5 * c_1_m2nu * d_1;
        p1 = 2.0 * h * c_d_3h * nu * e_c;
        p2 = 4.5 * h * (d_2 + 5.0 * h2);
        c_array(4, 1, 0, 6) = p1 - p2 * cos_p;
        c_array(4, 1, 1, 6) = ii * p1 - p2 * sin_p;
        c_array(4, 1, 2, 6) = d_1 * 
                (1.0 / 3.0 * c_1_m2nu * d_2 + (3.6 * c_3_mnu - 0.4) * h2);
        p1 = 0.5 * nu * h * c_d_h * c_d_m3h * e_c;
        p2 = 0.75 * h * (d_4 - 6.0 * d2h2 - 15.0 * h4);
        c_array(4, 1, 0, 7) = -p1 + p2 * cos_p;
        c_array(4, 1, 1, 7) = -ii * p1 + p2 * sin_p;
        c_array(4, 1, 2, 7) = d_1 * (-0.15 * c_1_m2nu * d_4 + 
                1.0 / 6.0 * c_7_m2nu * d2h2 + 
                (0.35 + 1.9 * (8 - nu)) / 3.0 * h4);
        p2 = -0.25 * h * c_d_h * (d_4 - 2.0 * d2h2 + 5.0 * h4);
        c_array(4, 1, 0, 8) = p2 * cos_p;
        c_array(4, 1, 1, 8) = p2 * sin_p;
        c_array(4, 1, 2, 8) = d_1 * h2 * c_d_h * 
                (1.9 / 3.0 * h2 - 0.3 * d_2);

        c_array(5, 1, 2, 0) = 6.4 / 3.0 * ii * c_1_m2nu * e_2 * tcos_x;
        p0 = d * h;
        p1 = 0.1875 + 1.25 * c_1_mnu;
        p2 = e_2 * (1.4375 + 2.5 * tcos_x);
        c_array(5, 1, 0, 1) = ii * p0 * (-p1 + p2);
        c_array(5, 1, 1, 1) = p0 * (p1 + p2);
        c_array(5, 1, 2, 1) = e_2 / 3.0 * 
                (c_1_m2nu * d_2 * (2.6 * ii - 0.725 * tan_x) +
                        12.8 * c_2_mnu * ii * h2 * tcos_x);
        //p1 = ; p2 = ;
        c_array(5, 1, 0, 2) = p0 * (0.09375 * ii * h2 * c_9_m4nu -
                0.03125 * d_2 * c_5_m4nu * tan_x + 
                e_2 * (d_2 * (-3.25 / 3.0 * ii + 0.46875 * tan_x) -
                         0.3125 * ii * h2 * (2.7 + 4.0 * tcos_x)));
        c_array(5, 1, 1, 2) = -p0 * (0.09375 * h2 * c_9_m4nu + 
                0.03125 * ii * d_2 * c_5_m4nu * tan_x +
                e_2 * (d_2 * (3.25 / 3.0 + 0.46875 * ii * tan_x) +
                        0.3125 * h2 * (2.7 + 4.0 * tcos_x)));
        c_array(5, 1, 2, 2) = e_2 * (0.0625 * c_1_m2nu * tan_x * d_4 + 
                (ii * (-5.0 + 1.6 * nu) + c_115_m38nu_80 * tan_x) / 3.0 * d2h2 -
                0.8 / 3.0 * ii * c_13_m2nu * tcos_x * h4);
        //p1 = ; p2 = ;
        c_array(5, 1, 0, 3) = p0 * (-(0.234375 * ii * h4 -
                0.09375 * d2h2 * tan_x + 0.046875 * d_4 * tan_x) +
                e_2 * (0.078125 * ii * h4 * c_3_4tcos +
                        (0.625 * ii - 0.28125 * tan_x) * d2h2 -
                        0.109375 * d_4 * tan_x));
        c_array(5, 1, 1, 3) = p0 * ((0.234375 * h4 +
                0.09375 * ii * d2h2 * tan_x - 0.046875 * ii * d_4 * tan_x) +
                e_2 * (0.078125 * h4 * c_3_4tcos + 
                        (0.625 + 0.28125 * ii * tan_x) * d2h2 +
                        0.109375 * ii * d_4 * tan_x));
        c_array(5, 1, 2, 3) = e_2 * h2 * (-0.1875 * d_4 * tan_x + 
                (0.8 * ii - 0.2375 * tan_x) * d2h2 + 0.8 * ii * h4 * tcos_x);
        p0 = e * h;
        p1 = 1.25 * e_2;
        p2 = 3.75 - 3.0 * nu;
        c_array(5, 1, 0, 5) = p0 * (p1 - p2);
        c_array(5, 1, 1, 5) = -ii * p0 * (p1 + p2);
        c_array(5, 1, 2, 5) = 1.5 * d * e * c_1_m2nu;
        p0 = 0.25 * p0;
        p1 = e_2 * (15.0 * h2 + 27.0 * d_2);
        p2 = 3.0 * c_15_m8nu * h2 + c_9_m8nu * d_2;
        c_array(5, 1, 0, 6) = p0 * (p1 - p2);
        c_array(5, 1, 1, 6) = -ii * p0 * (p1 + p2);
        c_array(5, 1, 2, 6) = d * e * (c_1_m2nu * d_2 + 6.0 * c_2_mnu * h2);
        p0 = 0.5 * p0;
        p1 = c_3_m4nu * d_4 - 2.0 * c_9_m4nu * d2h2 - 3.0 * c_15_m4nu * h4;
        p2 = 3.0 * e_2 * (5.0 * d_4 + 18.0 * d2h2 + 5.0 * h4);
        c_array(5, 1, 0, 7) = p0 * (p1 + p2);
        c_array(5, 1, 1, 7) = -ii * p0 * (-p1 + p2);
        c_array(5, 1, 2, 7) = 0.25 * d * e *
                     (-c_1_m2nu * d_4 + 2.0 * c_7_m2nu * d2h2 + 
                             3.0 * c_13_m2nu * h4);
        p0 = p0 * c_d_h;
        p1 = (-d_4 + 2.0 * d2h2 - 5.0 * h4);
        p2 = e_2 / 3.0 * (-7.0 * d_4 + 22.0 * d2h2 + 5.0 * h4);
        c_array(5, 1, 0, 8) = p0 * (p1 + p2);
        c_array(5, 1, 1, 8) = ii * p0 * (p1 - p2);
        c_array(5, 1, 2, 8) = -0.5 * d * e * h2 * c_d_h * c_d_m3h;


        // S_13 + I * S_23
        
        c_array(0, 2, 1, 1) = -0.5 * nu * e_2 * tcos_x;
        c_array(0, 2, 0, 1) = ii * c_array(0, 2, 1, 1);
        c_array(0, 2, 1, 2) = 0.5 * e_2 * h2 * tcos_x;
        c_array(0, 2, 0, 2) = ii * c_array(0, 2, 1, 2);
        c_array(0, 2, 0, 6) = -0.5 * (c_2_mnu * d_1 + nu * d * e);
        c_array(0, 2, 1, 6) = 0.5 * ii * (-c_2_mnu * d_1 + nu * d * e);
        c_array(0, 2, 2, 6) = -e * h;
        c_array(0, 2, 0, 7) = -h2 * d_1 * (e_2 + 1.0);
        c_array(0, 2, 1, 7) = ii * h2 * d_1 * (e_2 - 1.0);
        c_array(0, 2, 2, 7) = -2.0 * h3 * e;

        c_array(1, 2, 1, 1) = -0.0625 * d * e_2 * nu * c_5_8tcos;
        c_array(1, 2, 0, 1) = ii * c_array(1, 2, 1, 1);
        c_array(1, 2, 2, 1) = -ii * e_2 * h * tcos_x;
        c_array(1, 2, 1, 2) = d * e_2 * (0.03125 * nu * d_2 * c_8_3i_tan +
                h2 * (0.5 + 0.09375 * nu + 0.125 * (6.0 + nu) * tcos_x));
        c_array(1, 2, 0, 2) = ii * c_array(1, 2, 1, 2);
        c_array(1, 2, 2, 2) = ii * e_2 * h3 * tcos_x;
        c_array(1, 2, 1, 3) = -0.09375 * d * e_2 * h2 *
                (d_2 * c_8_3i_tan + h2 * c_3_4tcos);
        c_array(1, 2, 0, 3) = ii * c_array(1, 2, 1, 3);
        c_array(1, 2, 0, 5) = 0.25 * e * (c_2_mnu - nu * e_2);
        c_array(1, 2, 1, 5) = 0.25 * ii * e * (c_2_mnu + nu * e_2);
        p2 = 3.0 * nu * d_2 + c_3_nu * h2;
        c_array(1, 2, 0, 6) = 0.5 * e * (c_5_mnu * h2 - e_2 * p2);
        c_array(1, 2, 1, 6) = 0.5 * ii * e * (c_5_mnu * h2 + e_2 * p2);
        c_array(1, 2, 2, 6) = -d * e * h;
        p2 = e_2 * (0.625 * nu * d_4 + 0.75 * c_6_nu * d2h2 +
                0.125 * c_12_nu * h4);
        c_array(1, 2, 0, 7) = e * (h4 - p2);
        c_array(1, 2, 1, 7) = ii * e * (h4 + p2);
        c_array(1, 2, 2, 7) = -2.0 * d * e * h3;
        c_array(1, 2, 0, 8) = -0.25 * e_3 * h2 * c_d_h * (5.0 * d_2 + h2);
        c_array(1, 2, 1, 8) = -ii * c_array(1, 2, 0, 8);

        c_array(2, 2, 1, 1) = 0.3125 * nu * d;
        c_array(2, 2, 0, 1) = ii * c_array(2, 2, 1, 1);
        c_array(2, 2, 1, 2) = -0.03125 * d * (h2 * (16 + 3.0 * nu) +
                ii * nu * d_2 * tan_x);
        c_array(2, 2, 0, 2) = ii * c_array(2, 2, 1, 2);
        c_array(2, 2, 2, 2) = d_2 / 12.0 * h * tan_x;
        c_array(2, 2, 1, 3) = 0.09375 * d * h2 * (3.0 * h2 + ii * d_2 * tan_x);
        c_array(2, 2, 0, 3) = ii * c_array(2, 2, 1, 3);
        c_array(2, 2, 2, 3) = -0.25 * d_2 * h3 * tan_x;
        p1 = 3.0 * nu * e;
        p2 = c_2_mnu * e_c;
        c_array(2, 2, 0, 5) = 0.25 * (p1 + p2);
        c_array(2, 2, 1, 5) = -0.25 * ii * (p1 - p2);
        p1 = nu * d_2 + 3.0 * c_3_nu * h2;
        p2 = c_5_mnu * h2;
        c_array(2, 2, 0, 6) = 0.5 * (p1 * e + p2 * e_c);
        c_array(2, 2, 1, 6) = -0.5 * ii * (p1 * e - p2 * e_c);
        c_array(2, 2, 2, 6) = -10.0 / 3.0 * d_1 * h;
        p1 = 0.125 * (nu * d_4 - 2.0 * c_6_nu * d2h2 - 3.0 * c_12_nu * h4);
        c_array(2, 2, 0, 7) = -p1 * e + h4 * e_c;
        c_array(2, 2, 1, 7) = ii * (p1 * e + h4 * e_c);
        c_array(2, 2, 2, 7) = -d_1 * h * (d_2 + 11.0 * h2) / 3.0;
        c_array(2, 2, 0, 8) = -0.25 * e * h2 * c_d_m3h * c_d_h;
        c_array(2, 2, 1, 8) = -ii * c_array(2, 2, 0, 8);
        c_array(2, 2, 2, 8) = -2.0 / 3.0 * d_1 * h3 * c_d_h;

        p1 = 0.5 * c_2_mnu * tcos_x;
        p2 = 3.2 / 3.0 * nu * e_2 * c_tcos_n1;
        c_array(3, 2, 0, 0) = ii * e_2 * (p1 + p2);
        c_array(3, 2, 1, 0) = e_2 * (-p1 + p2);
        p1 = 0.5 * c_5_mnu * h2 * tcos_x;
        p2 = e_2 * (-3.2 / 3.0 * c_3_nu * h2 * c_tcos_n1 +
                    nu * d_2 / 3.0 * (2.6 + 0.3625 * ii * tan_x +
                            1.6 * tcos_x));
        c_array(3, 2, 0, 1) = ii * e_2 * (p1 - p2);
        c_array(3, 2, 1, 1) = -e_2 * (p1 + p2);
        c_array(3, 2, 2, 1) = -0.125 * ii * d * e_2 * h * c_5_8tcos;
        p1 = e_2 * (d_4 * nu * (0.4 + 0.18125 * ii * tan_x) -
                    h4 * 0.4 / 3.0 * c_12_nu * c_tcos_n1 +
                    d2h2 * (1.4 + 0.8 / 3.0 * nu +
                            0.4 / 3.0 * c_6_nu * tcos_x +
                            ii * (0.2 + 0.11875 / 3.0 * nu) * tan_x));
        p2 = 0.5 * tcos_x * h4;
        c_array(3, 2, 0, 2) = ii * e_2 * (p1 - p2);
        c_array(3, 2, 1, 2) = e_2 * (p1 + p2);
        c_array(3, 2, 2, 2) = 0.0625 * ii * d * e_2 * h *
                     (d_2 * c_8_3i_tan + h2 * (19.0 + 28.0 * tcos_x));
        p0 = e_4 * h2; //p1 = ; p2 = ;
        c_array(3, 2, 0, 3) = p0 * (d_4 * (-1.2 * ii + 0.54375 * tan_x) +
                0.4 * ii * h4 * c_tcos_n1 +
                d2h2 * (0.11875 * tan_x - 0.4 * ii * (2.0 + tcos_x)));
        c_array(3, 2, 1, 3) = p0 * (d_4 * (-1.2 - 0.54375 * ii * tan_x) +
                0.4 * h4 * c_tcos_n1 +
                d2h2 * (-0.11875 * ii * tan_x - 0.4 * (2.0 + tcos_x)));
        c_array(3, 2, 2, 3) = -0.1875 * ii * d * e_2 * h3 *
                (d_2 * c_8_3i_tan + h2 * c_3_4tcos);
        p1 = 0.25 * c_2_mnu * d * e;
        p2 = 0.75 * nu * d * e_3;
        c_array(3, 2, 0, 5) = p1 - p2;
        c_array(3, 2, 1, 5) = ii * (p1 + p2);
        c_array(3, 2, 2, 5) = -0.5 * e_3 * h;
        p1 = 0.5 * c_5_mnu * d * e * h2;
        p2 = 0.5 * d * e_3 * (5.0 * nu * d_2 + 3.0 * c_3_nu * h2);
        c_array(3, 2, 0, 6) = p1 - p2;
        c_array(3, 2, 1, 6) = ii * (p1 + p2);
        c_array(3, 2, 2, 6) = -e_3 * h * (3.0 * d_2 + 4.0 * h2);
        p1 = d * e * h4;
        p2 = 0.125 * d * e_3 *
             (7.0 * nu * d_4 + 10.0 * c_6_nu * d2h2 + 3.0 * c_12_nu * h4);
        c_array(3, 2, 0, 7) = p1 - p2;
        c_array(3, 2, 1, 7) = ii * (p1 + p2);
        c_array(3, 2, 2, 7) = -0.25 * e_3 * h *
                (5.0 * d_4 + 42.0 * d2h2 + 13.0 * h4);
        c_array(3, 2, 0, 8) = -0.25 * d * e_3 * h2 * c_d_h *
                (7.0 * d_2 + 3.0 * h2);
        c_array(3, 2, 1, 8) = -ii * c_array(3, 2, 0, 8);
        c_array(3, 2, 2, 8) = -0.5 * e_3 * h3 * c_d_h * (5.0 * d_2 + h2);

        c_array(4, 2, 1, 0) = 0.5 * c_2_mnu * e_2_c * tcos_c;
        c_array(4, 2, 0, 0) = -ii * c_array(4, 2, 1, 0);
        p1 = 0.3625 / 3.0 * nu * d_2 * tan_x;
        p2 = 0.5 * c_5_mnu * h2 * e_2_c * tcos_c;
        c_array(4, 2, 0, 1) = p1 - ii * p2;
        c_array(4, 2, 1, 1) = -ii * p1 + p2;
        p1 = (0.01875 * nu * d_2 -
                h2 * (0.2 + 0.11875 / 3.0 * nu)) * d_2 * tan_x;
        p2 = 0.5 * h4 * e_2_c * tcos_c;
        c_array(4, 2, 0, 2) = p1 + ii * p2;
        c_array(4, 2, 1, 2) = -ii * p1 - p2;
        c_array(4, 2, 0, 3) = d_2 * h2 *
                (-0.05625 * d_2 + 0.11875 * h2) * tan_x;
        c_array(4, 2, 1, 3) = -ii * c_array(4, 2, 0, 3);
        c_array(4, 2, 0, 4) = -2.0 * c_1_nu * abh;
        c_array(4, 2, 1, 4) = -ii * c_array(4, 2, 0, 4);
        p1 = 0.75 * nu * d_1;
        p2 = 0.25 * c_2_mnu * d_c * e_c;
        c_array(4, 2, 0, 5) = -p1 + p2;
        c_array(4, 2, 1, 5) = ii * (p1 + p2);
        p1 = d_1 * (0.5 / 3.0 * nu * d_2 + h2 * (4.3 + 0.9 * nu));
        p2 = 0.5 * c_5_mnu * h2 * d_c * e_c;
        c_array(4, 2, 0, 6) = p1 + p2;
        c_array(4, 2, 1, 6) = -ii * (p1 - p2);
        p1 = d_1 * (-0.075 * nu * d_4 + 0.25 / 3.0 * c_6_nu * d2h2 +
                   h4 / 3.0 * (7.3 + 0.475 * nu));
        p2 = h4 * d_c * e_c;
        c_array(4, 2, 0, 7) = p1 + p2;
        c_array(4, 2, 1, 7) = -ii * (p1 - p2);
        c_array(4, 2, 0, 8) = d_1 * h2 * c_d_h *
                (-0.15 * d_2 + 0.95 / 3.0 * h2);
        c_array(4, 2, 1, 8) = -ii * c_array(4, 2, 0, 8);

        c_array(5, 2, 1, 0) = 3.2 / 3.0 * nu * e_2 * tcos_x;
        c_array(5, 2, 0, 0) = ii * c_array(5, 2, 1, 0);
        p1 = 0.125 / 3.0 * c_2_mnu * d_2 * tan_x;
        p2 = e_2 / 3.0 * (nu * d_2 * (1.3 + 0.3625 * ii * tan_x) +
                3.2 * c_3_nu * h2 * tcos_x);
        c_array(5, 2, 0, 1) = p1 + ii * p2;
        c_array(5, 2, 1, 1) = ii * p1 + p2;
        p1 = 0.125 / 3.0 * c_5_mnu * d_2 * h2 * tan_x;
        p2 = e_2 * (0.03125 * nu * d_4 * tan_x +
                d2h2 / 3.0 * (-ii * (2.1 + 0.4 * nu) +
                        (0.6 + 0.11875 * nu) * tan_x) -
                0.4 / 3.0 * ii * h4 * c_12_nu * tcos_x);
        c_array(5, 2, 0, 2) = -p1 + p2;
        c_array(5, 2, 1, 2) = -ii * (p1 + p2);
        p1 = 0.125 * d_2 * h4 * tan_x;
        p2 = e_2 * h2 * (0.4 * tcos_x * h4 +
                (0.4 + 0.11875 * ii * tan_x) * d2h2 +
                0.09375 * ii * tan_x * d_4);
        c_array(5, 2, 0, 3) = p1 + ii * p2;
        c_array(5, 2, 1, 3) = ii * p1 + p2;
        c_array(5, 2, 0, 4) = -c_3_mnu * abh;
        c_array(5, 2, 1, 4) = ii * c_array(5, 2, 0, 4);
        p1 = 0.25 * c_2_mnu * d_1;
        p2 = 0.75 * nu * d * e;
        c_array(5, 2, 0, 5) = -p1 + p2;
        c_array(5, 2, 1, 5) = -ii * (p1 + p2);
        p1 = d_1 * (0.75 * (6.0 - nu) * h2 + 0.25 / 3.0 * c_2_mnu * d_2);
        p2 = 0.5 * d * e * (nu * d_2 + 3.0 * c_3_nu * h2);
        c_array(5, 2, 0, 6) = p1 + p2;
        c_array(5, 2, 1, 6) = ii * (p1 - p2);
        p1 = 1.0 / 6.0 * d_1 * h2 * ((15.0 - nu) * h2 + c_5_mnu * d_2);
        p2 = 0.125 * d * e *
                (3.0 * c_12_nu * h4 + 2.0 * c_6_nu * d2h2 - nu * d_4);
        c_array(5, 2, 0, 7) = p1 + p2;
        c_array(5, 2, 1, 7) = ii * (p1 - p2);
        p0 = h2 * c_d_h;
        p1 = 1.0 / 3.0 * h2 * d_1;
        p2 = 0.25 * d * e * c_d_m3h;
        c_array(5, 2, 0, 8) = p0 * (p1 - p2);
        c_array(5, 2, 1, 8) = ii * p0 * (p1 + p2);

        c_array(5, 2, 2, 1) = 0.625 * ii * h * d;
        c_array(5, 2, 2, 2) = 0.0625 * h * d * (-19.0 * ii * h2 + d_2 * tan_x);
        c_array(5, 2, 2, 3) = 0.1875 * h3 * d * (3.0 * ii * h2 - d_2 * tan_x);
        c_array(5, 2, 2, 5) = 1.5 * h * e;
        c_array(5, 2, 2, 6) = (d_2 + 12.0 * h2) * h * e;
        c_array(5, 2, 2, 7) = 0.25 * (-d_4 +
                14.0 * d2h2 + 39.0 * h4) * h * e;
        c_array(5, 2, 2, 8) = -0.5 * e * h3 * c_d_h * c_d_m3h;

        for (int j = 0; j < c_array.size(3); ++j) {
            c_array(4, 2, 2, j) = std::conj(c_array(5, 2, 2, j));
        }


        // S_33
        
        c_array(0, 3, 0, 6) = -2.0 * h * cos_p;
        c_array(0, 3, 1, 6) = -2.0 * h * sin_p;
        c_array(0, 3, 2, 6) = -2.0 * d_1;
        c_array(0, 3, 0, 7) = -4.0 * h3 * cos_p;
        c_array(0, 3, 1, 7) = -4.0 * h3 * sin_p;
        c_array(0, 3, 2, 7) = 4.0 * h2 * d_1;

        c_array(1, 3, 1, 1) = -e_2 * h * tcos_x;
        c_array(1, 3, 0, 1) = ii * c_array(1, 3, 1, 1);
        p1 = 1.0 / 12.0 * tan_x * d_2;
        p2 = e_2 * h2 * tcos_x;
        c_array(1, 3, 0, 2) = (p1 + ii * p2) * h;
        c_array(1, 3, 1, 2) = (ii * p1 + p2) * h;
        c_array(1, 3, 0, 3) = -0.25 * d_2 * h3 * tan_x;
        c_array(1, 3, 1, 3) = ii * c_array(1, 3, 0, 3);
        c_array(1, 3, 2, 5) = e;
        p2 = 10.0 / 3.0 * d_1;
        c_array(1, 3, 0, 6) = (-d * e - p2) * h;
        c_array(1, 3, 1, 6) = ii * (d * e - p2) * h;
        c_array(1, 3, 2, 6) = -4.0 * h2 * e;
        p0 = d_1 * h;
        p1 = (d_2 + 11.0 * h2) / 3.0;
        p2 = 2.0 * e_2 * h2;
        c_array(1, 3, 0, 7) = -(p1 + p2) * p0;
        c_array(1, 3, 1, 7) = -ii * (p1 - p2) * p0;
        c_array(1, 3, 2, 7) = -4.0 * h4 * e;
        c_array(1, 3, 0, 8) = -2.0 / 3.0 * h3 * d_1 * c_d_h;
        c_array(1, 3, 1, 8) = ii * c_array(1, 3, 0, 8);

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(2, 3, j, k) = std::conj(c_array(1, 3, j, k));
            }
        }

        c_array(3, 3, 2, 0) = 2.0 * ii * e_2 * tcos_x;
        p0 = d * h;
        p2 = e_2 * (0.625 + tcos_x);
        c_array(3, 3, 0, 1) = ii * p0 * (0.625 - p2);
        c_array(3, 3, 1, 1) = -p0 * (0.625 + p2);
        c_array(3, 3, 2, 1) = -4.0 * ii * e_2 * h2 * tcos_x;
        p0 = 0.0625 * d * h;
        p1 = -19.0 * ii * h2 + d_2 * tan_x;
        p2 = e_2 * (d_2 * c_8_3i_tan + h2 * (19.0 + 28.0 * tcos_x));
        c_array(3, 3, 0, 2) = p0 * (p1 + ii * p2);
        c_array(3, 3, 1, 2) = p0 * (ii * p1 + p2);
        c_array(3, 3, 2, 2) = 2.0 * ii * e_2 * h4 * tcos_x;
        p0 = 0.1875 * d * h3;
        p1 = 3.0 * ii * h2 - d_2 * tan_x;
        p2 = e_2 * (d_2 * c_8_3i_tan + h2 * c_3_4tcos);
        c_array(3, 3, 0, 3) = p0 * (p1 - ii * p2);
        c_array(3, 3, 1, 3) = p0 * (ii * p1 - p2);
        p0 = h * e;
        c_array(3, 3, 0, 5) = p0 * (-0.5 * e_2 + 1.5);
        c_array(3, 3, 1, 5) = ii * p0 * (0.5 * e_2 + 1.5);
        c_array(3, 3, 2, 5) = d * e;
        p1 = d_2 + 12.0 * h2;
        p2 = e_2 * (3.0 * d_2 + 4.0 * h2);
        c_array(3, 3, 0, 6) = p0 * (p1 - p2);
        c_array(3, 3, 1, 6) = ii * p0 * (p1 + p2);
        c_array(3, 3, 2, 6) = -4.0 * h2 * d * e;
        p1 = -0.25 * d_4 + 3.5 * d_2 * h2 + 9.75 * h4;
        p2 = e_2 * (1.25 * d_4 + 10.5 * d_2 * h2 + 3.25 * h4);
        c_array(3, 3, 0, 7) = p0 * (p1 - p2);
        c_array(3, 3, 1, 7) = ii * p0 * (p1 + p2);
        c_array(3, 3, 2, 7) = -4.0 * h4 * d * e;
        p0 = h3 * e * c_d_h;
        p1 = 0.5 * c_d_m3h;
        p2 = e_2 * (2.5 * d_2 + 0.5 * h2);
        c_array(3, 3, 0, 8) = -p0 * (p1 + p2);
        c_array(3, 3, 1, 8) = ii * p0 * (-p1 + p2);

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(4, 3, j, k) = std::conj(c_array(3, 3, j, k));
            }
        }

        c_array(5, 3, 0, 1) = -1.25 * h * d_sin_p;
        c_array(5, 3, 1, 1) = 1.25 * h * d_cos_p;
        c_array(5, 3, 2, 1) = 1.0 / 6.0 * d_2 * tan_x;
        p1 = 0.125 * d_2 * tan_x;
        p2 = 2.375 * h2;
        c_array(5, 3, 0, 2) = h * (p1 * d_cos_p + p2 * d_sin_p);
        c_array(5, 3, 1, 2) = h * (p1 * d_sin_p - p2 * d_cos_p);
        c_array(5, 3, 2, 2) = 1.0 / 3.0 * d_2 * h2 * tan_x;
        p1 = 3.0 * p1;
        p2 = 1.125 * h2;
        c_array(5, 3, 0, 3) = -h3 * (p1 * d_cos_p + p2 * d_sin_p);
        c_array(5, 3, 1, 3) = -h3 * (p1 * d_sin_p - p2 * d_cos_p);
        c_array(5, 3, 2, 3) = -0.5 * d_2 * h4 * tan_x;
        c_array(5, 3, 0, 5) = 3.0 * h * cos_p;
        c_array(5, 3, 1, 5) = 3.0 * h * sin_p;
        c_array(5, 3, 2, 5) = -d_1;
        p0 = 2.0 * h * (d_2 + 12.0 * h2);
        c_array(5, 3, 0, 6) = p0 * cos_p;
        c_array(5, 3, 1, 6) = p0 * sin_p;
        c_array(5, 3, 2, 6) = (1.0 / 3.0 * d_2 - 9.0 * h2) * d_1;
        p0 = h * (-0.5 * d_4 + 7.0 * d_2 * h2 + 19.5 * h4);
        c_array(5, 3, 0, 7) = p0 * cos_p;
        c_array(5, 3, 1, 7) = p0 * sin_p;
        c_array(5, 3, 2, 7) = -h2 * (4.0 / 3.0 * d_2 + 8.0 * h2) * d_1;
        p0 = h3 * c_d_h * c_d_m3h;
        c_array(5, 3, 0, 8) = -p0 * cos_p;
        c_array(5, 3, 1, 8) = -p0 * sin_p;
        c_array(5, 3, 2, 8) = -4.0 / 3.0 * h4 * c_d_h * d_1;

        return c_array;
    }

// Additional terms for a special case:
// reduced summation; collocation point projected onto
// an edge line or a vertex of the element

    il::StaticArray4D<std::complex<double>, 6, 4, 3, 5> s_ij_red_h(double nu, std::complex<double> eix,double h) {
        // const std::complex<double> I(0.0, 1.0);

        double c_1_nu = 1.0 + nu;
        double c_1_2nu = 1.0 + 2.0 * nu;
        double c_2_nu = 2.0 + nu;
        double c_3_nu = 3.0 + nu;
        double c_3_2nu = 3.0 + 2.0 * nu;
        double c_7_2nu = 7.0 + 2.0 * nu;
        double c_11_4nu = 11.0 + 4.0 * nu;
        double c_12_nu = 12.0 + nu;
        double c_13_2nu = 13.0 + 2.0 * nu;

        double c_1_mnu = 1.0 - nu;
        double c_1_m2nu = 1.0 - 2.0 * nu;
        double c_2_mnu = 2.0 - nu;
        double c_3_mnu = 3.0 - nu;
        double c_5_mnu = 5.0 - nu;
        double c_5_m4nu = 5.0 - 4.0 * nu;
        double c_13_m2nu = 13.0 - 2.0 * nu;
        double c_15_m4nu = 15.0 - 4.0 * nu;
        double c_15_m8nu = 15.0 - 8.0 * nu;

        double cos_x = std::real(eix);
        double sin_x = std::imag(eix);
        std::complex<double> e2x = eix * eix;
        std::complex<double> e3x = e2x * eix;
        std::complex<double> emx = std::conj(eix);
        std::complex<double> em2 = std::conj(e2x);
        std::complex<double> c_e3x_3emx = 3.0 * emx + e3x;
        std::complex<double> c_eix_3_1 = eix * (3.0 + e2x);
        std::complex<double> c_eix_3_m1 = eix * (3.0 - e2x);

        double h2 = h * h;
        double h3 = h2 * h;
        double h4 = h2 * h2;
        double h5 = h4 * h;
        double h6 = h4 * h2;
        double h7 = h5 * h2;
        double sgh = ((h < 0) ? -1.0 : static_cast<double>((h > 0))); // sign(h)
        double abh = std::fabs(h);

        std::complex<double> p1, p2, p3, p4;

        il::StaticArray4D<std::complex<double>, 6, 4, 3, 5> c_array{0.0};

        // S_11 + S_22
        
        c_array(0, 0, 0, 2) = -c_7_2nu * h * sin_x;
        c_array(0, 0, 1, 2) = c_7_2nu * h * cos_x;
        c_array(0, 0, 0, 3) = -9.0 * h3 * sin_x;
        c_array(0, 0, 1, 3) = 9.0 * h3 * cos_x;
        c_array(0, 0, 0, 4) = -1.5 * h5 * sin_x;
        c_array(0, 0, 1, 4) = 1.5 * h5 * cos_x;

        c_array(1, 0, 0, 0) = -0.5 * ii * c_1_nu * e2x * sgh;
        c_array(1, 0, 1, 0) = -0.5 * c_1_nu * e2x * sgh;
        c_array(1, 0, 2, 1) = 0.5 * ii * c_1_2nu * eix;
        c_array(1, 0, 2, 2) = 2.0 * ii * c_2_nu * h2 * eix;
        c_array(1, 0, 2, 3) = 2.0 * ii * h4 * eix;

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(2, 0, j, k) = std::conj(c_array(1, 0, j, k));
            }
        }

        p1 = 0.25 * c_3_2nu * h;
        p2 = 0.25 * c_11_4nu * h3;
        p3 = 0.125 * c_13_2nu * h5;
        p4 = 0.625 * h7;

        c_array(3, 0, 2, 0) = -2.0 * ii * c_1_nu * e2x * abh;
        c_array(3, 0, 0, 1) = -ii * p1 * c_eix_3_1;
        c_array(3, 0, 1, 1) = p1 * c_eix_3_m1;
        c_array(3, 0, 0, 2) = -ii * p2 * c_eix_3_1;
        c_array(3, 0, 1, 2) = p2 * c_eix_3_m1;
        c_array(3, 0, 0, 3) = -ii * p3 * c_eix_3_1;
        c_array(3, 0, 1, 3) = p3 * c_eix_3_m1;
        c_array(3, 0, 0, 4) = -ii * p4 / 3.0 * c_eix_3_1;
        c_array(3, 0, 1, 4) = p4 / 3.0 * c_eix_3_m1;

        for (int k = 0; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2); ++j) {
                c_array(4, 0, j, k) = std::conj(c_array(3, 0, j, k));
            }
        }

        c_array(5, 0, 0, 1) = 6.0 * p1 * sin_x;
        c_array(5, 0, 1, 1) = -6.0 * p1 * cos_x;
        c_array(5, 0, 0, 2) = 6.0 * p2 * sin_x;
        c_array(5, 0, 1, 2) = -6.0 * p2 * cos_x;
        c_array(5, 0, 0, 3) = 6.0 * p3 * sin_x;
        c_array(5, 0, 1, 3) = -6.0 * p3 * cos_x;
        c_array(5, 0, 0, 4) = 2.0 * p4 * sin_x;
        c_array(5, 0, 1, 4) = -2.0 * p4 * cos_x;


        // S11 - S_22 + 2 * I * S_12
        
        c_array(0, 1, 2, 0) = -ii * nu * e2x / abh;
        c_array(0, 1, 0, 2) = 0.5 * ii * h * (3.0 * c_eix_3_1 - 4.0 * nu * eix);
        c_array(0, 1, 1, 2) = -0.5 * h * (3.0 * c_eix_3_m1 - 4.0 * nu * eix);
        c_array(0, 1, 0, 3) = 1.5 * ii * h3 * c_eix_3_1;
        c_array(0, 1, 1, 3) = -1.5 * h3 * c_eix_3_m1;
        c_array(0, 1, 0, 4) = 0.25 * ii * h5 * c_eix_3_1;
        c_array(0, 1, 1, 4) = -0.25 * h5 * c_eix_3_m1;

        p1 = 0.5 * ii * c_1_m2nu * eix;
        p2 = 2.0 * ii * c_2_mnu * h2 * eix;
        p3 = 0.25 * ii * c_13_m2nu * h4 * eix;
        p4 = 0.5 * ii * h2 * h4 * eix;

        c_array(1, 1, 0, 0) = -ii * sgh * e2x * (c_1_mnu + 0.5 * e2x);
        c_array(1, 1, 1, 0) = sgh * e2x * (c_1_mnu - 0.5 * e2x);
        c_array(1, 1, 2, 1) = p1 * e2x;
        c_array(1, 1, 2, 2) = p2 * e2x;
        c_array(1, 1, 2, 3) = p3 * e2x;
        c_array(1, 1, 2, 4) = p4 * e2x;

        c_array(2, 1, 1, 0) = -sgh * e2x;
        c_array(2, 1, 0, 0) = ii * c_array(2, 1, 1, 0);
        c_array(2, 1, 2, 1) = 3.0 * p1;
        c_array(2, 1, 2, 2) = 3.0 * p2;
        c_array(2, 1, 2, 3) = 3.0 * p3;
        c_array(2, 1, 2, 4) = 3.0 * p4;

        p1 = 0.25 * h * eix;
        p2 = 0.25 * h3 * eix;
        p3 = 0.125 * h5 * eix;
        p4 = 0.125 * h7 * eix;

        c_array(3, 1, 2, 0) = -2.0 * ii * c_1_mnu * abh * e2x * e2x;
        c_array(3, 1, 0, 1) = -ii * e2x * p1 * (c_5_m4nu + 3.0 * e2x);
        c_array(3, 1, 0, 2) = -ii * e2x * p2 * (c_15_m8nu + 9.0 * e2x);
        c_array(3, 1, 0, 3) = -ii * e2x * p3 * (c_15_m4nu + 9.0 * e2x);
        c_array(3, 1, 0, 4) = -ii * e2x * p4 * (5.0 / 3.0 + e2x);

        c_array(3, 1, 1, 1) = e2x * p1 * (c_5_m4nu - 3.0 * e2x);
        c_array(3, 1, 1, 2) = e2x * p2 * (c_15_m8nu - 9.0 * e2x);
        c_array(3, 1, 1, 3) = e2x * p3 * (c_15_m4nu - 9.0 * e2x);
        c_array(3, 1, 1, 4) = e2x * p4 * (5.0 / 3.0 - e2x);

        c_array(5, 1, 2, 0) = -4.0 * ii * c_1_mnu * abh * e2x;
        c_array(5, 1, 0, 1) = -ii * p1 * (3.0 * c_5_m4nu + 5.0 * e2x);
        c_array(5, 1, 0, 2) = -3.0 * ii * p2 * (c_15_m8nu + 5.0 * e2x);
        c_array(5, 1, 0, 3) = -3.0 * ii * p3 * (c_15_m4nu + 5.0 * e2x);
        c_array(5, 1, 0, 4) = -5.0 * ii * p4 * (1.0 + e2x / 3.0);

        c_array(5, 1, 1, 1) = p1 * (3.0 * c_5_m4nu - 5.0 * e2x);
        c_array(5, 1, 1, 2) = 3.0 * p2 * (c_15_m8nu - 5.0 * e2x);
        c_array(5, 1, 1, 3) = 3.0 * p3 * (c_15_m4nu - 5.0 * e2x);
        c_array(5, 1, 1, 4) = 5.0 * p4 * (1.0 - e2x / 3.0);

        p1 = std::conj(p1); p2 = std::conj(p2); p3 = std::conj(p3);

        c_array(4, 1, 0, 1) = 3.0 * ii * p1 * (c_5_m4nu - 5.0 * e2x);
        //c_array(4, 1, 0, 1) = -0.75*I*h*(5.0*eix-c_5_m4nu*emx);
        c_array(4, 1, 1, 1) = -3.0 * p1 * (c_5_m4nu + 5.0 * e2x);
        //c_array(4, 1, 1, 1) = -0.75*h*(5.0*eix+c_5_m4nu*emx);
        c_array(4, 1, 0, 2) = 3.0 * ii * p2 * (c_15_m8nu - 15.0 * e2x);
        //c_array(4, 1, 0, 2) = -0.75*I*h3*(15.0*eix-c_15_m8nu*emx);
        c_array(4, 1, 1, 2) = -3.0 * p2 * (c_15_m8nu + 15.0 * e2x);
        //c_array(4, 1, 1, 2) = -0.75*h3*(15.0*eix+c_15_m8nu*emx);
        c_array(4, 1, 0, 3) = 3.0 * ii * p3 * (c_15_m4nu - 15.0 * e2x);
        //c_array(4, 1, 0, 3) = -0.375*I*h5*(15.0*eix-c_15_m4nu*emx);
        c_array(4, 1, 1, 3) = -3.0 * p3 * (c_15_m4nu + 15.0 * e2x);
        //c_array(4, 1, 1, 3) = -0.375*h5*(15.0*eix+c_15_m4nu*emx);
        c_array(4, 1, 0, 4) = 1.25 * h7 * sin_x;
        c_array(4, 1, 1, 4) = -1.25 * h7 * cos_x;
        
        
        // S_13 + S_23
        
        c_array(0, 2, 1, 0) = -0.25 * c_1_mnu * e2x / abh;
        c_array(0, 2, 0, 0) = ii * c_array(0, 2, 1, 0);
        c_array(0, 2, 2, 2) = -ii * h * eix;
        c_array(0, 2, 2, 3) = -2.0 * ii * h3 * eix;

        p1 = 0.25 * nu * c_e3x_3emx;
        p2 = 0.5 * h2 * c_3_nu * c_e3x_3emx;
        p3 = 0.125 * h4 * c_12_nu * c_e3x_3emx;
        // p4 = 0.25*h6*c_e3x_3emx;

        c_array(1, 2, 0, 1) = 0.25 * ii * eix * (c_2_mnu + nu * e2x);
        c_array(1, 2, 1, 1) = -0.25 * eix * (c_2_mnu - nu * e2x);
        c_array(1, 2, 0, 2) = 0.5 * ii * h2 * eix * (c_5_mnu + c_3_nu * e2x);
        c_array(1, 2, 1, 2) = -0.5 * h2 * eix * (c_5_mnu - c_3_nu * e2x);
        c_array(1, 2, 0, 3) = 0.125 * ii * h4 * eix * (8.0 + c_12_nu * e2x);
        c_array(1, 2, 1, 3) = -0.125 * h4 * eix * (8.0 - c_12_nu * e2x);
        c_array(1, 2, 1, 4) = 0.25 * h6 * e3x;
        c_array(1, 2, 0, 4) = ii * c_array(1, 2, 1, 4);

        c_array(2, 2, 0, 1) = std::conj(c_array(1, 2, 0, 1) - ii * p1);
        c_array(2, 2, 1, 1) = std::conj(p1 - c_array(1, 2, 1, 1));
        c_array(2, 2, 0, 2) = std::conj(c_array(1, 2, 0, 2) - ii * p2);
        c_array(2, 2, 1, 2) = std::conj(p2 - c_array(1, 2, 1, 2));
        c_array(2, 2, 0, 3) = std::conj(c_array(1, 2, 0, 3) - ii * p3);
        c_array(2, 2, 1, 3) = std::conj(p3 - c_array(1, 2, 1, 3));
        c_array(2, 2, 1, 4) = 0.75 * eix * h6;
        c_array(2, 2, 0, 4) = ii * c_array(2, 2, 1, 4);

        p1 = 0.5 * ii * h * eix;
        p2 = 4.0 * ii * h3 * eix;
        p3 = 3.25 * ii * h5 * eix;
        p4 = 0.5 * ii * h7 * eix;

        c_array(5, 2, 1, 0) = -c_1_nu * abh * e2x;
        c_array(5, 2, 0, 0) = ii * c_array(5, 2, 1, 0);
        c_array(5, 2, 2, 1) = 3.0 * p1;
        c_array(5, 2, 2, 2) = 3.0 * p2;
        c_array(5, 2, 2, 3) = 3.0 * p3;
        c_array(5, 2, 2, 4) = 3.0 * p4;

        c_array(3, 2, 1, 0) = 0.5 * (c_3_mnu - c_1_nu * e2x) * abh * e2x;
        c_array(3, 2, 0, 0) = -0.5 * ii * (c_3_mnu + c_1_nu * e2x) * abh * e2x;
        c_array(3, 2, 2, 1) = e2x * p1;
        c_array(3, 2, 2, 2) = e2x * p2;
        c_array(3, 2, 2, 3) = e2x * p3;
        c_array(3, 2, 2, 4) = e2x * p4;

        c_array(4, 2, 1, 0) = -0.5 * c_3_mnu * abh * em2;
        c_array(4, 2, 0, 0) = -ii * c_array(4, 2, 1, 0);
        c_array(4, 2, 2, 1) = std::conj(c_array(5, 2, 2, 1));
        c_array(4, 2, 2, 2) = std::conj(c_array(5, 2, 2, 2));
        c_array(4, 2, 2, 3) = std::conj(c_array(5, 2, 2, 3));
        c_array(4, 2, 2, 4) = std::conj(c_array(5, 2, 2, 4));

        
        // S_33
        
        c_array(0, 3, 0, 2) = 2.0 * h * sin_x;
        c_array(0, 3, 1, 2) = -2.0 * h * cos_x;
        c_array(0, 3, 0, 3) = 4.0 * h3 * sin_x;
        c_array(0, 3, 1, 3) = -4.0 * h3 * cos_x;

        c_array(1, 3, 2, 1) = ii * eix;
        c_array(1, 3, 2, 2) = -4.0 * ii * h2 * eix;
        c_array(1, 3, 2, 3) = -4.0 * ii * h4 * eix;

        for (int j = 1; j < c_array.size(3) - 1; ++j) {
            c_array(2, 3, 2, j) = std::conj(c_array(1, 3, 2, j));
        }

        c_array(3, 3, 0, 1) = 0.5 * ii * h * c_eix_3_1;
        c_array(3, 3, 1, 1) = -0.5 * h * c_eix_3_m1;
        c_array(3, 3, 0, 2) = 4.0 * ii * h3 * c_eix_3_1;
        c_array(3, 3, 1, 2) = -4.0 * h3 * c_eix_3_m1;
        c_array(3, 3, 0, 3) = 3.25 * ii * h5 * c_eix_3_1;
        c_array(3, 3, 1, 3) = -3.25 * h5 * c_eix_3_m1;
        c_array(3, 3, 0, 4) = 0.5 * ii * h7 * c_eix_3_1;
        c_array(3, 3, 1, 4) = -0.5 * h7 * c_eix_3_m1;

        for (int k = 1; k < c_array.size(3); ++k) {
            for (int j = 0; j < c_array.size(2) - 1; ++j) {
                c_array(4, 3, j, k) = std::conj(c_array(3, 3, j, k));
            }
        }

        c_array(5, 3, 0, 1) = -3.0 * h * sin_x;
        c_array(5, 3, 1, 1) = 3.0 * h * cos_x;
        c_array(5, 3, 0, 2) = -24.0 * h3 * sin_x;
        c_array(5, 3, 1, 2) = 24.0 * h3 * cos_x;
        c_array(5, 3, 0, 3) = -19.5 * h5 * sin_x;
        c_array(5, 3, 1, 3) = 19.5 * h5 * cos_x;
        c_array(5, 3, 0, 4) = -3.0 * h7 * sin_x;
        c_array(5, 3, 1, 4) = 3.0 * h7 * cos_x;

        return c_array;
    }

// Limit case (h==0, plane) - all stress components

    il::StaticArray3D<std::complex<double>, 6, 4, 3> s_ij_lim_h(double nu, std::complex<double> eix,std::complex<double> d) {
        // const std::complex<double> I(0.0, 1.0);

        double c_1_2nu = 1.0 + 2.0 * nu;
        double c_1_m2nu = 1.0 - 2.0 * nu;
        double c_2_mnu = 2.0 - nu;

        double cos_x = std::real(eix);
        double sin_x = std::imag(eix);
        // double tan_x = sin_x/cos_x;
        std::complex<double> e2x = eix * eix;
        double h0_lim = std::atanh(sin_x); // std::complex<double> ?
        // double h0_lim = 0.5*(std::log(1.0+sin_x)-std::log(1.0-sin_x));

        double d_1 = std::abs(d); 
        // double d_2 = d_1*d_1; double d_4 = d_2*d_2;
        std::complex<double> e = std::polar(1.0, std::arg(d)), // = d/d_1
                e_2 = e * e, 
                e_3 = e * e_2, 
                e_4 = e_2 * e_2,
                p1, p2;

        il::StaticArray3D<std::complex<double>, 6, 4, 3> c_array{0.0};

        il::StaticArray<std::complex<double>, 6> v1{}, v2{};

        v1[0] = sin_x / d_1;
        v1[1] = h0_lim * e;
        v1[2] = std::conj(v1[1]);
        v1[3] = d * e * (h0_lim + 2.0 * ii * eix);
        v1[4] = std::conj(v1[3]);
        v1[5] = -h0_lim * d_1;

        v2[0] = 0.5 * e_2 / d_1 * (sin_x - 2.0 * ii * eix * cos_x * cos_x);
        v2[1] = -0.125 * e_3 * (4.0 * h0_lim + ii * eix * (e2x + 8.0));
        v2[2] = 0.125 * e * (12.0 * h0_lim + 5.0 * ii * eix);
        v2[3] = -0.5 * e_4 * d_1 * (3.0 * h0_lim - 2.0 * ii * eix * (e2x - 3.0));
        v2[4] = -1.5 * d_1 * h0_lim;
        v2[5] = 1.5 * e_2 * d_1 * (h0_lim + 2.0 * ii * eix);

        for (int j = 0; j < c_array.size(0); ++j) {
            c_array(j, 0, 2) = 0.5 * c_1_2nu * v1[j];
            c_array(j, 1, 2) = c_1_m2nu * v2[j];
            p1 = 0.25 * c_2_mnu * v1[j];
            p2 = 0.5 * nu * v2[j];
            c_array(j, 2, 0) = p1 + p2;
            c_array(j, 2, 1) = ii * (p1 - p2);
            c_array(j, 3, 2) = v1[j];
        }

        return c_array;
    }
}
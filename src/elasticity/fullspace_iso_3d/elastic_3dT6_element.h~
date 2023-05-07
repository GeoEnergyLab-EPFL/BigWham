//
// This file part of BigWham
//
// Created by D. Nikolski on 1/12/2017.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2017.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//

#ifndef INC_HFPX3D_ELAST_KER_INT_H
#define INC_HFPX3D_ELAST_KER_INT_H

#include <complex>
#include <il/StaticArray.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/Array2D.h>
#include <il/StaticArray2D.h>

#include "core/oldies/FaceData.h"
#include "core/ElasticProperties.h"

namespace bie {

// Element-to-point influence matrix (submatrix of the global one)
// (Integration of a kernel of the elasticity equation over a triangular element
// with 2nd order polynomial approximating (shape) functions)
    il::StaticArray2D<double, 6, 18> make_local_3dbem_submatrix(const int kernel_id,double shear_m, double poiss_r, double h, std::complex<double> z,
             const il::StaticArray<std::complex<double>, 3> &tau,
             const il::StaticArray2D<std::complex<double>, 6, 6> &sfm);

// Integration of a kernel of the elasticity equation
// over a part of a polygonal element (a sector associated with one edge)
// with 2nd order polynomial approximating (shape) functions.

// Coefficient matrices (rank 3) to be contracted with the vector of
// constituing functions defined below (via right multiplication)
// and with the vector of shape function coefficients
// associated with each node of the element (via left multiplication)
//
// Stress components (vs local Cartesian coordinate system of the element)
// combined as S11+S22, S11-S22+2*I*S12, S13+I*S23, S33

    il::StaticArray4D<std::complex<double>, 6, 4, 3, 9> s_integral_gen(const int ker,double poiss_r, std::complex<double> eix,double h, std::complex<double> d);

    il::StaticArray4D<std::complex<double>, 6, 4, 3, 5> s_integral_red(const int kernel_id,double poiss_r, std::complex<double> eix,double h);

    il::StaticArray3D<std::complex<double>, 6, 4, 3> s_integral_lim(const int ker,double poiss_r, std::complex<double> eix,std::complex<double> d);

// Constituing functions for the integrals
// of any kernel of the elasticity equation
// over a part of a polygonal element.
// Example of usage:
// dot(S11_22H(poiss_r, eix, h, d), integral_cst_fun(h, d, a, x, eix))
// dot(S11_22H_red(poiss_r, eip, h, d), integral_cst_fun_red(h, d, a))
// dot(S13_23T(poiss_r, eix, h, d), integral_cst_fun(h, d, a, x, eix))
// dot(S33T_red(poiss_r, eip, h, d), integral_cst_fun_red(h, d, a))
// where eip = std::exp(I*std::arg(t-z));
// eix = std::exp(I*x); x = std::arg(t-z)-std::arg(d);
// a = std::fabs(t-z-d)*sign(x);

    il::StaticArray<std::complex<double>, 9> integral_cst_fun(double h, std::complex<double> d, double a,double x, std::complex<double> eix);

    il::StaticArray<std::complex<double>, 5> integral_cst_fun_red(double h, std::complex<double> d, double a);

/// Function to assemble by Nodes - required for hmat
// todo move to StaticArray
il::Array2D<double> traction_influence_3DT6(bie::FaceData &elem_data_s, bie::FaceData &elem_data_r,
                                            il::int_t n_s,il::int_t  n_t, bie::ElasticProperties const &elas_,
                                            il::int_t I_want_global_DD,il::int_t I_want_global_traction);

}

#endif //INC_HFPX3D_ELAST_KER_INT_H

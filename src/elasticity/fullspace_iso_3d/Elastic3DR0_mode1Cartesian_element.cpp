//
// This file is part of HFPx3D.
//
// Created by Carlo Peruzzo on 01.03.21.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//

#include <il/blas.h>
#include <il/linearAlgebra.h>
#include "elasticity/3d/Elastic3DR0_common.h"
#include "elasticity/3d/Elastic3DR0_mode1Cartesian_element.h"
#include <cmath>
#include <iostream>
#include <limits>

namespace bie{
    // RONGVED SOLUTION FOR A P0 Rectangular dislocation in a full space
    // dislocation is centered on the origin in the plane z=0 , (-a,a) in x (-b,b)
    // in y

    // Fundamental stress kernel
    double StressesKernelR0opening(
            double& x, double& y, double& z, double& a, double& b, double& G,
            double& nu) {
        // Assuming z=0
        // x , y , z location where to compute stress
        //  a,b  1/2 size of the rectangular DD
        //  G Shear modulus, nu Poisson's ratio'
        //  Rectangular DDon plan z=0   x \in [-a,a], y \in [-b,b]
        //  DDz (normal)

        double Stress;
        double EPSILON;
        EPSILON = 100000 * std::numeric_limits<double>::epsilon();

        if (!is_stress_singular_at_given_location(x, y, z ,a, b))
        {
            double Ce = G / (4. * il::pi * (1. - nu));
            // compute the Is function derivatives....

            double Ip33;

            // Assuming z=0
            // the following is needed because of numerical errors in some cases
            // we are using the symmetry of the function Ip33
            if ((il::abs(x)/a)<10.){
                double abs_y = il::abs(y);
                Ip33 = rectangular_integration(x, abs_y, z, a, b, ip33);
            } else{
                Ip33 = rectangular_integration(x, y, z, a, b, ip33);
            }

            // the following limits are needed to consider the cases where we evalute the kernel at the prolongation of an edge of one element
            double Ip33_lim_z_to_0;
            if (il::abs(il::abs(x) / a - 1.) <= EPSILON && (il::abs(y) > b))        // Assuming z=0
                Ip33_lim_z_to_0 = Ip33_lim_z_to_0_and_x_to_a(x, y, a, b);
            else if (il::abs(il::abs(y) / b - 1.) <= EPSILON && (il::abs(x) > a))        // Assuming z=0
                Ip33_lim_z_to_0 = Ip33_lim_z_to_0_and_y_to_b(x, y, a, b);
            else
                Ip33_lim_z_to_0 = Ip33;

            // stress due to displacement discontinuity DDz (normal)
            Stress = Ce * (Ip33_lim_z_to_0);   // szz
        }
        else { Stress = NAN;}
        return Stress;
        // DDz (normal) -> szz
    }


    double traction_influence_3DR0opening(
        FaceData &elem_data_s, // source element
        FaceData &elem_data_r, // receiver element
        ElasticProperties const &elas_ // elastic properties
        )
    {

        double G = elas_.shear_modulus(), nu = elas_.poisson_ratio();
        il::StaticArray<double,2> a_and_b = get_a_and_b(elem_data_s.getVertices(),elem_data_s.getNoV());
        double a = a_and_b[0], b = a_and_b[1];

        il::Array2D<double> el_cp_s;
        el_cp_s = elem_data_s.collocation_points();

        il::Array2D<double> el_cp_r;
        el_cp_r = elem_data_r.collocation_points();

        il::Array<double> dsr{3};
        for (int i = 0; i < 3; ++i) { dsr[i] = el_cp_r(0,i) - el_cp_s(0,i);}

        double Stress;

        Stress = StressesKernelR0opening(dsr[0],
                                  dsr[1],
                                  dsr[2],
                                  a, b,
                                  G,nu);

        // in the reference system of the source element both in the domain and in the codomain
        // szz

        return Stress; // thisis coincident with the traction on the plane  t3/Dnormal

    }

}

//
// This file part of BigWham
//
// Created by Brice Lecampion on 04.02.19.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//


#include <cmath>
#include <limits>
#include <iostream>
#include <il/linearAlgebra.h>
#include <il/blas.h>
#include "elasticity/3d/Elastic3DR0_common.h"
#include "elasticity/3d/Elastic3DR0_element.h"


namespace bie{
    // RONGVED SOLUTION FOR A P0 Rectangular dislocation in a full space
    // dislocation is centered on the origin in the plane z=0 , (-a,a) in x (-b,b)
    // in y


    il::StaticArray2D<double, 3, 6> StressesKernelR0(
            double& x, double& y, double& z, double& a, double& b, double& G,
            double& nu) {
        // x , y , z location where to compute stress
        //  a,b  1/2 size of the rectangular DD
        //  G Shear modulus, nu Poisson's ratio'
        //  Rectangular DDon plan z=0   x \in [-a,a], y \in [-b,b]
        //  DDx (shear), DDy (shear), DDz (normal)

        il::StaticArray2D<double, 3, 6> Stress;
        double EPSILON;
        EPSILON = 100000 * std::numeric_limits<double>::epsilon();

        if (!is_stress_singular_at_given_location(x, y, z ,a, b, true))
        {

            double Ce = G / (4. * il::pi * (1. - nu));
            //  double sxx, sxy, sxz, syy, syz, szz;
            //

            // compute the Is function derivatives....

            double Ip11, Ip22, Ip33, Ip23, Ip12, Ip13;

            double Ip111, Ip122, Ip133, Ip112, Ip113, Ip123, Ip222, Ip223, Ip233, Ip333 = 0.;
            if (il::abs(z)<0.01)
            {
                if ((il::abs(x)/a)<10.){
                    double abs_y = il::abs(y);
                    Ip11 = rectangular_integration(x, abs_y, z, a, b, ip11);
                    Ip33 = rectangular_integration(x, abs_y, z, a, b, ip33);

                    if (y < 0.){Ip12 = - rectangular_integration(x, abs_y, z, a, b, ip12);} else { Ip12 = rectangular_integration(x, abs_y, z, a, b, ip12);}

                    Ip13 = rectangular_integration(x, abs_y, z, a, b, ip13);

                    Ip111 = rectangular_integration(x, abs_y, z, a, b, ip111);
                    Ip122 = rectangular_integration(x, abs_y, z, a, b, ip122);
                    Ip133 = rectangular_integration(x, abs_y, z, a, b, ip133);

                    if (y < 0.){Ip112 = -rectangular_integration(x, abs_y, z, a, b, ip112);} else { Ip112 = rectangular_integration(x, abs_y, z, a, b, ip112);}

                    Ip113 = rectangular_integration(x, abs_y, z, a, b, ip113);

                    if (y < 0.){Ip123 = -rectangular_integration(x, abs_y, z, a, b, ip123);} else { Ip123 = rectangular_integration(x, abs_y, z, a, b, ip123);}

                    // Ip333 <--- this is non trivial that it goes to 0 when z->0 i.e. limit to be taken
                } else{

                    Ip11 = rectangular_integration(x, y, z, a, b, ip11);
                    Ip33 = rectangular_integration(x, y, z, a, b, ip33);
                    Ip12 = rectangular_integration(x, y, z, a, b, ip12);
                    Ip13 = rectangular_integration(x, y, z, a, b, ip13);

                    Ip111 = rectangular_integration(x, y, z, a, b, ip111);
                    Ip122 = rectangular_integration(x, y, z, a, b, ip122);
                    Ip133 = rectangular_integration(x, y, z, a, b, ip133);

                    Ip112 = rectangular_integration(x, y, z, a, b, ip112);
                    Ip113 = rectangular_integration(x, y, z, a, b, ip113);
                    Ip123 = rectangular_integration(x, y, z, a, b, ip123);

                    // Ip333 <--- this is non trivial that it goes to 0 when z->0 i.e. limit to be taken
                }


                if ((il::abs(y)/b)<10.){
                    double abs_x = il::abs(x);
                    Ip22 = rectangular_integration(abs_x , y, z, a, b, ip22);
                    Ip23 = rectangular_integration(abs_x , y, z, a, b, ip23);
                    Ip222 = rectangular_integration(abs_x, y, z, a, b, ip222);
                    Ip233 = rectangular_integration(abs_x, y, z, a, b, ip233);
                    Ip223 = rectangular_integration(abs_x, y, z, a, b, ip223);
                }
                else{
                    Ip22 = rectangular_integration(x, y, z, a, b, ip22);
                    Ip23 = rectangular_integration(x, y, z, a, b, ip23);
                    Ip222 = rectangular_integration(x, y, z, a, b, ip222);
                    Ip233 = rectangular_integration(x, y, z, a, b, ip233);
                    Ip223 = rectangular_integration(x, y, z, a, b, ip223);
                }
            }
            else{
                Ip11 = rectangular_integration(x, y, z, a, b, ip11);
                Ip22 = rectangular_integration(x, y, z, a, b, ip22);
                Ip33 = rectangular_integration(x, y, z, a, b, ip33);
                Ip23 = rectangular_integration(x, y, z, a, b, ip23);
                Ip12 = rectangular_integration(x, y, z, a, b, ip12);
                Ip13 = rectangular_integration(x, y, z, a, b, ip13);

                Ip111 = rectangular_integration(x, y, z, a, b, ip111);
                Ip122 = rectangular_integration(x, y, z, a, b, ip122);
                Ip133 = rectangular_integration(x, y, z, a, b, ip133);
                Ip112 = rectangular_integration(x, y, z, a, b, ip112);
                Ip113 = rectangular_integration(x, y, z, a, b, ip113);
                Ip123 = rectangular_integration(x, y, z, a, b, ip123);
                Ip222 = rectangular_integration(x, y, z, a, b, ip222);
                Ip233 = rectangular_integration(x, y, z, a, b, ip233);
                Ip223 = rectangular_integration(x, y, z, a, b, ip223);
                // Ip333 <--- this is non trivial that it goes to 0 when z->0 i.e. limit to be taken
            }
//            Ip11 = rectangular_integration(x, y, z, a, b, ip11);
//            Ip22 = rectangular_integration(x, y, z, a, b, ip22);
//            Ip33 = rectangular_integration(x, y, z, a, b, ip33);
//            Ip23 = rectangular_integration(x, y, z, a, b, ip23);
//            Ip12 = rectangular_integration(x, y, z, a, b, ip12);
//            Ip13 = rectangular_integration(x, y, z, a, b, ip13);
//
//            Ip111 = rectangular_integration(x, y, z, a, b, ip111);
//            Ip122 = rectangular_integration(x, y, z, a, b, ip122);
//            Ip133 = rectangular_integration(x, y, z, a, b, ip133);
//            Ip112 = rectangular_integration(x, y, z, a, b, ip112);
//            Ip113 = rectangular_integration(x, y, z, a, b, ip113);
//            Ip123 = rectangular_integration(x, y, z, a, b, ip123);
//            Ip222 = rectangular_integration(x, y, z, a, b, ip222);
//            Ip233 = rectangular_integration(x, y, z, a, b, ip233);
//            Ip223 = rectangular_integration(x, y, z, a, b, ip223);
            // Ip333 <--- this is non trivial that it goes to 0 when z->0 i.e. limit to be taken

            double Ip33_lim_z_to_0, z_times_Ip333;

            if (il::abs(z) <= EPSILON )
                z_times_Ip333 = 0.;
            else{ Ip333 = rectangular_integration(x, y, z, a, b, ip333);
                  z_times_Ip333 = z * Ip333;}

            if (il::abs(il::abs(x) / a - 1.) <= EPSILON && (il::abs(y) > b) && il::abs(z) <= EPSILON)
                Ip33_lim_z_to_0 = Ip33_lim_z_to_0_and_x_to_a(x, y, a, b);
            else if (il::abs(il::abs(y) / b - 1.) <= EPSILON && (il::abs(x) > a) && il::abs(z) <= EPSILON)
                Ip33_lim_z_to_0 = Ip33_lim_z_to_0_and_y_to_b(x, y, a, b);
            else
                Ip33_lim_z_to_0 = Ip33;

            // Stress row is dof (DDx,DDy,DDx), columns are sxx,syy,szz,sxy,sxz,syz

            // stress due to displacement discontinuity DDx (shear)
            Stress(0, 0) = Ce * (2. * Ip13 - z * Ip111);                    // sxx -> if z=0. it will be 0. (unchanged expresssion)
            Stress(0, 1) = Ce * (2.   * nu * Ip13 - z * Ip122);             // syy -> if z=0. it will be 0. (unchanged expresssion)
            Stress(0, 2) = Ce * (-z * Ip133);                               // szz -> if z=0. it will be 0. (unchanged expresssion)
            Stress(0, 3) = Ce * ((1. - nu) * Ip23 - z * Ip112);             // sxy -> if z=0. it will be 0. (unchanged expresssion)
            Stress(0, 4) = Ce * (Ip33_lim_z_to_0 + nu * Ip22 - z * Ip113);  // sxz
            Stress(0, 5) = Ce * (-nu * Ip12 - z * Ip123);                   // syz if z=0. (unchanged expresssion)

            // stress due to displacement discontinuity  DDy (shear)

            Stress(1, 0) = Ce * (2. * nu * Ip23 - z * Ip112);               // sxx -> if z=0. it will be 0. (unchanged expresssion)
            Stress(1, 1) = Ce * (2. * Ip23 - z * Ip222);                    // syy -> if z=0. it will be 0. (unchanged expresssion)
            Stress(1, 2) = Ce * (-z * Ip233);                               // szz -> if z=0. it will be 0. (unchanged expresssion)
            Stress(1, 3) = Ce * ((1. - nu) * Ip13 - z * Ip122);             // sxy -> if z=0. it will be 0. (unchanged expresssion)
            Stress(1, 4) = Stress(0, 5);                             // sxz -> if z=0. (unchanged expresssion)
            Stress(1, 5) = Ce * (Ip33_lim_z_to_0 + nu * Ip11 - z * Ip223);  // syz

            // stress due to displacement discontinuity DDz (normal)
            Stress(2, 0) = Ce * (Ip33_lim_z_to_0 + (1 - 2 * nu) * Ip22 - z * Ip113); // sxx
            Stress(2, 1) = Ce * (Ip33_lim_z_to_0 + (1 - 2 * nu) * Ip11 - z * Ip223); // syy
            Stress(2, 2) = Ce * (Ip33_lim_z_to_0 - z_times_Ip333);                   // szz
            Stress(2, 3) = Ce * ((-1 + 2 * nu) * Ip12 - z * Ip123);                  // sxy if z=0. (unchanged expresssion)
            Stress(2, 4) = Ce * (-z * Ip133);                                        // sxz -> if z=0. it will be 0. (unchanged expresssion)
            Stress(2, 5) = Ce * (-z * Ip233);                                        // syz -> if z=0. it will be 0. (unchanged expresssion)

//            for (il::int_t i = 0; i < 3; i++) {
//                for (il::int_t j = 0; j < 6; j++) {
//                    if (std::isnan(Stress(i,j))){
//                        printf("found NAN");
//                    }
//                }
//            }

        }
        else {  for (il::int_t i = 0; i < 3; i++) {
                    for (il::int_t j = 0; j < 6; j++) {
                        Stress(i,j) = NAN;
                     }
                 }
        }
        return Stress;
        // DDx (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDy (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDz (normal) -> | sxx, syy, szz, sxy, sxz, syz  |
    }


    // Fundamental displacement kernel
    il::Array2D<double> DisplacementKernelR0(
            double& x, double& y, double& z, double& a, double& b,
            double& nu) {
        //  x , y , z location where to compute displacement
        //  a,b  1/2 size of the rectangular DD
        //  nu Poisson's ratio'
        //  Rectangular DDon plan z=0   x \in [-a,a], y \in [-b,b]
        //  DDx (shear), DDy (shear), DDz (normal)

        double Ip1, Ip2, Ip3_out_plane_z_EQ_0;

        double Ip11, Ip22, Ip33, Ip23, Ip12, Ip13;

        double Ce = (-1. / (8. * il::pi * (1. - nu)));

        il::Array2D<double> Displacement{3,3,0.};

        // compute the Is function derivatives....
        Ip3_out_plane_z_EQ_0 = rectangular_integration(x, y, z, a, b, ip3);


        if (il::abs(z)<0.01)
        {
            if ((il::abs(x)/a)<10.){
                double abs_y = il::abs(y);
                Ip1 = rectangular_integration(x, abs_y, z, a, b, ip1);
                Ip11 = rectangular_integration(x, abs_y, z, a, b, ip11);
                Ip33 = rectangular_integration(x, abs_y, z, a, b, ip33);
                if (y < 0.){Ip12 = - rectangular_integration(x, abs_y, z, a, b, ip12);} else { Ip12 = rectangular_integration(x, abs_y, z, a, b, ip12);}
                Ip13 = rectangular_integration(x, abs_y, z, a, b, ip13);
            }else{
                Ip1 = rectangular_integration(x, y, z, a, b, ip1);
                Ip11 = rectangular_integration(x, y, z, a, b, ip11);
                Ip33 = rectangular_integration(x, y, z, a, b, ip33);
                Ip12 = rectangular_integration(x, y, z, a, b, ip12);
                Ip13 = rectangular_integration(x, y, z, a, b, ip13);
            }

            if ((il::abs(y)/b)<10.){
                double abs_x = il::abs(x);
                Ip2 = rectangular_integration(abs_x , y, z, a, b, ip2);
                Ip22 = rectangular_integration(abs_x , y, z, a, b, ip22);
                Ip23 = rectangular_integration(abs_x , y, z, a, b, ip23);
            }else{
                Ip2 = rectangular_integration(x, y, z, a, b, ip2);
                Ip22 = rectangular_integration(x, y, z, a, b, ip22);
                Ip23 = rectangular_integration(x, y, z, a, b, ip23);
            }

        }
        else{
            Ip1 = rectangular_integration(x, y, z, a, b, ip1);
            Ip2 = rectangular_integration(x, y, z, a, b, ip2);
            Ip11 = rectangular_integration(x, y, z, a, b, ip11);
            Ip22 = rectangular_integration(x, y, z, a, b, ip22);
            Ip33 = rectangular_integration(x, y, z, a, b, ip33);
            Ip23 = rectangular_integration(x, y, z, a, b, ip23);
            Ip12 = rectangular_integration(x, y, z, a, b, ip12);
            Ip13 = rectangular_integration(x, y, z, a, b, ip13);
        }

//        Ip1 = rectangular_integration(x, y, z, a, b, ip1);
//        Ip2 = rectangular_integration(x, y, z, a, b, ip2);
//        Ip11 = rectangular_integration(x, y, z, a, b, ip11);
//        Ip22 = rectangular_integration(x, y, z, a, b, ip22);
//        Ip33 = rectangular_integration(x, y, z, a, b, ip33);
//        Ip23 = rectangular_integration(x, y, z, a, b, ip23);
//        Ip12 = rectangular_integration(x, y, z, a, b, ip12);
//        Ip13 = rectangular_integration(x, y, z, a, b, ip13);

        double Ip3 = get_Ip3 ( x, y, z, a, b, Ip3_out_plane_z_EQ_0);

        // Displacement row is dof (DDx,DDy,DDx), columns are Ux,Uy,Uz in the local reference system

        // displacement due to displacement discontinuity DDx (shear)
        Displacement(0, 0) = Ce * (z * Ip11 - 2 * (1 - nu) * Ip3);  // Ux
        Displacement(1, 0) = Ce * (z * Ip12);                       // Uy -> if z=0. it will be 0. (unchanged expresssion)
        Displacement(2, 0) = Ce * (z * Ip13 - (1 - 2 * nu) * Ip1);  // Uz

        // displacement due to displacement discontinuity  DDy (shear)
        Displacement(0, 1) = Displacement(1, 0);             // Ux  -> if z=0. it will be 0. (unchanged expresssion)
        Displacement(1, 1) = Ce * (z * Ip22 - 2 * (1 - nu) * Ip3);  // Uy
        Displacement(2, 1) = Ce * (z * Ip23 - (1 - 2 * nu) * Ip2);  // Uz

        // displacement due to displacement discontinuity DDz (normal)
        Displacement(0, 2) = Ce * (z * Ip13 + (1 - 2 * nu) * Ip1);  // Ux
        Displacement(1, 2) = Ce * (z * Ip23 + (1 - 2 * nu) * Ip2);  // Uy
        Displacement(2, 2) = Ce * (z * Ip33 - 2 * (1 - nu) * Ip3);  // Uz

//        Uncomment to check nan
//        for (il::int_t i = 0; i < 3; i++) {
//            for (il::int_t j = 0; j < 3; j++) {
//                if (std::isnan(Displacement(i,j))){
//                    printf("found NAN");
//                }
//            }
//        }
        return Displacement; // expressed in the reference system of the DD element
        // index        ->    DDx (shear)    DDy (shear)     DDz (normal)
        //   0      -> |       Ux,            Ux,             Ux            |
        //   1      -> |       Uy,            Uy,             Uy            |
        //   2      -> |       Uz,            Uz,             Uz            |
    }

    il::Array2D<double> traction_influence_3DR0(
        FaceData &elem_data_s, // source element
        FaceData &elem_data_r, // receiver element
        ElasticProperties const &elas_, // elastic properties
        il::int_t I_want_global_DD = 0,
        il::int_t I_want_global_traction = 0)
    {

        double G = elas_.shear_modulus(), nu = elas_.poisson_ratio();
        il::StaticArray<double,2> a_and_b = get_a_and_b(elem_data_s.getVertices(),elem_data_s.getNoV());
        double a = a_and_b[0], b = a_and_b[1];

        il::Array2D<double> el_cp_s;
        el_cp_s = elem_data_s.collocation_points();

        il::Array2D<double> el_cp_r;
        el_cp_r = elem_data_r.collocation_points();

        il::Array2D<double> R = elem_data_s.rotationMatrix(true); // from global to local

        il::Array<double> dsr{3};
        for (int i = 0; i < 3; ++i) { dsr[i] = el_cp_r(0,i) - el_cp_s(0,i);}

        // dsr contains the component of the distance between the source and the receiver
        dsr = il::dot(R, dsr);

        il::StaticArray2D<double, 3, 6> Stress;

        Stress = StressesKernelR0(dsr[0],
                                  dsr[1],
                                  dsr[2],
                                  a, b,
                                  G,nu);

        // in the reference system of the source element both in the domain and in the codomain
        // index        ->    0    1    2    3    4    5
        // DDx (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDy (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDz (normal) -> | sxx, syy, szz, sxy, sxz, syz  |

        // normal vector at the receiver location in the reference system of the source element
        il::Array<double> nr = elem_data_r.getNormal();
        nr = il::dot(R,nr);

        il::Array<double> traction_temp;
        il::Array2D<double> DDs_to_traction_local_local{3,3,0.0}, sigma_temp{3,3,0.0};

        for (int i = 0; i < 3; ++i)
        { //loop over the rows of Stress
                sigma_temp(0,0) = Stress(i,0); // sig_xx
                sigma_temp(0,1) = Stress(i,3); // sig_xy
                sigma_temp(0,2) = Stress(i,4); // sig_xz
                sigma_temp(1,0) = Stress(i,3); // sig_yx
                sigma_temp(1,1) = Stress(i,1); // sig_yy
                sigma_temp(1,2) = Stress(i,5); // sig_yz
                sigma_temp(2,0) = Stress(i,4); // sig_zx
                sigma_temp(2,1) = Stress(i,5); // sig_zy
                sigma_temp(2,2) = Stress(i,2); // sig_zz

                traction_temp = il::dot(sigma_temp, nr);
                for (int j = 0; j < 3; ++j) {
                    DDs_to_traction_local_local(j,i) = traction_temp[j];
                    // | t1/Dshear1   t1/Dshear2  t1/Dnormal |
                    // | t2/Dshear1   t2/Dshear2  t2/Dnormal |
                    // | t3/Dshear1   t3/Dshear2  t3/Dnormal |
                    // localDD & local traction
                    // in the reference system of the source element both in the domain and in the codomain
                }
        }

        return change_ref_system(DDs_to_traction_local_local, I_want_global_DD, I_want_global_traction, R, elem_data_r.rotationMatrix(true));

        // | t1/Dshear1   t1/Dshear2  t1/Dnormal |
        // | t2/Dshear1   t2/Dshear2  t2/Dnormal |
        // | t3/Dshear1   t3/Dshear2  t3/Dnormal |
        //
        // directions 1, 2 or 3
    }

    il::Array2D<double> displacement_influence_3DR0(
            FaceData &elem_data_s, // source element
            FaceData &elem_data_r, // receiver element
            ElasticProperties const &elas_, // elastic properties
            il::int_t I_want_global_DD,
            il::int_t I_want_global_displacement)
    {

        il::StaticArray<double,2> a_and_b = get_a_and_b(elem_data_s.getVertices(),elem_data_s.getNoV());
        double a = a_and_b[0], b = a_and_b[1];
        double nu = elas_.poisson_ratio();

        il::Array2D<double> el_cp_s;
        el_cp_s = elem_data_s.collocation_points();

        il::Array2D<double> el_cp_r;
        el_cp_r = elem_data_r.collocation_points();

        il::Array2D<double> R;
        R = elem_data_s.rotationMatrix(true);

        il::Array<double> dsr{3};
        for (int i = 0; i < 3; ++i) { dsr[i] = el_cp_r(0,i) - el_cp_s(0,i); }

        // dsr contains the component of the distance between the source and the receiver
        // in the reference system of the source element
        dsr = il::dot(R, dsr);

        // displacement in the referece system local to the source element
        il::Array2D<double> DDs_to_Displacement_local_local =DisplacementKernelR0(dsr[0],
                                                                               dsr[1],
                                                                               dsr[2],
                                                                               a, b,
                                                                               nu);
        // index        ->    DDx (shear)    DDy (shear)     DDz (normal)
        //   0      -> |       Ux,            Ux,             Ux            |
        //   1      -> |       Uy,            Uy,             Uy            |
        //   2      -> |       Uz,            Uz,             Uz            |

        return change_ref_system(DDs_to_Displacement_local_local, I_want_global_DD, I_want_global_displacement, R, elem_data_r.rotationMatrix(true));

        // | U1/Dshear1   U1/Dshear2  U1/Dnormal |
        // | U2/Dshear1   U2/Dshear2  U2/Dnormal |
        // | U3/Dshear1   U3/Dshear2  U3/Dnormal |
        //
        // directions 1, 2 or 3
    }

    il::Array<double> point_stress_3DR0(
            il::Array<double> &observ_pt,
            FaceData &elem_data_s, // source element
            il::Array<double> &dd,
            ElasticProperties const &elas_ // elastic properties
            )
    {
        /*
         * It returns the stress components:
         * sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz
         * expressed in the global reference system
         *
         */
        double G = elas_.shear_modulus(), nu = elas_.poisson_ratio();
        il::StaticArray<double,2> a_and_b = get_a_and_b(elem_data_s.getVertices(),elem_data_s.getNoV());
        double a = a_and_b[0], b = a_and_b[1];

        il::Array2D<double> el_cp_s;
        el_cp_s = elem_data_s.collocation_points();

        il::Array2D<double> R = elem_data_s.rotationMatrix(true); // R(g->l)

        il::Array<double> dsr{3};
        for (int i = 0; i < 3; ++i) { dsr[i] = observ_pt[i] - el_cp_s(0,i);}

        // dsr contains the component of the distance between the source and the receiver
        dsr = il::dot(R, dsr);

        il::StaticArray2D<double, 3, 6> Stress = StressesKernelR0(dsr[0],
                                                                  dsr[1],
                                                                  dsr[2],
                                                                  a, b,
                                                                  G,nu);
        // Attention!
        // It is in the reference system of the source element both in the domain and in the codomain
        // index        ->    0    1    2    3    4    5
        // DDx (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDy (shear)  -> | sxx, syy, szz, sxy, sxz, syz  |
        // DDz (normal) -> | sxx, syy, szz, sxy, sxz, syz  |

        il::Array2D<double> stress_local_2_local{3,3};
        stress_local_2_local(0,0) = Stress(0,0) * dd[0] + Stress(1,0) * dd[1] + Stress(2,0) * dd[2] ; // sxx
        stress_local_2_local(0,1) = Stress(0,3) * dd[0] + Stress(1,3) * dd[1] + Stress(2,3) * dd[2] ; // sxy
        stress_local_2_local(0,2) = Stress(0,4) * dd[0] + Stress(1,4) * dd[1] + Stress(2,4) * dd[2] ; // sxz
        stress_local_2_local(1,0) = stress_local_2_local(0,1); // syx = sxy
        stress_local_2_local(1,1) = Stress(0,1) * dd[0] + Stress(1,1) * dd[1] + Stress(2,1) * dd[2] ; // syy
        stress_local_2_local(1,2) = Stress(0,5) * dd[0] + Stress(1,5) * dd[1] + Stress(2,5) * dd[2] ; // syz
        stress_local_2_local(2,0) = stress_local_2_local(0,2); // szx = sxz
        stress_local_2_local(2,1) = stress_local_2_local(1,2); // szy = syz
        stress_local_2_local(2,2) = Stress(0,2) * dd[0] + Stress(1,2) * dd[1] + Stress(2,2) * dd[2] ; // szz

        il::Array2D<double> RT = transpose(R); // R(l->g)
        // the matrix RT will rotate any vector from the local coordinate system to the global one

        il::Array2D<double> stress_global_2_global = il::dot(RT, il::dot(stress_local_2_local, R));

        il::Array<double> stress_at_point{6};
        stress_at_point[0] = stress_global_2_global(0,0) ; // sxx
        stress_at_point[1] = stress_global_2_global(1,1) ; // syy
        stress_at_point[2] = stress_global_2_global(2,2) ; // szz
        stress_at_point[3] = stress_global_2_global(0,1) ; // sxy
        stress_at_point[4] = stress_global_2_global(0,2) ; // sxz
        stress_at_point[5] = stress_global_2_global(1,2) ; // syz

        return stress_at_point;
    }

    il::Array<double> point_displacement_3DR0(
            il::Array<double> &observ_pt,
            FaceData &elem_data_s, // source element
            il::Array<double> &dd,
            ElasticProperties const &elas_ // elastic properties
            )
    {
        /*
         * It returns the displacement components:
         * u_xx, u_yy, u_zz
         * expressed in the global reference system
         *
         */

        il::StaticArray<double,2> a_and_b = get_a_and_b(elem_data_s.getVertices(),elem_data_s.getNoV());
        double a = a_and_b[0], b = a_and_b[1];
        double nu = elas_.poisson_ratio();

        il::Array2D<double> el_cp_s;
        el_cp_s = elem_data_s.collocation_points();

        il::Array2D<double> R = elem_data_s.rotationMatrix(true); // R(g->l)

        il::Array<double> dsr{3};
        for (int i = 0; i < 3; ++i) { dsr[i] = observ_pt[i] - el_cp_s(0,i); }


        dsr = il::dot(R, dsr);
        // after being rotated dsr contains the component of the distance between the source and the receiver
        // in the reference system of the source element


        // displacement in the referece system local to the source element
        il::Array2D<double> DDs_to_Displacement_local_local =DisplacementKernelR0(dsr[0],
                                                                                  dsr[1],
                                                                                  dsr[2],
                                                                                  a, b,
                                                                                  nu);
        // Attention!
        // It is in the reference system of the source element both in the domain and in the codomain
        // index        ->    DDx (shear)    DDy (shear)     DDz (normal)
        //   0      -> |       Ux,            Ux,             Ux            |
        //   1      -> |       Uy,            Uy,             Uy            |
        //   2      -> |       Uz,            Uz,             Uz            |

        // Apply immediately the DD in order to get a displacement vector in the reference system local to the source element
        il::Array<double> displacement_at_point_local = il::dot(DDs_to_Displacement_local_local,dd);

        il::Array2D<double> RT = transpose(R);
        // Get the displacements in the global reference system
        il::Array<double> displacement_at_point = il::dot(RT,displacement_at_point_local);

        return displacement_at_point;
    }
}

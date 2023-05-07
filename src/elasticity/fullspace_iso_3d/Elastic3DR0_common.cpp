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


namespace bie{
    // RONGVED SOLUTION FOR A P0 Rectangular dislocation in a full space
    // dislocation is centered on the origin in the plane z=0 , (-a,a) in x (-b,b)
    // in y

    //--------------------------------------------------------------------------------------------------//
    //                                                                                                  //
    //  Special case to treat any case when z = 0.                                                      //
    //  i.e. when the point where we ask for the coordinates lies on the plane of the element z == 0    //
    //                                                                                                  //
    //  PLEASE: do note that any stress is theoretically singular when evaluated at the edges of an     //
    //  element.                                                                                        //
    //                                                                                                  //
    //--------------------------------------------------------------------------------------------------//

    double Ip33_lim_z_to_0_and_x_to_a(double& x, double& y, double& a, double& b) {
        // -(Sqrt[(a + x)^2 + (-b + y)^2]/((a + x) (-b + y))) + Sqrt[(a +
        //    x)^2 + (b + y)^2]/((a + x) (b + y))
        double a_plus_x, b_plus_y, y_minus_b, sqrt_1st, sqrt_2nd;
        a_plus_x = a + x ;
        b_plus_y = b + y ;
        y_minus_b = y - b ;
        sqrt_1st = sqrt(a_plus_x * a_plus_x + y_minus_b * y_minus_b);
        sqrt_2nd = sqrt(a_plus_x * a_plus_x + b_plus_y * b_plus_y);

        return - sqrt_1st / (a_plus_x * y_minus_b) + sqrt_2nd / (a_plus_x * b_plus_y);
    }

    double Ip33_lim_z_to_0_and_y_to_b(double& x, double& y, double& a, double& b) {
        // -(Sqrt[(-a + x)^2 + (b + y)^2]/((-a + x) (b + y))) + Sqrt[(a +
        //    x)^2 + (b + y)^2]/((a + x) (b + y))
        double a_plus_x, b_plus_y, x_minus_a, sqrt_1st, sqrt_2nd;
        a_plus_x = a + x ;
        b_plus_y = b + y ;
        x_minus_a = x - a ;
        sqrt_1st = sqrt(b_plus_y * b_plus_y + x_minus_a * x_minus_a);
        sqrt_2nd = sqrt(a_plus_x * a_plus_x + b_plus_y * b_plus_y);

        return - sqrt_1st / (b_plus_y * x_minus_a) + sqrt_2nd / (a_plus_x * b_plus_y);
    }

    //--------------------------------------------------------------------------------------------------//

    // first order derivatives of I(x,y,z,xi,eta)
    double ip1(double& x, double& y, double& z, double& xi, double& eta) {
        double R;
        R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

        return log(R + y - eta);
    }

    double ip2(double& x, double& y, double& z, double& xi, double& eta) {
        double R;
        R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

        return log(R + x - xi);
    }

    double ip3(double& x, double& y, double& z, double& xi, double& eta) {
        double R;
        R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

        return -atan((x - xi) * (y - eta) / (z * R));
    }

    // second order derivatives of I(x,y,z,xi,eta)
    double ip11(double& x, double& y, double& z, double& xi, double& eta) {
    //    (x - \[Xi])/((y - \[Eta] + Sqrt(Power(z,2) + Power(y - \[Eta],2) + Power(x
    //    - \[Xi],2)))*
    //        Sqrt(Power(z,2) + Power(y - \[Eta],2) + Power(x - \[Xi],2)))
      double R;
      R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

      return (x - xi) / ((R + y - eta) * R);
    }

    double ip12(double& x, double& y, double& z, double& xi, double& eta) {
      // double R ;

      return 1. / sqrt(x * x + y * y + z * z - 2 * y * eta + eta * eta -
          2 * x * xi + xi * xi);
    }

    double ip13(double& x, double& y, double& z, double& xi, double& eta) {
      double R;
      R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

      return z / ((R + y - eta) * R);
    }

    double ip22(double& x, double& y, double& z, double& xi, double& eta) {
      //  (y - \[Eta])/(Sqrt[
      //  z^2 + (y - \[Eta])^2 + (x - \[Xi])^2] (x + Sqrt[
      //  z^2 + (y - \[Eta])^2 + (x - \[Xi])^2] - \[Xi]))
      double R;
      R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

      return (y - eta) / ((R + x - xi) * R);
    }

    double ip23(double& x, double& y, double& z, double& xi, double& eta) {
      //  z/(Sqrt[z^2 + (y - \[Eta])^2 + (x - \[Xi])^2] (x + Sqrt[
      //  z^2 + (y - \[Eta])^2 + (x - \[Xi])^2] - \[Xi]))
      double R;
      R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

      return z / ((R + x - xi) * R);
    }

    double ip33(double& x, double& y, double& z, double& xi, double& eta) {
        /*
         *
         *   The following way of writing leads to indeterminate results on the plane z = 0
         *   Reimplementing below - CP 2021
         *
              //  ((y - \[Eta]) (2 z^2 + (y - \[Eta])^2 + (x - \[Xi])^2) (x - \
                //\[Xi]))/((z^2 + (y - \[Eta])^2) (z^2 + (x - \[Xi])^2) Sqrt[
              //  z^2 + (y - \[Eta])^2 + (x - \[Xi])^2])
              double R;
              R = sqrt((x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z);

              return (x - xi) * (y - eta) *
                  (2 * z * z + (y - eta) * (y - eta) + (xi - x) * (xi - x)) /
                  (R * (z * z + (x - xi) * (x - xi)) * (z * z + (y - eta) * (y - eta)));

          *   The new way:
          */

            //        ((y - \[Eta]) (x - \[Xi]) (x^2 + y^2 + 2 z^2 -
            //        2 y \[Eta] + \[Eta]^2 - 2 x \[Xi] + \[Xi]^2))/((y^2 + z^2 -
            //        2 y \[Eta] + \[Eta]^2) (x^2 + z^2 - 2 x \[Xi] + \[Xi]^2) Sqrt[
            //        x^2 + y^2 + z^2 - 2 y \[Eta] + \[Eta]^2 - 2 x \[Xi] + \[Xi]^2])

            double xx = x * x, yy = y * y, zz = z * z, xixi = xi * xi, etaeta = eta * eta ;
            double mysqrt = sqrt( xx + yy + zz - 2 * y * eta + etaeta - 2 * x * xi + xixi);
            return ((y - eta) * (x - xi) * (xx + yy + 2 * zz - 2 * y * eta + etaeta - 2 * x *xi + xixi)) / (
                    (yy + zz - 2 * y * eta + etaeta) * (xx + zz - 2 * x * xi + xixi) * mysqrt);
    }

    //// third order derivatives of I(x,y,z,xi,eta)

    double ip111(double& x, double& y, double& z, double& xi, double& eta) {
      //  (R2 (Sqrt[R2] + y - \[Eta]) -
      //      Sqrt[R2] (x - \[Xi])^2 - (Sqrt[R2] + y - \[Eta]) (x - \[Xi])^2)/(R2^(
      //      3/2) (Sqrt[R2] + y - \[Eta])^2)

      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return (R * (R2 - 2. * pow(x - xi, 2)) + (y - eta) * (R2 - pow(x - xi, 2))) /
          (pow(R2, 3. / 2.) * pow(R + y - eta, 2.));
    }

    double ip112(double& x, double& y, double& z, double& xi, double& eta) {
      //  (-x + \[Xi])/(z^2 + (y - \[Eta])^2 + (x - \[Xi])^2)^(3/2)
      double R2;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;

      return (xi - x) / pow(R2, 3. / 2.);
    }

    double ip113(double& x, double& y, double& z, double& xi, double& eta) {
      //-((z (y - \[Eta] +
      //  2 Sqrt[z^2 + (y - \[Eta])^2 + (x - \[Xi])^2]) (x - \[Xi]))/((y - \
        //\[Eta] + Sqrt[
      //  z^2 + (y - \[Eta])^2 + (x - \[Xi])^2])^2 (z^2 + (y - \[Eta])^2 + \
        //(x - \[Xi])^2)^(3/2)))

      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return z * (xi - x) * (2. * R + y - eta) /
          (pow(R2, 3. / 2.) * pow(R + y - eta, 2));
    }

    double ip122(double& x, double& y, double& z, double& xi, double& eta) {
      //(-y + \[Eta])/(z^2 + (y - \[Eta])^2 + (x - \[Xi])^2)^(3/2)
      double R2;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;

      return (eta - y) / pow(R2, 3. / 2.);
    }

    double ip123(double& x, double& y, double& z, double& xi, double& eta) {
      double R2;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;

      return -z / pow(R2, 3. / 2.);
    }

    double ip133(double& x, double& y, double& z, double& xi, double& eta) {
      //  (R (R2 - 2 z^2) + (R2 - z^2) (y - \[Eta]))/(R2^(
      //      3/2) (R + y - \[Eta])^2)

      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return (R * (R2 - 2. * z * z) + (R2 - z * z) * (y - eta)) /
          (pow(R2, 3. / 2.) * pow(R + y - eta, 2.));
    }

    double ip222(double& x, double& y, double& z, double& xi, double& eta) {
      //  (R (R2 - 2 (y - \[Eta])^2) + (R2 - (y - \[Eta])^2) (x - \[Xi]))/(R2^(
      //      3/2) (R + x - \[Xi])^2)
      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return (R * (R2 - 2. * pow(y - eta, 2.)) +
          (x - xi) * (R2 - (y - eta) * (y - eta))) /
          (pow(R2, 3. / 2.) * pow(R + x - xi, 2.));
    }

    double ip223(double& x, double& y, double& z, double& xi, double& eta) {
      //  -((z (y - \[Eta]) (2 R + x - \[Xi]))/(R2^(3/2) (R + x - \[Xi])^2))

      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return z * (eta - y) * (2 * R + x - xi) /
          (pow(R2, 3. / 2.) * pow(R + x - xi, 2.));
    }

    double ip233(double& x, double& y, double& z, double& xi, double& eta) {
      //  (R (R2 - 2 z^2) + (R2 - z^2) (x - \[Xi]))/(R2^(3/2) (R + x - \[Xi])^2)
      double R2, R;
      R2 = (x - xi) * (x - xi) + (y - eta) * (y - eta) + z * z;
      R = sqrt(R2);

      return (R * (R2 - 2. * z * z) + (x - xi) * (R2 - z * z)) /
          (pow(R2, 3. / 2.) * pow(R + x - xi, 2.));
    }

    double ip333(double& x, double& y, double& z, double& xi, double& eta) {
    //    (-(2 R2^2 + R2 z^2 + 3 z^4) (z^2 + (y - \[Eta])^2) (y - \[Eta]) (z^2 + (x - \
    //    \[Xi])^2) (x - \[Xi]) + 2 (y - \[Eta])^3 (2 z^2 + (y - \[Eta])^2 + (x - \[Xi])^2)^2 (x - \
    //    \[Xi])^3)/(R2^(3/2) z (z^2 + (y - \[Eta])^2)^2 (z^2 + (x - \[Xi])^2)^2)

        double R2, z2, xmxi, ymeta;
        xmxi = (x - xi);
        ymeta = (y - eta);
        R2 = xmxi * xmxi + ymeta * ymeta + z * z;
        z2 = z * z;

        return (-(2 * R2 * R2 + R2 * z2 + 3 * z2 * z2) *
                 (z2 + pow(ymeta,2)) * ymeta *
                 (z2 + pow(xmxi,2)) * xmxi + 2 * pow(ymeta,3) *
                 pow(2 * z2 + pow(ymeta,2) + pow(xmxi,2),2) *
                 pow(xmxi,3)) / (pow(R2,1.5) * z * pow(z2 + pow(ymeta,2),2) *
                 pow(z2 + pow(xmxi,2),2));
    }

    // Chinnery Integration function - M. A. Chinnery, The deformations of the ground around surface faults, Bulletin of the Seismological Society of America (1961), 51 355-372

    double rectangular_integration(double& x, double& y, double& z, double& a,
                                   double& b, vFunctionCall Func) {
      double ma, mb;
      ma = -a;
      mb = -b;
      return (Func(x, y, z, a, b) - Func(x, y, z, a, mb) - Func(x, y, z, ma, b) +
          Func(x, y, z, ma, mb));
    }

    // Fundamental stress kernel
    bool is_stress_singular_at_given_location(double& x, double& y, double& z, double& a, double& b, bool verbose)
    {   double EPSILON;
        EPSILON = 100000 * std::numeric_limits<double>::epsilon();
        if (il::abs(z) <= EPSILON ){
            if (
                    ( il::abs(il::abs(x)/a - 1.) <= EPSILON && (il::abs(y) <= b) )  ||
                    ( il::abs(il::abs(y)/b - 1.) <= EPSILON && (il::abs(x) <= a ))
                    )
            {   if (verbose){
                    std::cout << "WARNING: \n " \
                          << " you are computing the stress along the edge of a 3DR0 element. \n" \
                          << " At that location some components of the stress tensor are theoretically infinite. \n" \
                          << " Suggestions: \n" \
                          << "    1) check that your collocation points do not lie on the edge of another element \n" \
                          << "    2) check that you are not computing the stress on the edge of an element \n";
                    }
                return true ;
            }
            else return false;
        }
        else return false; // meaning that z !=0
    }


    double get_Ip3 (double & x,double &y,double &z,double &a,double &b, double& Ip3_out_plane_z_EQ_0, bool verbose){
    /*
     * This function evaluate the position x,y,z with respect to a rectangle of coordinates
     * A=(-a,-b,0), B=(a,-b,0), C=(a,b,0) and D=(-a,b,0) and it returns the proper limit for the
     * kernel Ip3_lim_z_to_0
     *
     *              D________I____C
     *              |             |
     *              |             |
     *              |      +      |
     *              H        E    G      F
     *              |             |
     *              A--------J----B
     *
     *  Point   Value   Location
     *    A     -Pi/2   corner
     *    B     -Pi/2   corner
     *    C     -Pi/2   corner
     *    D     -Pi/2   corner
     *    E     -2*Pi   inside ABCDA
     *    F      0      outside ABCDA
     *    G     +Pi/2   on an edge (not corner)
     *    H     +Pi/2   on an edge (not corner)
     *    I     +Pi/2   on an edge (not corner)
     *    J     +Pi/2   on an edge (not corner)
     *
     */
        double EPSILON;
        EPSILON = 100000 * std::numeric_limits<double>::epsilon();
        if (il::abs(z) > EPSILON) { // NOT on the plane z==0
             return Ip3_out_plane_z_EQ_0;
        }
        else
        {   if (il::abs(il::abs(x) / a - 1.) <= EPSILON && (il::abs(y) / b - 1 < (-EPSILON )))   // internal vertical edges
                return -il::pi;
            else if (il::abs(il::abs(y) / b - 1.) <= EPSILON && (il::abs(x) / a - 1 < (- EPSILON) ))   // internal horizontal edges
                return -il::pi;
            else if ((il::abs(il::abs(x) / a - 1.) <= EPSILON) && (il::abs(il::abs(y) / b - 1.) <= EPSILON)) // corners
                {   if (verbose){
                        std::cout << "WARNING: \n " \
                          << " you are computing the displacement at the corner of a 3DR0 element. \n" \
                          << " At that location some components of the displacement are theoretically infinite (because Ip1 and Ip2 go to -inf). \n" \
                          << " Suggestions: \n" \
                          << "    - check that your collocation points do not lie on the corner of another element \n" ;
                    }
                    return -il::pi/2.;}
            else if  (((il::abs(x) / a - 1.) <= (-EPSILON)) && ((il::abs(y) / b - 1.) <= (-EPSILON)))// inside the rectangular element
                return -2.*il::pi;

            else // on the plane and outside the rectangle
                //  also on the prolongation of vertical edges
                //  also on the prolongation of horizontal edges
                return 0.;
        }
    }


    il::Array2D<double> transpose(il::Array2D<double> M){
        // this function returns the transposed matrix
        il::Array2D<double> MT{3,3,0.};
        for (il::int_t i = 0; i < 3; i++) {
            for (il::int_t j = 0; j < 3; j++) {
                MT(j,i) = M(i,j);
            }
        }
        return MT;
    }

    il::Array2D<double> change_ref_system (const il::Array2D<double>& linearApplication,il::int_t change_domain, il::int_t change_codomain, const il::Array2D<double>& RglobalTOlocal_domain, const il::Array2D<double>& RglobalTOlocal_codomain){
        // Description:
        // A linear application takes values from a domain and outputs values in a codomain.
        // This function changes the base in the domain, in the codomain or in bonth.
        //
        // Input:
        // linearApplication is a matrix that thought to be in a local reference system both in the domain and in the codomain
        // change_domain can be 0 (false) or 1 (true)
        //      - if false the domain will be expressed with respect to the local reference system to the source element
        //      - if true the domain will be expressed with respect to the global reference system
        // change_codomain can be 0 (false) or 1 (true)
        //      - if false the domain will be expressed with respect to the local reference system to the receiver element
        //      - if true the codomain will be expressed with respect to the global reference system
        //
        // RglobalTOlocal_domain it is a matrix that rotates a vector from the global reference system (r.s.) to the local r.s. of the source element
        // RglobalTOlocal_codomain it is a matrix that rotates a vector from the global reference system (r.s.) to the local r.s. of the receiver element
        //
        // Output:
        // A rotated matrix i.e. linear application
        il::Array2D<double> rotatedLinearApplication{3,3,0.};

        if (change_domain == 0 && change_codomain == 0)
        {   // False - False  <=> local source and local receiver
            // Explanation with regard to the problem at hand:
            //    R(global to local codomain)*R(from local to global domain)*(M_localDD_localTraction_source) = (M_globalDD_localTraction_receiver)
            //    R(global to local codomain)*R(from local to global domain)*(M_localDD_localTraction_source) = (M_globalDD_localDisplacements_receiver)
            il::Array2D<double> RT = transpose(RglobalTOlocal_domain);
            rotatedLinearApplication = il::dot(RT,linearApplication);
            rotatedLinearApplication = il::dot(RglobalTOlocal_codomain,rotatedLinearApplication);
        }
        else if (change_domain == 0 && change_codomain == 1)
        {   // False - True  <=> local source and global receiver
            // Explanation with regard to the problem at hand:
            //    R(from local domain to global)*(M_localDD_localTraction) = (M_globalDD_localTraction)
            //    R(from local domain to global)*(M_localDD_localDisplacement) = (M_globalDD_localDisplacement)
            il::Array2D<double> RT = transpose(RglobalTOlocal_domain);
            rotatedLinearApplication = il::dot(RT,linearApplication);

        }
        else if (change_domain == 1 && change_codomain == 1)
        {   // True - True  <=> global source and global receiver
            // Explanation with regard to the problem at hand:
            //    R(from local domain to global)*(M_localDD_localTraction)*R(from global to local domain) = (M_globalDD_globalTraction)
            //    R(from local domain to global)*(M_localDD_localDisplacement)*R(from global to local domain) = (M_globalDD_globalDisplacement)
            rotatedLinearApplication = il::dot(linearApplication,RglobalTOlocal_domain);
            il::Array2D<double> RT = transpose(RglobalTOlocal_domain);
            rotatedLinearApplication = il::dot(RT,rotatedLinearApplication);
        }
        else if (change_domain == 1 && change_codomain == 0)
        {   // True - False  <=> global source and local receiver
            // Explanation with regard to the problem at hand:
            //    R(global to local codomain)*R(from local domain to global)*(M_localDD_localTraction)*R(from global to local domain) = (M_globalDD_localTraction)
            //    R(global to local codomain)*R(from local domain to global)*(M_localDD_localDisplacement)*R(from global to local domain) = (M_globalDD_localDisplacement)

            rotatedLinearApplication = il::dot(linearApplication,RglobalTOlocal_domain);
            il::Array2D<double> RT = transpose(RglobalTOlocal_domain);
            rotatedLinearApplication = il::dot(RT,rotatedLinearApplication);
            rotatedLinearApplication = il::dot(RglobalTOlocal_codomain,rotatedLinearApplication);
        }
        else { std::cout << "ERROR: bad options given for switch in routine: change_ref_system = " << change_domain << "\n";}

        return rotatedLinearApplication;
    }

    il::StaticArray<double,2> get_a_and_b(il::Array2D <double>xv, double NoV)
    {
        /*
         * This function returns two values:
         *
         * a := half length of the 1st edge of an element
         * b := half length of the last edge of an element
         */
        il::StaticArray<double,2> a_and_b;

        // vec01 goes from vertex 0 to vertex 1
        // vec02 goes from vertex 0 to vertex NoV_ (vertex 2 in case of triangular element)
        il::StaticArray<double, 3> vec01, vec02;
        vec01[0] = xv(1, 0) - xv(0, 0);
        vec01[1] = xv(1, 1) - xv(0, 1);
        vec01[2] = xv(1, 2) - xv(0, 2);
        vec02[0] = xv(NoV - 1, 0) - xv(0, 0);
        vec02[1] = xv(NoV - 1, 1) - xv(0, 1);
        vec02[2] = xv(NoV - 1, 2) - xv(0, 2);

        double vec01norm=sqrt(il::dot(vec01, vec01)), vec02norm=sqrt(il::dot(vec02, vec02));

        a_and_b[0] =vec01norm/2.; // a
        a_and_b[1] =vec02norm/2.; // b

        return a_and_b ;
    }

}

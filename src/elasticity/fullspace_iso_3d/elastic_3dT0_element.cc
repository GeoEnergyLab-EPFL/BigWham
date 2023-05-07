//
// This file part of BigWham
//
// Created by Brice Lecampion on 23.05.20.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2020.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <iostream>

#include <il/linearAlgebra.h>
#include <il/linearAlgebra/dense/blas/dot.h>
#include <il/linearAlgebra/dense/norm.h>

#include "elastic_3dT0_element.h"

namespace bie {

//    il::Array2D<double> traction_influence_3DT0(
//            bie::FaceData &elem_data_s, // source element
//            bie::FaceData &elem_data_r, // receiver element
//            bie::ElasticProperties const &elas_, // elastic properties
//            il::int_t local_global = 0){ // 0 if local-local, 1 if
//            global-global
//
//        // get constitutive parameters
//        double G = elas_.shear_modulus();
//        double nu = elas_.poisson_ratio();
//
//        // get coordinates receiver cp
//        il::Array2D<double> el_cp_r;
//        el_cp_r = elem_data_r.collocation_points();
//
//        // get coordinates vertices of triangular source element
//        il::Array2D<double> el_vertices_s;
//        el_vertices_s = elem_data_s.getVertices();
//
//        // get stress influence coefficients - in the local coordinate system
//        of the source element il::StaticArray2D<double, 3, 6> Stress; Stress =
//        StressesKernelT0(el_cp_r,el_vertices_s,G,nu);
//
//        // index        ->    0    1    2    3    4    5
//        // DD1 (shear)  -> | s11, s22, s33, s12, s13, s23  |
//        // DD2 (shear)  -> | s11, s22, s33, s12, s13, s23  |
//        // DD3 (normal) -> | s11, s22, s33, s12, s13, s23  |
//
//        // normal vector at the receiver location in the reference system of
//        the source element il::Array2D<double> R_source =
//        elem_data_s.rotationMatrix(true);
////        il::Array2D<double> RT = transpose(R);
//        il::Array<double> nr = elem_data_r.getNormal();
//        nr = il::dot(R_source,nr);
//
//        // compute traction vectors at receiver element cp due to
//        (DD1,DD2,DD3) source element
//        // in the reference system of the source element
//        il::Array2D<double> DDs_to_traction_local{3,3,0.0}; // traction
//        vectors il::Array<double> traction_temp; // temporary traction vector
//        il::Array2D<double> sigma_temp{3,3,0.0}; // temporary stress tensor
//
//        for (int i = 0; i < 3; ++i)
//        { //loop over the rows of Stress, i.e., over each DD component effect
//            // definition of temporary stress tensor
//            sigma_temp(0,0) = Stress(i,0); // S11
//            sigma_temp(0,1) = Stress(i,3); // S12
//            sigma_temp(0,2) = Stress(i,4); // S13
//            sigma_temp(1,0) = Stress(i,3); // S21
//            sigma_temp(1,1) = Stress(i,1); // S22
//            sigma_temp(1,2) = Stress(i,5); // S23
//            sigma_temp(2,0) = Stress(i,4); // S31
//            sigma_temp(2,1) = Stress(i,5); // S32
//            sigma_temp(2,2) = Stress(i,2); // S33
//
//            // compute temporary traction vector
//            traction_temp = il::dot(sigma_temp, nr);
//            for (int j = 0; j < 3; ++j) {
//                // fill the traction vectors at receiver element cp due to
//                (DD1,DD2,DD3) source element DDs_to_traction_local(j,i) =
//                traction_temp[j];
//                // | t1/D1   t1/D2  t1/D3 |
//                // | t2/D1   t2/D2  t2/D3 |
//                // | t3/D1   t3/D2  t3/D3 |
//                // local DD & local traction
//                // both in the reference system of the source element
//            }
//        }
//
//        return
//        change_local_global(DDs_to_traction_local,local_global,R_source,elem_data_r.rotationMatrix(true));
//
//        // | t1 due to D1   t1 due to D2  t1 due to D3 |
//        // | t2 due to D1   t2 due to D2  t2 due to D3 |
//        // | t3 due to D1   t3 due to D2  t3 due to D3 |
//    }
//
//    il::Array2D<double> displacement_influence_3DT0(
//            FaceData &elem_data_s, // source element
//            FaceData &elem_data_r, // receiver element
//            ElasticProperties const &elas_, // elastic properties
//            il::int_t local_global = 0) // 0 if local-local, 1 if
//            global-global
//    {
//
//        // get constitutive parameters - only nu is needed
//        double nu = elas_.poisson_ratio();
//
//        // get coordinates receiver cp
//        il::Array2D<double> el_cp_r;
//        el_cp_r = elem_data_r.collocation_points();
//
//        // get coordinates vertices of triangular source element
//        il::Array2D<double> el_vertices_s;
//        el_vertices_s = elem_data_s.getVertices();
//
//        // compute displacement components at receiver element cp due to
//        (DD1,DD2,DD3) source element
//        // in the reference system of the source element
//
//        il::StaticArray2D<double, 3, 3> DDs_to_Displacement_local;
//        DDs_to_Displacement_local =
//        DisplacementKernelT0(el_cp_r,el_vertices_s,nu);
//        // index        ->    DD1 (shear)    DD2 (shear)     DD3 (normal)
//        //   0      -> |       U1,            U1,             U1            |
//        //   1      -> |       U2,            U2,             U2            |
//        //   2      -> |       U3,            U3,             U3            |
//
//        // convert from static 2D array to dynamic, might be inefficient
//        il::Array2D<double> X{3,3};
//        for (int i = 0; i < 3; i++) {
//            X(0,i) = DDs_to_Displacement_local(0,i);
//            X(1,i) = DDs_to_Displacement_local(1,i);
//            X(2,i) = DDs_to_Displacement_local(2,i);
//        }
//
//        return
//        change_local_global(X,local_global,elem_data_s.rotationMatrix(true),
//        elem_data_r.rotationMatrix(true));
//        // | U1/D1   U1/D2  U1/D3 |
//        // | U2/D1   U2/D2  U2/D3 |
//        // | U3/D1   U3/D2  U3/D3 |
//    }

// Fundamental stress kernel = stress influence coefficients
il::StaticArray2D<double, 3, 6>
StressesKernelT0(il::StaticArray<double, 3> &x,
                 il::StaticArray2D<double, 3, 3> &xv, double &G, double &nu) {
  // this routine is based on the works of Nintcheu Fata (2009,2011)

  // inputs
  //   -x: source/receiver point's coordinates in the global reference system in
  //   the form (x,y,z) -xv: vertices' coordinates in the global reference
  //   system, in the following 2D array form:
  //     x0 y0 z0
  //     x1 y1 z1
  //     x2 y2 z2
  //   -G: shear modulus
  //   -nu: Poisson's ratio
  // output
  //   Stress = 3 x 6 matrix with the stress influence coefficients arranged as:
  //   DD1 (shear), DD2 (shear), DD3 (normal) -> for rows
  //   S11, S22, S33, S12, S13, S23 -> for columns

  //        double eps_tol = 2.22045e-16; // parameter used for "if conditions"
  //        involving inequalities due to numerical precision

  // get triangle vertices coordinates separated
  il::StaticArray<double, 3> y1, y2, y3;
  for (il::int_t i = 0; i < 3; i++) {
    y1[i] = xv(0, i);
    y2[i] = xv(1, i);
    y3[i] = xv(2, i);
  }
  // subtractions of coordinates vectors
  il::StaticArray<double, 3> y31, y21, y1x;
  for (il::int_t i = 0; i < 3; i++) {
    y31[i] = y3[i] - y1[i];
    y21[i] = y2[i] - y1[i];
    y1x[i] = y1[i] - x[i];
  }
  // local reference system (e1,e2,e3)
  il::StaticArray<double, 3> e1, e2, e3;

  // e1
  e1[0] = y21[0] / il::norm(y21, il::Norm::L2);
  e1[1] = y21[1] / il::norm(y21, il::Norm::L2);
  e1[2] = y21[2] / il::norm(y21, il::Norm::L2);

  // e2
  double sigma = il::dot(y31, y21) / pow(il::norm(y21, il::Norm::L2), 2.0);
  il::StaticArray<double, 3> e2vector; // not normalized
  e2vector[0] = y31[0] - sigma * y21[0];
  e2vector[1] = y31[1] - sigma * y21[1];
  e2vector[2] = y31[2] - sigma * y21[2];
  e2[0] = e2vector[0] / il::norm(e2vector, il::Norm::L2);
  e2[1] = e2vector[1] / il::norm(e2vector, il::Norm::L2);
  e2[2] = e2vector[2] / il::norm(e2vector, il::Norm::L2);

  // e3
  e3 = il::cross(e1, e2);

  // geometrical parameters
  double a, b, c;
  a = il::dot(y31, e2);
  b = il::dot(y21, e1);
  c = il::dot(y31, e1);
  double theta1, theta2, theta3;
  theta1 = acos(c / sqrt(pow(c, 2.0) + pow(a, 2.0)));
  theta2 = acos((b - c) / sqrt(pow(b - c, 2.0) + pow(a, 2.0)));
  theta3 = il::pi - (theta1 + theta2);

  // endpoints' coordinates of each triangle edge for the local coordinate
  // systems of each edge

  // vertex 1 coordinates
  double xi1, zeta1, eta;
  xi1 = il::dot(y1x, e1);
  zeta1 = il::dot(y1x, e2);
  eta = il::dot(y1x, e3);

  // auxiliary angles
  double alpha1, alpha2, alpha3;
  alpha1 = 0.;
  alpha2 = il::pi - theta2;
  alpha3 = il::pi + theta1;

  // endpoints' coordinates edge L1
  double p11, p12, q1;
  p11 = xi1;
  p12 = b + xi1;
  q1 = zeta1;

  // endpoints' coordinates edge L2
  double p22, p23, q2;
  p22 = (b + xi1) * cos(alpha2) + zeta1 * sin(alpha2);
  p23 = (c + xi1) * cos(alpha2) + (a + zeta1) * sin(alpha2);
  q2 = -(b + xi1) * sin(alpha2) + zeta1 * cos(alpha2);

  // endpoints' coordinates edge L3
  double p33, p31, q3;
  p33 = (c + xi1) * cos(alpha3) + (a + zeta1) * sin(alpha3);
  p31 = xi1 * cos(alpha3) + zeta1 * sin(alpha3);
  q3 = -xi1 * sin(alpha3) + zeta1 * cos(alpha3);

  // generic recursive functions

  // previous definitions

  // p
  il::StaticArray2D<double, 3, 3> p;
  p(0, 0) = p11;
  p(1, 1) = p22;
  p(2, 2) = p33;
  p(0, 1) = p12;
  p(1, 2) = p23;
  p(2, 0) = p31;

  // q
  il::StaticArray<double, 3> q;
  q[0] = q1;
  q[1] = q2;
  q[2] = q3;

  // rho - distance between source/receiver point 'x' and i-th vertex of
  // triangle
  il::StaticArray<double, 3> rho;
  for (int i = 0; i < 3; i++) {
    rho[i] = sqrt(pow(p(i, i), 2.0) + pow(q[i], 2.0) + pow(eta, 2.0));
  }

  // now the generic recursive functions

  int i1; // i+1, will be used in all recursive generic functions

  // d
  il::StaticArray<double, 3> d;
  for (int i = 0; i < 3; i++) {
    d[i] = pow(q[i], 2.0) + pow(eta, 2.0);
  }

  //        // rho tilde
  //        il::StaticArray<double, 3> rho_t;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            rho_t[i] = rho[i] - rho[i1];
  //        }

  //        // phi
  //        il::StaticArray<double, 3> phi;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            phi[i] = p(i, i) * rho[i] - p(i, i1) * rho[i1];
  //        }

  //        // chi
  //        // Note: in Nintcheu Fata (2009, 2011) there are no comments for the
  //        case
  //        // in which the log arguments are negative or zero and this can
  //        happen! Brice
  //        // regularized this in his mathematica notebook but I don't know yet
  //        how (Alexis).
  //        // I use here Brice's solution
  //        il::StaticArray<double, 3> chi;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            if ((p(i, i) + rho[i]) > eps_tol && (p(i, i1) + rho[i1]) >
  //            eps_tol) {
  //                // if the log arguments are strictly positive
  //                chi[i] = log(p(i, i) + rho[i]) - log(p(i, i1) + rho[i1]);
  //            } else if ((p(i, i) + rho[i]) < -eps_tol && (p(i, i1) + rho[i1])
  //            < -eps_tol) {
  //                // if the log arguments are strictly negative
  //                double d = pow(q[i], 2.0) + pow(eta, 2.0);
  //                double p1 = p(i, i) + rho[i];
  //                double p2 = p(i, i1) + rho[i1];
  //                double z1 = (d / p1) / p1;
  //                double z2 = (d / p2) / p2;
  //                chi[i] = log(
  //                        abs(p2 / p1) * ((1.0 - 0.25 * z1 * (1.0 - 0.5 * z1))
  //                        / (1.0 - 0.25 * z2 * (1.0 - 0.5 * z2))));
  //            } else {
  //                // if any of the log arguments is zero
  //                chi[i] = 0.0;
  //            }
  //        }

  // delta
  il::StaticArray<double, 3> delta;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    delta[i] = p(i, i) / rho[i] - p(i, i1) / rho[i1];
  }

  // L
  il::StaticArray<double, 3> L;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    L[i] = 1.0 / rho[i] - 1.0 / rho[i1];
  }

  // D
  il::StaticArray<double, 3> D;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    D[i] = p(i, i) / pow(rho[i], 3.0) - p(i, i1) / pow(rho[i1], 3.0);
  }

  // Lambda
  il::StaticArray<double, 3> Lambda;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    Lambda[i] = 1.0 / pow(rho[i], 3.0) - 1.0 / pow(rho[i1], 3.0);
  }

  //        // definition of cases - where the projected source/receiver point
  //        lies
  //        // also, computation of theta_0
  //
  //        // rho_plane - distance between projected source/receiver point 'x'
  //        and i-th vertex of triangle il::StaticArray<double, 3> rho_plane;
  //        for (int i = 0; i < 3; i++) {
  //            rho_plane[i] = sqrt(pow(p(i, i), 2.0) + pow(q[i], 2.0));
  //        }
  //
  //        int id;
  //        double theta_0;
  //        if (q[0] > eps_tol || q[1] > eps_tol || q[2] > eps_tol) {
  //            // if projected point lies strictly outside of the triangle
  //            id = -1;
  //            theta_0 = 0.0;
  //        } else if (q[0] < -eps_tol && q[1] < -eps_tol && q[2] < -eps_tol) {
  //            // if projected point lies strictly inside of the triangle
  //            id = 0;
  //            theta_0 = 2 * il::pi;
  //        } else if (rho_plane[0] > eps_tol && rho_plane[1] > eps_tol &&
  //        rho_plane[2] > eps_tol) {
  //            // if projected point lies on an edge of the triangle but not on
  //            a vertex id = 4; theta_0 = il::pi;
  //        } else if (rho_plane[0] < eps_tol) {
  //            // if projected point lies exactly on vertex 1
  //            id = 1;
  //            theta_0 = theta1;
  //        } else if (rho_plane[1] < eps_tol) {
  //            // if projected point lies exactly on vertex 2
  //            id = 2;
  //            theta_0 = theta2;
  //        } else if (rho_plane[2] < eps_tol) {
  //            // if projected point lies exactly on vertex 3
  //            id = 3;
  //            theta_0 = theta3;
  //        }

  //        // gamma
  //
  //        // Note: in Nintcheu Fata (2009, 2011) there are no comments for the
  //        case
  //        // in which the arctan arguments are not defined (in this case, 0/0)
  //        and this can happen! Brice
  //        // regularized this in his mathematica notebook by taking the
  //        limits, I understand the case of
  //        // the edge, but not the cases for the vertices. I just coded up
  //        Brice's solution for now, but
  //        // I didn't include the case inside the element and the case right
  //        at the vertices il::StaticArray<double, 3> gamma;
  //
  //        if (id == 1) {
  //            // if the projected source/receiver point lies on vertex 1
  //            gamma[0] = 0.0;
  //            gamma[2] = 0.0;
  //            int i = 1;
  //            i1 = 2;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else if (id == 2) {
  //            // if the projected source/receiver point lies on vertex 2
  //            gamma[0] = 0.0;
  //            gamma[1] = 0.0;
  //            int i = 2;
  //            i1 = 0;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else if (id == 3) {
  //            // if the projected source/receiver point lies on vertex 3
  //            gamma[1] = 0.0;
  //            gamma[2] = 0.0;
  //            int i = 0;
  //            i1 = 1;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else if (abs(q[0]) < eps_tol) {
  //            // if the projected source/receiver point lies on the edge 1 or
  //            its prolongation (in both directions)
  //            // the limit is needed only for the case eta = 0. However, for
  //            eta =! 0, the value of gamma1 is the same
  //            // as the limit (=zero), so that's why I don't make the
  //            difference gamma[0] = 0.0; int i = 1; i1 = 2; gamma[i] =
  //            atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //            i = 2;
  //            i1 = 0;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else if (abs(q[1]) < eps_tol) {
  //            // if the projected source/receiver point lies on the edge 2 or
  //            its prolongation (in both directions)
  //            // the limit is needed only for the case eta = 0. However, for
  //            eta =! 0, the value of gamma2 is the same
  //            // as the limit (=zero), so that's why I don't make the
  //            difference gamma[1] = 0.0; int i = 0; i1 = 1; gamma[i] =
  //            atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //            i = 2;
  //            i1 = 0;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else if (abs(q[2]) < eps_tol) {
  //            // if the projected source/receiver point lies on the edge 3 or
  //            its prolongation (in both directions)
  //            // the limit is needed only for the case eta = 0. However, for
  //            eta =! 0, the value of gamma3 is the same
  //            // as the limit (=zero), so that's why I don't make the
  //            difference gamma[2] = 0.0; int i = 0; i1 = 1; gamma[i] =
  //            atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //            i = 1;
  //            i1 = 2;
  //            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
  //                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i,
  //                            i), 2.0) * pow(eta, 2.0)))
  //                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i,
  //                              i1), 2.0) * pow(eta, 2.0)));
  //        } else {
  //            // any other case
  //            for (int i = 0; i < 3; i++) {
  //                i1 = i + 1;
  //                if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace
  //                by 0 gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i])
  //                /
  //                                (pow(q[i], 2.0) * pow(rho[i], 2.0) -
  //                                pow(p(i, i), 2.0) * pow(eta, 2.0)))
  //                           - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
  //                                  (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
  //                                  pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  //            }
  //        }

  //        // theta for displacements and tractions at the boundary of the
  //        domain double theta; double thetaSum = 0.0; for (int i = 0; i < 3;
  //        i++) {
  //            thetaSum += gamma[i];
  //        }
  //        if (eta < eps_tol) {
  //            // if eta is negative or zero
  //            theta = 0.5 * thetaSum - theta_0;
  //        } else {
  //            // if eta is strictly positive
  //            theta = 0.5 * thetaSum + theta_0;
  //        }

  // sin(alpha[i]) and cos(alpha[i])

  // alpha
  il::StaticArray<double, 3> alpha;
  alpha[0] = alpha1;
  alpha[1] = alpha2;
  alpha[2] = alpha3;

  il::StaticArray<double, 3> sinAlpha;
  il::StaticArray<double, 3> cosAlpha;
  for (int i = 0; i < 3; i++) {
    sinAlpha[i] = sin(alpha[i]);
    cosAlpha[i] = cos(alpha[i]);
  }

  // compute necessary generic (+ auxiliary) integrals
  // in the order they appear in the stress influence coefficients

  // first, for stress influence coefficients due to DD1

  double I5_Xi = i5_Xi(delta, d, sinAlpha);
  double I7_Xi_Xi_Xi = i7_Xi_Xi_Xi(q, d, sinAlpha, cosAlpha, D, Lambda, delta);
  double I7_Xi_Zeta_Zeta =
      i7_Xi_Zeta_Zeta(q, d, sinAlpha, cosAlpha, D, Lambda, delta);
  double I7_Xi = i7_Xi(d, D, delta, sinAlpha);
  double I7_Xi_Xi_Zeta =
      i7_Xi_Xi_Zeta(q, d, sinAlpha, cosAlpha, D, Lambda, delta);
  double I5_Zeta = i5_Zeta(delta, d, cosAlpha);
  double I7_Xi_Xi_Aux =
      i7_Xi_Xi_Aux(eta, cosAlpha, Lambda, sinAlpha, q, d, D, delta);
  double I5_Zeta_Zeta_Aux =
      i5_Zeta_Zeta_Aux(L, sinAlpha, q, d, delta, cosAlpha);
  double I7_Xi_Zeta = i7_Xi_Zeta(sinAlpha, Lambda, cosAlpha, q, d, D, delta);
  double I5_Xi_Zeta = i5_Xi_Zeta(L, sinAlpha, q, d, delta, cosAlpha);

  // second, for stress influence coefficients due to DD2

  double I7_Zeta_Zeta_Zeta =
      i7_Zeta_Zeta_Zeta(q, d, sinAlpha, cosAlpha, D, Lambda, delta);
  double I7_Zeta = i7_Zeta(d, D, delta, cosAlpha);
  double I7_Zeta_Zeta_Aux =
      i7_Zeta_Zeta_Aux(eta, cosAlpha, Lambda, sinAlpha, q, d, D, delta);
  double I5_Xi_Xi_Aux = i5_Xi_Xi_Aux(L, sinAlpha, q, d, delta, cosAlpha);

  // third, for stress influence coefficients due to DD3

  double I5_Aux = i5_Aux(q, d, delta);
  double I7_Aux = i7_Aux(eta, q, d, D, delta);

  // And finally, the stress influence coefficients

  double prefactor =
      (G / (4.0 * il::pi * (1.0 - nu))); // common prefactor of all coefficients

  il::StaticArray2D<double, 3, 6> Stress; // output

  // recall:
  // DD1 (shear), DD2 (shear), DD3 (normal) -> for rows
  // Stress components: S11, S22, S33, S12, S13, S23 -> for columns

  // stress components due to the unit displacement discontinuity component DD1
  // (shear)

  Stress(0, 0) = prefactor * (-3.0 * eta * (I5_Xi - 5.0 * I7_Xi_Xi_Xi));
  // s11 = b111
  Stress(0, 1) =
      prefactor *
      (3.0 * eta * (I5_Xi * (-1.0 + 2.0 * nu) + 5.0 * I7_Xi_Zeta_Zeta));
  // s22 = b221
  Stress(0, 2) =
      prefactor * (-3.0 * eta * (I5_Xi - 5.0 * I7_Xi * pow(eta, 2.0)));
  // s33 = b331
  Stress(0, 3) = prefactor * (3.0 * eta * (5.0 * I7_Xi_Xi_Zeta - I5_Zeta * nu));
  // s12 = b121
  Stress(0, 4) =
      prefactor * (3.0 * (5.0 * I7_Xi_Xi_Aux + I5_Zeta_Zeta_Aux * nu));
  // s13 = b131
  Stress(0, 5) =
      prefactor * (15.0 * I7_Xi_Zeta * pow(eta, 2.0) - 3.0 * I5_Xi_Zeta * nu);
  // s23 = b231

  // stress components due to the unit displacement discontinuity component DD2
  // (shear)

  Stress(1, 0) =
      prefactor *
      (3.0 * eta *
       (5.0 * I7_Xi_Xi_Zeta + I5_Zeta * (-1.0 + 2.0 * nu))); // s11 = b112
  Stress(1, 1) = prefactor * (-3.0 * eta * (I5_Zeta - 5.0 * I7_Zeta_Zeta_Zeta));
  // s22 = b222
  Stress(1, 2) =
      prefactor * (-3.0 * eta * (I5_Zeta - 5.0 * I7_Zeta * pow(eta, 2.0)));
  // s33 = b332
  Stress(1, 3) = prefactor * (3.0 * eta * (5.0 * I7_Xi_Zeta_Zeta - I5_Xi * nu));
  // s12 = b122
  Stress(1, 4) =
      prefactor * (15.0 * I7_Xi_Zeta * pow(eta, 2.0) - 3.0 * I5_Xi_Zeta * nu);
  // s13 = b132
  Stress(1, 5) =
      prefactor * (3.0 * (5.0 * I7_Zeta_Zeta_Aux + I5_Xi_Xi_Aux * nu));
  // s23 = b232

  // stress components due to the unit displacement discontinuity component DD3
  // (normal)

  //        Stress(2, 0) = prefactor * ( 3.0 * ( I5_Zeta_Zeta_Aux + 5.0 *
  //        I7_Xi_Xi_Aux
  //                                             - 2.0 * I5_Zeta_Zeta_Aux * nu )
  //                                             ); // s11 = b113 without
  //                                             factorization
  Stress(2, 0) = prefactor * (3.0 * ((1.0 - 2.0 * nu) * I5_Zeta_Zeta_Aux +
                                     5.0 * I7_Xi_Xi_Aux)); // s11 = b113
  //        Stress(2, 1) = prefactor * ( 3.0 * ( I5_Xi_Xi_Aux + 5.0 *
  //        I7_Zeta_Zeta_Aux
  //                                             - 2.0 * I5_Xi_Xi_Aux * nu ) );
  //                                             // s22 = b223 without
  //                                             factorization
  Stress(2, 1) = prefactor * (3.0 * ((1.0 - 2.0 * nu) * I5_Xi_Xi_Aux +
                                     5.0 * I7_Zeta_Zeta_Aux)); // s22 = b223
  Stress(2, 2) = prefactor * (-6.0 * I5_Aux + 15.0 * I7_Aux);
  // s33 = b333
  Stress(2, 3) = prefactor * (15.0 * I7_Xi_Zeta * pow(eta, 2.0) +
                              I5_Xi_Zeta * (-3.0 + 6.0 * nu)); // s12 = b123
  Stress(2, 4) =
      prefactor * (-3.0 * eta * (I5_Xi - 5.0 * I7_Xi * pow(eta, 2.0)));
  // s13 = b133
  Stress(2, 5) =
      prefactor * (-3.0 * eta * (I5_Zeta - 5.0 * I7_Zeta * pow(eta, 2.0)));
  // s23 = b233

  return Stress;
}

// generic (and auxiliary) integrals first appearing for stress influence
// coefficients

// By order of appearance in stress influence coefficients due to DD1

double i5_Xi(il::StaticArray<double, 3> &delta, il::StaticArray<double, 3> &d,
             il::StaticArray<double, 3> &sinAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (delta[i] / d[i]) * sinAlpha[i];
  }
  return (1.0 / 3.0) * sum;
}

double i7_Xi_Xi_Xi(il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                   il::StaticArray<double, 3> &sinAlpha,
                   il::StaticArray<double, 3> &cosAlpha,
                   il::StaticArray<double, 3> &D,
                   il::StaticArray<double, 3> &Lambda,
                   il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (((pow(q[i], 2.0) / d[i]) * pow(sinAlpha[i], 2.0) -
             pow(cosAlpha[i], 2.0)) *
                D[i] +
            2.0 * sinAlpha[i] * cosAlpha[i] * q[i] * Lambda[i] +
            (1.0 / d[i]) *
                (3.0 - pow(sinAlpha[i], 2.0) +
                 2.0 * (pow(q[i], 2.0) / d[i]) * pow(sinAlpha[i], 2.0)) *
                delta[i]) *
           sinAlpha[i];
  }
  return (1.0 / 15.0) * sum;
}

double i7_Xi_Zeta_Zeta(il::StaticArray<double, 3> &q,
                       il::StaticArray<double, 3> &d,
                       il::StaticArray<double, 3> &sinAlpha,
                       il::StaticArray<double, 3> &cosAlpha,
                       il::StaticArray<double, 3> &D,
                       il::StaticArray<double, 3> &Lambda,
                       il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (((pow(q[i], 2.0) / d[i]) * pow(cosAlpha[i], 2.0) -
             pow(sinAlpha[i], 2.0)) *
                D[i] -
            2.0 * sinAlpha[i] * cosAlpha[i] * q[i] * Lambda[i] +
            (1.0 / d[i]) *
                (pow(sinAlpha[i], 2.0) +
                 2.0 * (pow(q[i], 2.0) / d[i]) * pow(cosAlpha[i], 2.0)) *
                delta[i]) *
           sinAlpha[i];
  }
  return (1.0 / 15.0) * sum;
}

double i7_Xi(il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
             il::StaticArray<double, 3> &delta,
             il::StaticArray<double, 3> &sinAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (1.0 / d[i]) * (D[i] + (2.0 / d[i]) * delta[i]) * sinAlpha[i];
  }
  return (1.0 / 15.0) * sum;
}

double i7_Xi_Xi_Zeta(il::StaticArray<double, 3> &q,
                     il::StaticArray<double, 3> &d,
                     il::StaticArray<double, 3> &sinAlpha,
                     il::StaticArray<double, 3> &cosAlpha,
                     il::StaticArray<double, 3> &D,
                     il::StaticArray<double, 3> &Lambda,
                     il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (((pow(q[i], 2.0) / d[i]) * pow(sinAlpha[i], 2.0) -
             pow(cosAlpha[i], 2.0)) *
                D[i] +
            2.0 * sinAlpha[i] * cosAlpha[i] * q[i] * Lambda[i] +
            (1.0 / d[i]) *
                (pow(cosAlpha[i], 2.0) +
                 2.0 * (pow(q[i], 2.0) / d[i]) * pow(sinAlpha[i], 2.0)) *
                delta[i]) *
           cosAlpha[i];
  }
  return -(1.0 / 15.0) * sum;
}

double i5_Zeta(il::StaticArray<double, 3> &delta, il::StaticArray<double, 3> &d,
               il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (delta[i] / d[i]) * cosAlpha[i];
  }
  return -(1.0 / 3.0) * sum;
}

double i7_Xi_Xi_Aux(double &eta, il::StaticArray<double, 3> &cosAlpha,
                    il::StaticArray<double, 3> &Lambda,
                    il::StaticArray<double, 3> &sinAlpha,
                    il::StaticArray<double, 3> &q,
                    il::StaticArray<double, 3> &d,
                    il::StaticArray<double, 3> &D,
                    il::StaticArray<double, 3> &delta) {
  double sum1 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum1 += (cosAlpha[i] * Lambda[i] +
             sinAlpha[i] * (q[i] / d[i]) * (D[i] + (2.0 / d[i]) * delta[i])) *
            sinAlpha[i];
  }
  double sum2 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum2 += (q[i] / d[i]) * delta[i];
  }
  return -(pow(eta, 2.0) / 15.0) * sum1 + (1.0 / 15.0) * sum2;
}

double i5_Zeta_Zeta_Aux(il::StaticArray<double, 3> &L,
                        il::StaticArray<double, 3> &sinAlpha,
                        il::StaticArray<double, 3> &q,
                        il::StaticArray<double, 3> &d,
                        il::StaticArray<double, 3> &delta,
                        il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (L[i] * sinAlpha[i] - (q[i] / d[i]) * delta[i] * cosAlpha[i]) *
           cosAlpha[i];
  }
  return (1.0 / 3.0) * sum;
}

double i7_Xi_Zeta(il::StaticArray<double, 3> &sinAlpha,
                  il::StaticArray<double, 3> &Lambda,
                  il::StaticArray<double, 3> &cosAlpha,
                  il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                  il::StaticArray<double, 3> &D,
                  il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (sinAlpha[i] * Lambda[i] -
            cosAlpha[i] * (q[i] / d[i]) * (D[i] + (2.0 / d[i]) * delta[i])) *
           sinAlpha[i];
  }
  return -(1.0 / 15.0) * sum;
}

double i5_Xi_Zeta(il::StaticArray<double, 3> &L,
                  il::StaticArray<double, 3> &sinAlpha,
                  il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                  il::StaticArray<double, 3> &delta,
                  il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (L[i] * sinAlpha[i] - (q[i] / d[i]) * delta[i] * cosAlpha[i]) *
           sinAlpha[i];
  }
  return -(1.0 / 3.0) * sum;
}

// By order of appearance in stress influence coefficients due to DD2

double i7_Zeta_Zeta_Zeta(il::StaticArray<double, 3> &q,
                         il::StaticArray<double, 3> &d,
                         il::StaticArray<double, 3> &sinAlpha,
                         il::StaticArray<double, 3> &cosAlpha,
                         il::StaticArray<double, 3> &D,
                         il::StaticArray<double, 3> &Lambda,
                         il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (((pow(q[i], 2.0) / d[i]) * pow(cosAlpha[i], 2.0) -
             pow(sinAlpha[i], 2.0)) *
                D[i] -
            2.0 * sinAlpha[i] * cosAlpha[i] * q[i] * Lambda[i] +
            (1.0 / d[i]) *
                (3.0 - pow(cosAlpha[i], 2.0) +
                 2.0 * (pow(q[i], 2.0) / d[i]) * pow(cosAlpha[i], 2.0)) *
                delta[i]) *
           cosAlpha[i];
  }
  return -(1.0 / 15.0) * sum;
}

double i7_Zeta(il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
               il::StaticArray<double, 3> &delta,
               il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (1.0 / d[i]) * (D[i] + (2.0 / d[i]) * delta[i]) * cosAlpha[i];
  }
  return -(1.0 / 15.0) * sum;
}

double i7_Zeta_Zeta_Aux(double &eta, il::StaticArray<double, 3> &cosAlpha,
                        il::StaticArray<double, 3> &Lambda,
                        il::StaticArray<double, 3> &sinAlpha,
                        il::StaticArray<double, 3> &q,
                        il::StaticArray<double, 3> &d,
                        il::StaticArray<double, 3> &D,
                        il::StaticArray<double, 3> &delta) {
  double sum1 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum1 += (sinAlpha[i] * Lambda[i] -
             cosAlpha[i] * (q[i] / d[i]) * (D[i] + (2.0 / d[i]) * delta[i])) *
            cosAlpha[i];
  }
  double sum2 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum2 += (q[i] / d[i]) * delta[i];
  }
  return (pow(eta, 2.0) / 15.0) * sum1 + (1.0 / 15.0) * sum2;
}

double i5_Xi_Xi_Aux(il::StaticArray<double, 3> &L,
                    il::StaticArray<double, 3> &sinAlpha,
                    il::StaticArray<double, 3> &q,
                    il::StaticArray<double, 3> &d,
                    il::StaticArray<double, 3> &delta,
                    il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (L[i] * cosAlpha[i] + (q[i] / d[i]) * delta[i] * sinAlpha[i]) *
           sinAlpha[i];
  }
  return -(1.0 / 3.0) * sum;
}

// By order of appearance in stress influence coefficients due to DD3

double i5_Aux(il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
              il::StaticArray<double, 3> &delta) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += (q[i] / d[i]) * delta[i];
  }
  return (1.0 / 3.0) * sum;
}

double i7_Aux(double &eta, il::StaticArray<double, 3> &q,
              il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
              il::StaticArray<double, 3> &delta) {
  double sum1 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum1 += (q[i] / d[i]) * (D[i] + (2.0 / d[i]) * delta[i]);
  }
  double sum2 = 0.0;
  for (int i = 0; i < 3; i++) {
    sum2 += (q[i] / d[i]) * delta[i];
  }
  return (pow(eta, 2.0) / 15.0) * sum1 + (1.0 / 5.0) * sum2;
}

// Fundamental displacement kernel = displacement influence coefficients
il::StaticArray2D<double, 3, 3> DisplacementKernelT0(il::Array2D<double> &x,
                                                     il::Array2D<double> &xv,
                                                     double &nu) {

  // this routine is based on the works of Nintcheu Fata (2009,2011)

  // inputs
  //   -x: source/receiver point's coordinates in the global reference system in
  //   the form (x,y,z) -xv: vertices' coordinates in the global reference
  //   system, in the following 2D array form:
  //     x0 y0 z0
  //     x1 y1 z1
  //     x2 y2 z2
  //   -nu: Poisson's ratio
  // (note the shear modulus does not enter here)
  // output
  //   Displacement = 3 x 3 matrix with the displacement influence coefficients
  //   arranged as: U1, U2, U3 -> for rows DD1 (shear), DD2 (shear), DD3
  //   (normal) -> for columns

  double eps_tol = 2.22045e-16; // parameter used for "if conditions" involving
                                // inequalities due to numerical precision

  // get triangle vertices coordinates separated
  il::StaticArray<double, 3> y1, y2, y3;
  for (il::int_t i = 0; i < 3; i++) {
    y1[i] = xv(0, i);
    y2[i] = xv(1, i);
    y3[i] = xv(2, i);
  }
  // subtractions of coordinates vectors
  il::StaticArray<double, 3> y31, y21, y1x;
  for (il::int_t i = 0; i < 3; i++) {
    y31[i] = y3[i] - y1[i];
    y21[i] = y2[i] - y1[i];
    y1x[i] = y1[i] - x(0, i);
  }
  // local reference system (e1,e2,e3)
  il::StaticArray<double, 3> e1, e2, e3;

  // e1
  e1[0] = y21[0] / il::norm(y21, il::Norm::L2);
  e1[1] = y21[1] / il::norm(y21, il::Norm::L2);
  e1[2] = y21[2] / il::norm(y21, il::Norm::L2);

  // e2
  double sigma = il::dot(y31, y21) / pow(il::norm(y21, il::Norm::L2), 2.0);
  il::StaticArray<double, 3> e2vector; // not normalized
  e2vector[0] = y31[0] - sigma * y21[0];
  e2vector[1] = y31[1] - sigma * y21[1];
  e2vector[2] = y31[2] - sigma * y21[2];
  e2[0] = e2vector[0] / il::norm(e2vector, il::Norm::L2);
  e2[1] = e2vector[1] / il::norm(e2vector, il::Norm::L2);
  e2[2] = e2vector[2] / il::norm(e2vector, il::Norm::L2);

  // e3
  e3 = il::cross(e1, e2);

  // geometrical parameters
  double a, b, c;
  a = il::dot(y31, e2);
  b = il::dot(y21, e1);
  c = il::dot(y31, e1);
  double theta1, theta2, theta3;
  theta1 = acos(c / sqrt(pow(c, 2.0) + pow(a, 2.0)));
  theta2 = acos((b - c) / sqrt(pow(b - c, 2.0) + pow(a, 2.0)));
  theta3 = il::pi - (theta1 + theta2);

  // endpoints' coordinates of each triangle edge for the local coordinate
  // systems of each edge

  // vertex 1 coordinates
  double xi1, zeta1, eta;
  xi1 = il::dot(y1x, e1);
  zeta1 = il::dot(y1x, e2);
  eta = il::dot(y1x, e3);

  // auxiliary angles
  double alpha1, alpha2, alpha3;
  alpha1 = 0.;
  alpha2 = il::pi - theta2;
  alpha3 = il::pi + theta1;

  // endpoints' coordinates edge L1
  double p11, p12, q1;
  p11 = xi1;
  p12 = b + xi1;
  q1 = zeta1;

  // endpoints' coordinates edge L2
  double p22, p23, q2;
  p22 = (b + xi1) * cos(alpha2) + zeta1 * sin(alpha2);
  p23 = (c + xi1) * cos(alpha2) + (a + zeta1) * sin(alpha2);
  q2 = -(b + xi1) * sin(alpha2) + zeta1 * cos(alpha2);

  // endpoints' coordinates edge L3
  double p33, p31, q3;
  p33 = (c + xi1) * cos(alpha3) + (a + zeta1) * sin(alpha3);
  p31 = xi1 * cos(alpha3) + zeta1 * sin(alpha3);
  q3 = -xi1 * sin(alpha3) + zeta1 * cos(alpha3);

  // generic recursive functions

  // previous definitions

  // p
  il::StaticArray2D<double, 3, 3> p;
  p(0, 0) = p11;
  p(1, 1) = p22;
  p(2, 2) = p33;
  p(0, 1) = p12;
  p(1, 2) = p23;
  p(2, 0) = p31;

  // q
  il::StaticArray<double, 3> q;
  q[0] = q1;
  q[1] = q2;
  q[2] = q3;

  // rho - distance between source/receiver point 'x' and i-th vertex of
  // triangle
  il::StaticArray<double, 3> rho;
  for (int i = 0; i < 3; i++) {
    rho[i] = sqrt(pow(p(i, i), 2.0) + pow(q[i], 2.0) + pow(eta, 2.0));
  }

  // now the generic recursive functions

  int i1; // i+1, will be used in all recursive generic functions

  // d
  il::StaticArray<double, 3> d;
  for (int i = 0; i < 3; i++) {
    d[i] = pow(q[i], 2.0) + pow(eta, 2.0);
  }

  //        // rho tilde
  //        il::StaticArray<double, 3> rho_t;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            rho_t[i] = rho[i] - rho[i1];
  //        }

  //        // phi
  //        il::StaticArray<double, 3> phi;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            phi[i] = p(i, i) * rho[i] - p(i, i1) * rho[i1];
  //        }

  // chi
  // Note: in Nintcheu Fata (2009, 2011) there are no comments for the case
  // in which the log arguments are negative or zero and this can happen! Brice
  // regularized this in his mathematica notebook but I don't know yet how
  // (Alexis). I use here Brice's solution
  il::StaticArray<double, 3> chi;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    if ((p(i, i) + rho[i]) > eps_tol && (p(i, i1) + rho[i1]) > eps_tol) {
      // if the log arguments are strictly positive
      chi[i] = log(p(i, i) + rho[i]) - log(p(i, i1) + rho[i1]);
    } else if ((p(i, i) + rho[i]) < -eps_tol &&
               (p(i, i1) + rho[i1]) < -eps_tol) {
      // if the log arguments are strictly negative
      double d = pow(q[i], 2.0) + pow(eta, 2.0);
      double p1 = p(i, i) + rho[i];
      double p2 = p(i, i1) + rho[i1];
      double z1 = (d / p1) / p1;
      double z2 = (d / p2) / p2;
      chi[i] = log(std::abs(p2 / p1) * ((1.0 - 0.25 * z1 * (1.0 - 0.5 * z1)) /
                                        (1.0 - 0.25 * z2 * (1.0 - 0.5 * z2))));
    } else {
      // if any of the log arguments is zero
      chi[i] = 0.0;
    }
  }

  // delta
  il::StaticArray<double, 3> delta;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    delta[i] = p(i, i) / rho[i] - p(i, i1) / rho[i1];
  }

  // L
  il::StaticArray<double, 3> L;
  for (int i = 0; i < 3; i++) {
    i1 = i + 1;
    if (i1 > 2) {
      i1 = 0;
    } // if index is equal to 3, replace by 0
    L[i] = 1.0 / rho[i] - 1.0 / rho[i1];
  }

  //        // D
  //        il::StaticArray<double, 3> D;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            D[i] = p(i, i) / pow(rho[i], 3.0) - p(i, i1) /
  //            pow(rho[i1], 3.0);
  //        }

  //        // Lambda
  //        il::StaticArray<double, 3> Lambda;
  //        for (int i = 0; i < 3; i++) {
  //            i1 = i + 1;
  //            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
  //            Lambda[i] = 1.0 / pow(rho[i], 3.0) - 1.0 / pow(rho[i1], 3.0);
  //        }

  // definition of cases - where the projected source/receiver point lies
  // also, computation of theta_0

  // rho_plane - distance between projected source/receiver point 'x' and i-th
  // vertex of triangle
  il::StaticArray<double, 3> rho_plane;
  for (int i = 0; i < 3; i++) {
    rho_plane[i] = sqrt(pow(p(i, i), 2.0) + pow(q[i], 2.0));
  }

  int id;
  double theta_0;
  if (q[0] > eps_tol || q[1] > eps_tol || q[2] > eps_tol) {
    // if projected point lies strictly outside of the triangle
    id = -1;
    theta_0 = 0.0;
  } else if (q[0] < -eps_tol && q[1] < -eps_tol && q[2] < -eps_tol) {
    // if projected point lies strictly inside of the triangle
    id = 0;
    theta_0 = 2 * il::pi;
  } else if (rho_plane[0] > eps_tol && rho_plane[1] > eps_tol &&
             rho_plane[2] > eps_tol) {
    // if projected point lies on an edge of the triangle but not on a vertex
    id = 4;
    theta_0 = il::pi;
  } else if (rho_plane[0] < eps_tol) {
    // if projected point lies exactly on vertex 1
    id = 1;
    theta_0 = theta1;
  } else if (rho_plane[1] < eps_tol) {
    // if projected point lies exactly on vertex 2
    id = 2;
    theta_0 = theta2;
  } else if (rho_plane[2] < eps_tol) {
    // if projected point lies exactly on vertex 3
    id = 3;
    theta_0 = theta3;
  }

  // gamma

  // Note: in Nintcheu Fata (2009, 2011) there are no comments for the case
  // in which the arctan arguments are not defined (in this case, 0/0) and this
  // can happen! Brice regularized this in his mathematica notebook by taking
  // the limits, I understand the case of the edge, but not the cases for the
  // vertices. I just coded up Brice's solution for now, but I didn't include
  // the case inside the element and the case right at the vertices
  il::StaticArray<double, 3> gamma;

  if (id == 1) {
    // if the projected source/receiver point lies on vertex 1
    gamma[0] = 0.0;
    gamma[2] = 0.0;
    int i = 1;
    i1 = 2;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else if (id == 2) {
    // if the projected source/receiver point lies on vertex 2
    gamma[0] = 0.0;
    gamma[1] = 0.0;
    int i = 2;
    i1 = 0;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else if (id == 3) {
    // if the projected source/receiver point lies on vertex 3
    gamma[1] = 0.0;
    gamma[2] = 0.0;
    int i = 0;
    i1 = 1;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else if (std::abs(q[0]) < eps_tol) {
    // if the projected source/receiver point lies on the edge 1 or its
    // prolongation (in both directions) the limit is needed only for the case
    // eta = 0. However, for eta =! 0, the value of gamma1 is the same as the
    // limit (=zero), so that's why I don't make the difference
    gamma[0] = 0.0;
    int i = 1;
    i1 = 2;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    i = 2;
    i1 = 0;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else if (std::abs(q[1]) < eps_tol) {
    // if the projected source/receiver point lies on the edge 2 or its
    // prolongation (in both directions) the limit is needed only for the case
    // eta = 0. However, for eta =! 0, the value of gamma2 is the same as the
    // limit (=zero), so that's why I don't make the difference
    gamma[1] = 0.0;
    int i = 0;
    i1 = 1;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    i = 2;
    i1 = 0;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else if (std::abs(q[2]) < eps_tol) {
    // if the projected source/receiver point lies on the edge 3 or its
    // prolongation (in both directions) the limit is needed only for the case
    // eta = 0. However, for eta =! 0, the value of gamma3 is the same as the
    // limit (=zero), so that's why I don't make the difference
    gamma[2] = 0.0;
    int i = 0;
    i1 = 1;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    i = 1;
    i1 = 2;
    gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                    (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                     pow(p(i, i), 2.0) * pow(eta, 2.0))) -
               atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                    (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                     pow(p(i, i1), 2.0) * pow(eta, 2.0)));
  } else {
    // any other case
    for (int i = 0; i < 3; i++) {
      i1 = i + 1;
      if (i1 > 2) {
        i1 = 0;
      } // if index is equal to 3, replace by 0
      gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                      (pow(q[i], 2.0) * pow(rho[i], 2.0) -
                       pow(p(i, i), 2.0) * pow(eta, 2.0))) -
                 atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                      (pow(q[i], 2.0) * pow(rho[i1], 2.0) -
                       pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    }
  }

  // theta for displacements and tractions at the boundary of the domain
  double theta;
  double thetaSum = 0.0;
  for (int i = 0; i < 3; i++) {
    thetaSum += gamma[i];
  }
  if (eta < eps_tol) {
    // if eta is negative or zero
    theta = 0.5 * thetaSum - theta_0;
  } else {
    // if eta is strictly positive
    theta = 0.5 * thetaSum + theta_0;
  }

  // sin(alpha[i]) and cos(alpha[i])

  // alpha
  il::StaticArray<double, 3> alpha;
  alpha[0] = alpha1;
  alpha[1] = alpha2;
  alpha[2] = alpha3;

  il::StaticArray<double, 3> sinAlpha;
  il::StaticArray<double, 3> cosAlpha;
  for (int i = 0; i < 3; i++) {
    sinAlpha[i] = sin(alpha[i]);
    cosAlpha[i] = cos(alpha[i]);
  }

  // compute necessary generic (+ auxiliary) integrals
  // in the order they appear in the displacement influence coefficients

  // first, for displacement influence coefficients due to DD1
  double I5_Xi_Xi_Aux = i5_Xi_Xi_Aux(L, sinAlpha, q, d, delta, cosAlpha);
  double I5_Xi_Zeta = i5_Xi_Zeta(L, sinAlpha, q, d, delta, cosAlpha);
  double I3_Xi = i3_Xi(chi, sinAlpha);
  double I5_Xi = i5_Xi(delta, d, sinAlpha);

  // second, for displacement influence coefficients due to DD2

  double I5_Zeta_Zeta_Aux =
      i5_Zeta_Zeta_Aux(L, sinAlpha, q, d, delta, cosAlpha);
  double I3_Zeta = i3_Zeta(chi, cosAlpha);
  double I5_Zeta = i5_Zeta(delta, d, cosAlpha);

  // third, for displacement influence coefficients due to DD3
  double I5_Aux = i5_Aux(q, d, delta);

  // And finally, the displacement influence coefficients

  double prefactor =
      (1.0 /
       (8.0 * il::pi * (1.0 - nu))); // common prefactor of all coefficients

  il::StaticArray2D<double, 3, 3> Displacement; // output

  // Displacement row is dof (DDx,DDy,DDx), columns are Ux,Uy,Uz in the local
  // reference system

  // displacement components due to the unit displacement discontinuity DD1
  // (shear)
  Displacement(0, 0) =
      prefactor * (3.0 * I5_Xi_Xi_Aux * eta - 2.0 * theta * (-1.0 + nu));
  // U1 = a11
  Displacement(1, 0) = prefactor * (3.0 * I5_Xi_Zeta * eta);
  // U2 = a21
  //        Displacement(2, 0) = prefactor * ( I3_Xi + 3.0 * I5_Xi *
  //        pow(eta,2.0) - 2.0 * I3_Xi * nu );
  //        // U3 = a31 without factorization
  Displacement(2, 0) =
      prefactor * ((1.0 - 2.0 * nu) * I3_Xi + 3.0 * I5_Xi * pow(eta, 2.0));
  // U3 = a31

  // displacement components due to the unit displacement discontinuity DD2
  // (shear)
  Displacement(0, 1) = prefactor * (3.0 * I5_Xi_Zeta * eta);
  // U1 = a12
  Displacement(1, 1) =
      prefactor * (3.0 * I5_Zeta_Zeta_Aux * eta - 2.0 * theta * (-1 + nu));
  // U2 = a22
  //        Displacement(2, 1) = prefactor * ( I3_Zeta + 3.0 * I5_Zeta *
  //        pow(eta,2.0) - 2.0 * I3_Zeta * nu );
  //        // U3 = a32 without factorization
  Displacement(2, 1) =
      prefactor * ((1.0 - 2.0 * nu) * I3_Zeta + 3.0 * I5_Zeta * pow(eta, 2.0));
  // U3 = a32

  // displacement components due to the unit displacement discontinuity DD3
  // (normal)
  Displacement(0, 2) =
      prefactor * (3.0 * I5_Xi * pow(eta, 2.0) + I3_Xi * (-1.0 + 2.0 * nu));
  // U1 = a13
  Displacement(1, 2) =
      prefactor * (3.0 * I5_Zeta * pow(eta, 2.0) + I3_Zeta * (-1.0 + 2.0 * nu));
  // U2 = a23
  Displacement(2, 2) =
      prefactor * (3.0 * I5_Aux * eta - 2.0 * theta * (-1.0 + nu));
  // U3 = a33

  return Displacement; // expressed in the reference system of the DD element
  // index        ->    DD1 (shear)    DD2 (shear)     DD3 (normal)
  //   0      -> |       U1,            U1,             U1            |
  //   1      -> |       U2,            U2,             U2            |
  //   2      -> |       U3,            U3,             U3            |
}

// generic (and auxiliary) integrals first appearing for displacement influence
// coefficients

// By order of appearance in displacement influence coefficients due to DD1

double i3_Xi(il::StaticArray<double, 3> &chi,
             il::StaticArray<double, 3> &sinAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += chi[i] * sinAlpha[i];
  }
  return sum;
}

// By order of appearance in displacement influence coefficients due to DD2

double i3_Zeta(il::StaticArray<double, 3> &chi,
               il::StaticArray<double, 3> &cosAlpha) {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += chi[i] * cosAlpha[i];
  }
  return -sum;
}

// il::Array2D<double> change_local_global(const il::Array2D<double> &A,
//                                         il::int_t local_global,
//                                         const il::Array2D<double> &R_source,
//                                         const il::Array2D<double>
//                                         &R_receiver) {
//   // Note: all inputs and output can be static arrays instead, all are 3-by-3
//   // matrices

//   // Input:
//   // - A: is a 3-by-3 matrix such that t = A*d, where t can be the traction
//   // vector (or the displacement vector) the collocation point of the
//   receiver
//   // element (or at the observation point), and d is the displacement
//   // discontinuity vector at the node of the source element. The matrix A is
//   // such that both t and d are expressed in the local reference system of
//   the
//   // source element
//   // - local_global:
//   //      - 0 if d and t want to be expressed at the local reference system
//   of
//   //      the source element an the local reference system of the receiver
//   //      element, respectively
//   //      - 1 if d and t want to be expressed at the global reference system
//   // - R_source: is a 3-by-3 rotation matrix that goes from the global
//   reference
//   // system to the local reference system of the source element
//   // - R_receiver: is a 3-by-3 rotation matrix that goes from the global
//   // reference system to the local reference system of the receiver element
//   //
//   // Output:
//   // The 3-by-3 matrix A (input), but rotated as wished

//   il::Array2D<double> A_rotated{3, 3, 0.};
//   il::Array2D<double> R_source_transposed{3, 3, 0.};

//   // transpose R_source
//   for (il::int_t i = 0; i < 3; i++) {
//     for (il::int_t j = 0; j < 3; j++) {
//       R_source_transposed(j, i) = R_source(i, j);
//     }
//   }

//   if (local_global == 0) // local-local
//   {
//     A_rotated = il::dot(R_source_transposed, A);
//     A_rotated = il::dot(R_receiver, A_rotated);
//   } else // global-global
//   {
//     A_rotated = il::dot(A, R_source);
//     A_rotated = il::dot(R_source_transposed, A_rotated);
//   }

//   return A_rotated;
// }

//    il::Array<double> point_stress_3DT0(il::Array<double> &observ_pt,FaceData
//    &elem_data_s, // source element
//            il::Array<double> &dd,
//            ElasticProperties const &elas_ // elastic properties
//    )
//    {
//        /*
//         * It returns the stress components:
//         * S11, S22, S33, S12, S13, S23
//         * expressed in the global reference system
//         */
//        double G = elas_.shear_modulus(), nu = elas_.poisson_ratio();
//
//        // get coordinates vertices of triangular source element
//        il::Array2D<double> el_vertices_s;
//        el_vertices_s = elem_data_s.getVertices();
//
//        // observation point coordinates from 1D array to 2D array
//        il::Array2D<double> x{1,3,0.};
//        for (int i = 0; i < 3; ++i) { x(0,i) = observ_pt[i];}
//
//        // get stress influence coefficients - in the local reference system
//        of the source element il::StaticArray2D<double, 3, 6>
//        stress_influence; stress_influence =
//        StressesKernelT0(x,el_vertices_s,G,nu);
//        // Note:
//        // Stress is in the reference system of the source element
//        // index        ->    0    1    2    3    4    5
//        // DD1 (shear)  -> | s11, s22, s33, s12, s13, s23  |
//        // DD2 (shear)  -> | s11, s22, s33, s12, s13, s23  |
//        // DD3 (normal) -> | s11, s22, s33, s12, s13, s23  |
//
//        // compute the stress tensor at the observation point in the local
//        reference system of the source element il::Array2D<double>
//        stress_tensor_local{3,3,0.};
//        // S11
//        stress_tensor_local(0,0) = stress_influence(0,0) * dd[0] +
//        stress_influence(1,0) * dd[1] + stress_influence(2,0) * dd[2];
//        // s12
//        stress_tensor_local(0,1) = stress_influence(0,3) * dd[0] +
//        stress_influence(1,3) * dd[1] + stress_influence(2,3) * dd[2];
//        // s13
//        stress_tensor_local(0,2) = stress_influence(0,4) * dd[0] +
//        stress_influence(1,4) * dd[1] + stress_influence(2,4) * dd[2];
//        // s21 = s12
//        stress_tensor_local(1,0) = stress_tensor_local(0,1);
//        // s22
//        stress_tensor_local(1,1) = stress_influence(0,1) * dd[0] +
//        stress_influence(1,1) * dd[1] + stress_influence(2,1) * dd[2];
//        // s23
//        stress_tensor_local(1,2) = stress_influence(0,5) * dd[0] +
//        stress_influence(1,5) * dd[1] + stress_influence(2,5) * dd[2] ;
//        // s31 = s13
//        stress_tensor_local(2,0) = stress_tensor_local(0,2);
//        // s32 = s23
//        stress_tensor_local(2,1) = stress_tensor_local(1,2);
//        // s33
//        stress_tensor_local(2,2) = stress_influence(0,2) * dd[0] +
//        stress_influence(1,2) * dd[1] + stress_influence(2,2) * dd[2] ;
//
//        // rotation matrix from local to global
//        il::Array2D<double> R_source = elem_data_s.rotationMatrix(false); //
//        false: R(l->g)
//        // transpose of rotation matrix, or from global to local
//        il::Array2D<double> R_source_transposed =
//        elem_data_s.rotationMatrix(true); // true: R(g->l)
//
////        il::Array2D<double> stress_tensor_global =
/// il::dot(R_source_transposed, il::dot(stress_tensor_local, R_source));
//        il::Array2D<double> stress_tensor_global = il::dot(R_source,
//        il::dot(stress_tensor_local, R_source_transposed));
//
//        il::Array<double> stress_at_point{6,0.};
//        stress_at_point[0] = stress_tensor_global(0,0) ; // s11
//        stress_at_point[1] = stress_tensor_global(1,1) ; // s22
//        stress_at_point[2] = stress_tensor_global(2,2) ; // s33
//        stress_at_point[3] = stress_tensor_global(0,1) ; // s12
//        stress_at_point[4] = stress_tensor_global(0,2) ; // s13
//        stress_at_point[5] = stress_tensor_global(1,2) ; // s23
//
//        return stress_at_point;
//    }

// il::Array<double>
// point_displacement_3DT0(il::Array<double> &observ_pt,
//                         FaceData &elem_data_s, // source element
//                         il::Array<double> &dd,
//                         ElasticProperties const &elas_ // elastic properties
// ) {
//   /*
//    * It returns the displacement components:
//    * u_xx, u_yy, u_zz
//    * expressed in the global reference system
//    *
//    */

//   double G = elas_.shear_modulus(), nu = elas_.poisson_ratio();

//   // get coordinates vertices of triangular source element
//   il::Array2D<double> el_vertices_s;
//   el_vertices_s = elem_data_s.getVertices();

//   // observation point coordinates from 1D array to 2D array
//   il::Array2D<double> x{1, 3, 0.};
//   for (int i = 0; i < 3; ++i) {
//     x(0, i) = observ_pt[i];
//   }

//   // get stress influence coefficients - in the local reference system of the
//   // source element
//   il::StaticArray2D<double, 3, 3> disp_influence_aux =
//       DisplacementKernelT0(x, el_vertices_s, nu);
//   il::Array2D<double> disp_influence{3, 3};
//   for (int j = 0; j < 3; j++) {
//     for (int i = 0; i < 3; i++) {
//       disp_influence(i, j) = disp_influence_aux(i, j);
//     }
//   }
//   // Note:
//   // expressed in the reference system of the DD element
//   // index        ->    DD1 (shear)    DD2 (shear)     DD3 (normal)
//   //   0      -> |       U1,            U1,             U1            |
//   //   1      -> |       U2,            U2,             U2            |
//   //   2      -> |       U3,            U3,             U3            |

//   // Apply immediately the DD in order to get a displacement vector in the
//   // reference system local to the source element
//   il::Array<double> displacement_at_point_local = il::dot(disp_influence,
//   dd);

//   // rotation matrix from local to global
//   il::Array2D<double> R_source =
//       elem_data_s.rotationMatrix(false); // false: R(l->g)
//   // transpose of rotation matrix, or from global to local
//   il::Array2D<double> R_source_transposed =
//       elem_data_s.rotationMatrix(true); // true: R(g->l)

//   // Get the displacements in the global reference system
//   il::Array<double> displacement_at_point =
//       il::dot(R_source_transposed, displacement_at_point_local);

//   return displacement_at_point;
// }
} // namespace bie

//
// This file part of BigWham
//
// Created by D. Nikolski on 1/24/2017.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2017.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//

// Integration of a kernel of the elasticity equation
// over a part of a polygonal element (a sector associated with one edge)
// with 2nd order polynomial approximating (shape) functions.

#include <iostream>
#include <complex>
//#include <il/math.h>
#include <elasticity/3d/constants.h>
#include <elasticity/3d/tensor_utilities_3DT6.h>
#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/linearAlgebra.h>
#include "core/oldies/element_utilities.h"

#include <elasticity/3d/Elastic3DT6_element.h>
#include <elasticity/3d/h_potential_3DT6.h>

namespace bie {

// Element-to-point influence matrix (submatrix of the global one)
// (Integration of a kernel of the elasticity equation over a triangular element
// with 2nd order polynomial approximating (shape) functions)
    il::StaticArray2D<double, 6, 18>  make_local_3dbem_submatrix(const int kernel_id,double shear_m, double poiss_r, double h, std::complex<double> z,
                                       const il::StaticArray<std::complex<double>, 3> &tau,
                                       const il::StaticArray2D<std::complex<double>, 6, 6> &sfm) {

        // This function assembles a local "stiffness" sub-matrix
        // (influence of DD at the element nodes to stresses at the point z)
        // in terms of a triangular element's local coordinates
        //
        // tau (3) are coordinates of element's vertices and
        // the rows of sfm (6*6) are coefficients of shape functions
        // in terms of the element's own local coordinate system (tau-coordinates);
        // h and z define the position of the (collocation) point x
        // in the same coordinates

        il::StaticArray2D<double, 6, 18> stress_el_2_el_infl{0.0};

        // scaling ("-" sign comes from traction Somigliana ID, H-term)
        double scale = - shear_m / (4.0 * il::pi * (1.0 - poiss_r));

        // tz[m] and d[m] can be calculated here
        // we are expressing tau in the local coordinate system shifted to z
        il::StaticArray<std::complex<double>, 3> tz, d, dtau;
        std::complex<double> ntau2;
        for (int j = 0; j < 3; ++j) {
            int q = (j + 1) % 3; //--> j=0 , q=1
                                 //--> j=1 , q=2
                                 //--> j=2 , q=0
            tz[j] = tau[j] - z;
            dtau[j] = tau[q] - tau[j]; // edge tau_j_q
            ntau2 = dtau[j] / conj(dtau[j]);
            d[j] = 0.5 * (tz[j] - ntau2 * conj(tz[j]));
        }
        // also, "shifted" sfm from z, tau[m], and local sfm
        il::StaticArray2D<std::complex<double>, 6, 6> shft_z = shift_el_sfm(z);
        il::StaticArray2D<std::complex<double>, 6, 6> sfm_z = dot(sfm, shft_z);

        // searching for "degenerate" edges:
        // point x (collocation pt) projects onto an edge line or a vertex
        bool IsDegen = il::abs(d[0]) < bie::h_tol || abs(d[1]) < bie::h_tol || // moved form abs to il::abs() CP2021
                il::abs(d[2]) < bie::h_tol; // (d[0]*d[1]*d[2]==0); // moved form abs to il::abs() CP2021
        il::StaticArray2D<bool, 2, 3> is_90_ang{false};

        // calculating angles (phi, psi, chi)
        il::StaticArray<double, 3> phi{0.0}, psi{0.0};
        il::StaticArray2D<double, 2, 3> chi{0.0};
        for (int j = 0; j < 3; ++j) {
            phi[j] = arg(tz[j]); // 3 angles one for each vertex
            psi[j] = arg(d[j]);
        }
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                int q = (j + k) % 3;
                chi(k, j) = phi[q] - psi[j];
                // make sure it's between -pi and pi (add or subtract 2*pi)
                if (chi(k, j) <= -il::pi)
                    while (chi(k, j) <= -il::pi)
                        chi(k, j) += 2.0 * il::pi;
                else if (chi(k, j) > il::pi)
                    while (chi(k, j) > il::pi)
                        chi(k, j) -= 2.0 * il::pi;
                //double sin_mon = 1.0 - std::fabs(std::sin(chi(k, j)));
                double com_chi = 0.5 * il::pi - fabs(chi(k, j));
                // reprooving for "degenerate" edges
                // (chi angles too close to 90 degrees)
                if (fabs(com_chi) < bie::a_tol) {
                    //if (std::fabs(sin_mon) < h_tol) {
                    is_90_ang(k, j) = true;
                    IsDegen = true;
                }
            }
        }

        // DD-to-stress influence
        // [(S11+S22)/2; (S11-S22)/2+i*S12; (S13+i*S23)/2; S33]
        // vs SF monomials (s_ij_infl_mon) and nodal values (s_ij_infl_nod)
        il::StaticArray3D<std::complex<double>, 6, 4, 3> s_ij_infl_mon{0.0};
             //s_ij_infl_mon(..?..,number of stresses,..?.. )
        // summation over edges
        for (int m = 0; m < 3; ++m) {
            int n = (m + 1) % 3;//--> m=0 , n=1
                                //--> m=1 , n=2
                                //--> m=2 , n=0
            std::complex<double> dm = d[m];
            if (il::abs(dm) >= bie::h_tol && !is_90_ang(0, m) && !is_90_ang(1, m)) { // moved form abs to il::abs() CP2021
                std::complex<double>
                        // exp(I * chi(0, m))
                        eixm = exp(std::complex<double>(0.0, chi(0, m))),
                // exp(I * chi(1, m))
                        eixn = exp(std::complex<double>(0.0, chi(1, m)));
                // limit case (point x on the element's plane)
                if (fabs(h) < bie::h_tol) {
                    il::StaticArray3D<std::complex<double>, 6, 4, 3> s_incr_n =bie::s_integral_lim(kernel_id, poiss_r, eixn, dm),
                            s_incr_m =bie::s_integral_lim(kernel_id, poiss_r, eixm, dm);
                    for (int l = 0; l < 3; ++l){
                        for (int k = 0; k < 4; ++k) {
                            for (int j = 0; j < 6; ++j)  {
                                s_ij_infl_mon(j, k, l) += s_incr_n(j, k, l) - s_incr_m(j, k, l);
                            }
                        }
                    }
                } else { // out-of-plane case
                    double an = il::abs(tz[n] - dm), // moved form abs to il::abs() CP2021
                            am = il::abs(tz[m] - dm); // moved form abs to il::abs() CP2021
                    an = (chi(1, m) < 0) ? -an : an;
                    am = (chi(0, m) < 0) ? -am : am;
                    // constituing functions of the integrals
                    il::StaticArray<std::complex<double>, 9>
                            f_n = bie::integral_cst_fun(h, dm, an, chi(1, m), eixn),
                            f_m = bie::integral_cst_fun(h, dm, am, chi(0, m), eixm);
                    // coefficients, by 2nd index:
                    // 0: S11+S22; 1: S11-S22+2*I*S12; 2: S13+S23; 3: S33
                    il::StaticArray4D<std::complex<double>, 6, 4, 3, 9>
                            c_n = bie::s_integral_gen(kernel_id, poiss_r, eixn, h, dm),
                            c_m = bie::s_integral_gen(kernel_id, poiss_r, eixm, h, dm);
                    // combining constituing functions & coefficients
                    blas(1.0, c_n, f_n, 1.0, il::io, s_ij_infl_mon);
                    blas(-1.0, c_m, f_m, 1.0, il::io, s_ij_infl_mon);
                    // additional terms for "degenerate" case
                    if (IsDegen) {
                        std::complex<double>
                        // exp(I * phi[n])
                                eipn = exp(std::complex<double>(0.0, phi[n])),
                        // exp(I * phi[m])
                                eipm = exp(std::complex<double>(0.0, phi[m]));
                        il::StaticArray<std::complex<double>, 5>
                                f_n_red = bie::integral_cst_fun_red(h, dm, an),
                                f_m_red = bie::integral_cst_fun_red(h, dm, am);
                        il::StaticArray4D<std::complex<double>, 6, 4, 3, 5>
                                c_n_red = bie::s_integral_red(kernel_id, poiss_r, eipn, h),
                                c_m_red = bie::s_integral_red(kernel_id, poiss_r, eipm, h);
                        blas(1.0, c_n_red, f_n_red, 1.0,il::io, s_ij_infl_mon);
                        blas(-1.0, c_m_red, f_m_red, 1.0,il::io, s_ij_infl_mon);
                    }
                }
            }
        }

        // contraction with "shifted" sfm (left)
        il::StaticArray3D<std::complex<double>, 6, 4, 3>  s_ij_infl_nod = dot(sfm_z, s_ij_infl_mon);
        //s_ij_infl_nod(nodes of source elem,stress component in complex not., dd compoment)

        // re-shaping and scaling (by elastic properties) of the resulting matrix
        for (int j = 0; j < 6; ++j) //looping over the nodes of the source elem
            {
            int q = j * 3;
            for (int k = 0; k < 3; ++k) //looping over the components of dd
            {
                // [S11; S22; S33; S12; S13; S23] vs \delta{u}_k at j-th node
                stress_el_2_el_infl(0, q + k) =scale * (real(s_ij_infl_nod(j, 0, k)) + real(s_ij_infl_nod(j, 1, k)));
                stress_el_2_el_infl(1, q + k) =scale * (real(s_ij_infl_nod(j, 0, k)) - real(s_ij_infl_nod(j, 1, k)));
                stress_el_2_el_infl(2, q + k) =scale * real(s_ij_infl_nod(j, 3, k));
                stress_el_2_el_infl(3, q + k) =scale * imag(s_ij_infl_nod(j, 1, k));
                stress_el_2_el_infl(4, q + k) =scale * 2.0 * real(s_ij_infl_nod(j, 2, k));
                stress_el_2_el_infl(5, q + k) =scale * 2.0 * imag(s_ij_infl_nod(j, 2, k));
            }
        }
        return stress_el_2_el_infl;
        // First index is the stress component [S11; S22; S33; S12; S13; S23]
        // second index is [DD1_1 DD1_2 DD1_3 DD2_1 DD2_2 ... DD6_3] where DDj_k
        // means that it's DD at node j in direction k.
    }


// Coefficient matrices (rank 3) to be contracted with the vector of
// constituing functions defined below (via right multiplication)
// and with the vector of shape function coefficients
// associated with each node of the element (via left multiplication)
//
// Stress components (vs local Cartesian coordinate system of the element)
// combined as S11+S22, S11-S22+2*I*S12, S13+I*S23, S33

    il::StaticArray4D<std::complex<double>, 6, 4, 3, 9> s_integral_gen(const int kernel_id,double poiss_r, std::complex<double> eix,double h, std::complex<double> d) {

        il::StaticArray4D<std::complex<double>, 6, 4, 3, 9> c;
        switch (kernel_id) {
            case 1:
                c = s_ij_gen_h(poiss_r, eix, h, d);
                break;
            case 0:
                // c = s_ij_gen_t(poiss_r, eix, h, d);
                break;
            default:break;
        }
        return c;
    }


    il::StaticArray4D<std::complex<double>, 6, 4, 3, 5> s_integral_red(const int kernel_id,
             double poiss_r, std::complex<double> eix,double h) {
        il::StaticArray4D<std::complex<double>, 6, 4, 3, 5> c;
        switch (kernel_id) {
            case 1:
                c = s_ij_red_h(poiss_r, eix, h);
                break;
            case 0:
                // c = s_ij_red_t(poiss_r, eix, h, d);
                break;
            default:break;
        }
        return c;
    }

    il::StaticArray3D<std::complex<double>, 6, 4, 3> s_integral_lim(const int kernel_id,
             double poiss_r, std::complex<double> eix,std::complex<double> d) {
        il::StaticArray3D<std::complex<double>, 6, 4, 3> c;
        switch (kernel_id) {
            case 1:
                c = s_ij_lim_h(poiss_r, eix, d);
                break;
            case 0:
                    // c = s_ij_lim_t(poiss_r, eix, sgnh, d);
                break;
            default:break;
        }
        return c;
    }


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

// General case (h!=0, collocation point projected into or outside the element)
// powers of r, g0=arctan((ah)/(dr)),
// f0=arctanh(a/r) and its derivatives w.r. to h

    il::StaticArray<std::complex<double>, 9> integral_cst_fun (double h, std::complex<double> d, double a,double x, std::complex<double> eix) {

        double abs_d = std::abs(d), d2 = abs_d * abs_d, a2 = a * a,
                r = std::sqrt(h * h + a2 + d2),
                r2 = r * r, r3 = r2 * r, r5 = r3 * r2,
                ar = a / r, ar2 = ar * ar,
                hr = std::fabs(h / r),
                b = 1.0 / (r2 - a2), b2 = b * b, b3 = b2 * b;
        double tah_x = std::imag(eix) / std::real(eix), tr = hr * tah_x,
                g0 = std::atan(tr), f0 = std::atanh(ar),
                f1 = -0.5 * ar * b, f2 = 0.25 * (3.0 - ar2) * ar * b2,
                f3 = -0.125 * (15.0 - 10.0 * ar2 + 3.0 * ar2 * ar2) * ar * b3;

        il::StaticArray<std::complex<double>, 9> fun_list{0.0};
        fun_list[0] = r;
        fun_list[1] = 1.0 / r;
        fun_list[2] = 1.0 / r3;
        fun_list[3] = 1.0 / r5;
        fun_list[4] = g0 - x;
        fun_list[5] = f0;
        fun_list[6] = f1;
        fun_list[7] = f2;
        fun_list[8] = f3;

        return fun_list;
    }

// Special case (reduced summation,
// collocation point projected onto the element contour) - additional terms

    il::StaticArray<std::complex<double>, 5> integral_cst_fun_red(double h, std::complex<double> d, double a) {

        double h2 = h * h, h4 = h2 * h2, h6 = h4 * h2,
                abs_d = std::abs(d), d2 = abs_d * abs_d, a2 = a * a,
                ro = std::sqrt(a2 + d2),
                r = std::sqrt(h2 + a2 + d2),
                rr = ro / r, rr2 = rr * rr, rr4 = rr2 * rr2,
                f0 = std::atanh(rr), f1 = -0.5 * rr / h2, f2 =
                0.25 * (3.0 - rr2) * rr / h4,
                f3 = -0.125 * (15.0 - 10.0 * rr2 + 3.0 * rr4) * rr / h6;

        il::StaticArray<std::complex<double>, 5> fun_list{0.0};
        fun_list[0] = 1.0;
        fun_list[1] = f0;
        fun_list[2] = f1;
        fun_list[3] = f2;
        fun_list[4] = f3;

        return fun_list;
    }


////////////////////////////////////////////////////////////////////////////////

il::Array2D<double> traction_influence_3DT6(bie::FaceData &elem_data_s, bie::FaceData &elem_data_r,
                                            il::int_t n_s,  // n_s is a node of the "source" element
                                            il::int_t  n_t,  // n_t is the collocation point of the "target" element
                                            bie::ElasticProperties const &elas_,
                                            il::int_t I_want_global_DD,
                                            il::int_t I_want_global_traction) {

  il::Array2D<double> NodeDDtriplet_to_CPtraction_influence_matrix{3,3,0.0};

  // Vertexes coordinates of the source element
    //  x0 x1 x2
    //  y0 y1 y2
    //  z0 z1 z2

  il::StaticArray2D<double, 3, 3> el_vert_s;
  il::Array2D<double> el_vert_s_aux{3,3};
  el_vert_s_aux = elem_data_s.getVertices();
  // now we have to transpose due to different conventions of Mesh3D classes and functions to compute
  // the quadratic triangular element kernel
  for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
          el_vert_s(j,i) = el_vert_s_aux(i, j);
      }
  }

  // Basis (shape) functions and rotation tensor of the el-t
  il::StaticArray2D<double, 3, 3> r_tensor_s;
  il::StaticArray2D<std::complex<double>, 6, 6> sfm = make_el_sfm_uniform(el_vert_s, il::io, r_tensor_s);

  // Complex-valued positions of "source" element nodes
  // from cartesian global coord to local complex repr. coord. (only 1st 2 component
  // the 3rd component is 0 because of the translation)
  il::StaticArray<std::complex<double>, 3> tau = make_el_tau_crd(el_vert_s, r_tensor_s);

  // Vertexes coordinates of the target element
  //  x0 x1 x2
  //  y0 y1 y2
  //  z0 z1 z2

  il::StaticArray2D<double,3 , 3> el_vert_t;
  il::Array2D<double> el_vert_t_aux{3, 3};
  el_vert_t_aux = elem_data_r.getVertices();
  // now we have to transpose due to different conventions of Mesh3D/TriangularElementData classes and functions to compute
  // the quadratic triangular element kernel
  for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
          el_vert_t(j,i) = el_vert_t_aux(i, j);
      }
  }

  // Rotation tensor for the target element (the same function as for the source is being used)
  il::StaticArray2D<double, 3, 3> r_tensor_t =make_el_r_tensor(el_vert_t);

  // Take the Normal vector (n) at collocation point (x) from the r_tensor_t
  // minus because of n:=-e3
  il::StaticArray<double, 3> nrm_cp_glob;
  for (int j = 0; j < 3; ++j) {
    nrm_cp_glob[j] = -r_tensor_t(2, j);
  }

  // Collocation points' coordinates  (in the Global coordinate system)
  il::StaticArray<il::StaticArray<double, 3>, 6> el_cp_crd = el_cp_uniform(el_vert_t, elem_data_r.getBeta1(), elem_data_r.getBeta2());

  // Shifting to the n_t-th collocation pt
  HZ hz = make_el_pt_hz(el_vert_s, el_cp_crd[n_t], r_tensor_s);

  //
  // Calculating DD-to stress influence
  // w.r. to the source element's local coordinate system (both DD and stresses!)
  il::StaticArray2D<double, 6, 18> stress_infl_el2p_loc_h =make_local_3dbem_submatrix(1,  elas_.shear_modulus(), elas_.poisson_ratio(), hz.h, hz.z, tau, sfm);

  // Alternative 2: rotating nrm_cp_glob to
  // the source element's local coordinate system
  // because I need the normal ar the cp in the source coordinates system
  il::StaticArray<double, 3> nrm_cp_loc =il::dot(r_tensor_s, nrm_cp_glob);
  // nrm_cp_glob is the normal @ cp in the global coord system
  // Expressing it through the coord. system of the source el.
  // you get nrm_cp_loc

  il::StaticArray2D<double, 3, 18> trac_el2p_loc =nv_dot_sim(nrm_cp_loc, stress_infl_el2p_loc_h);

  // Computing the traction vector @ cp
  // by multiplying the stress tensor (is in the coord. system of the
  // source element) with the normal in local coord of the source element.
  // we obtain traction in the collocation point in the local system of the source point
  //
  // trac_el2p_loc--> traction and DD are both in local system of the SOURCE element
  //

  il::StaticArray2D<double, 3, 18> trac_cp_local2global =il::dot(r_tensor_s, il::Dot::Transpose,trac_el2p_loc);
  // 3 component of traction vs 6 source nodes * 3 DD components
  // so 3*18 <(1 coll. point vs 6 source nodes)>
  //
  // trac_cp_local2global--> traction and DD are in the local system of the source element ,traction are in global coord. system
  // (for all the nodes of the source)
  //
  //
  // taking a block (one node of the "source" element)
  // traction matrix
  // t_dir_x_node(n_t)_dd1_on_node_(n_s)  t_dir_x_node(n_t)_dd2_on_node_(n_s)  t_dir_x_node(n_t)_dd3_on_node_(n_s)
  // t_dir_y_node(n_t)_dd1_on_node_(n_s)  t_dir_y_node(n_t)_dd2_on_node_(n_s)  t_dir_y_node(n_t)_dd3_on_node_(n_s)
  // t_dir_z_node(n_t)_dd1_on_node_(n_s)  t_dir_z_node(n_t)_dd2_on_node_(n_s)  t_dir_z_node(n_t)_dd3_on_node_(n_s)
  //
  // DD in the coordinate system: local of the source
  // Traction in the coordinate system: global coord system
  // <meaning that the DD have to be expressed in the
  // local reference system and the effect is
  // obtained in the global reference system.>
  //
  il::StaticArray2D<double, 3, 3> trac_infl_n2p_local2global;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      trac_infl_n2p_local2global(k, j) =trac_cp_local2global(k, 3 * n_s + j);
    }
  }
  if (I_want_global_DD==0 && I_want_global_traction==1)
  {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        NodeDDtriplet_to_CPtraction_influence_matrix(k, j)=trac_infl_n2p_local2global(k, j);
      }
    }
  }
  //
  // trac_infl_n2p_local2global--> traction and DD are in the local system of the source element ,traction are in global coord. system
  //
  //

  if (I_want_global_DD==0 && I_want_global_traction==0) {
    il::StaticArray2D<double, 3, 3> trac_infl_n2p_local2local;
    trac_infl_n2p_local2local = il::dot(r_tensor_t,trac_infl_n2p_local2global);
    //
    // trac_infl_n2p_local2local--> traction and DD are in the local system of the source element ,traction are in local system of the target
    //
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        NodeDDtriplet_to_CPtraction_influence_matrix(k, j)=trac_infl_n2p_local2local(k, j);
      }
    }
  }

  if  (I_want_global_DD==1 && I_want_global_traction==0)
  {
    // Coordinate rotation (for the unknown DD)
    // We transform the DD from local to global,
    // meaning that the DD have to be expressed
    // in the global coordinates to obtain global traction.
    il::StaticArray2D<double, 3, 3> trac_infl_n2p_global2local;
    trac_infl_n2p_global2local = il::dot(trac_infl_n2p_local2global,r_tensor_s);
    trac_infl_n2p_global2local = il::dot(r_tensor_t,trac_infl_n2p_global2local);
    //
    // trac_infl_n2p_global2local--> traction and DD are in the global system of the source element ,traction are in local system of the target
    //
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        NodeDDtriplet_to_CPtraction_influence_matrix(k, j)=trac_infl_n2p_global2local(k, j);
      }
    }
  }

  if  (I_want_global_DD==1 && I_want_global_traction==1)
  {
    il::StaticArray2D<double, 3, 3> trac_infl_n2p_global2global;
    trac_infl_n2p_global2global = il::dot(trac_infl_n2p_local2global,r_tensor_s);
    //
    // trac_infl_n2p_global2global--> traction and DD & traction are both in the global system
    //
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        NodeDDtriplet_to_CPtraction_influence_matrix(k, j)=trac_infl_n2p_global2global(k, j);
      }
    }
  }

  return NodeDDtriplet_to_CPtraction_influence_matrix;
  // t_dir_x_node(n_t)_dd1_on_node_(n_s)  t_dir_x_node(n_t)_dd2_on_node_(n_s)  t_dir_x_node(n_t)_dd3_on_node_(n_s)
  // t_dir_y_node(n_t)_dd1_on_node_(n_s)  t_dir_y_node(n_t)_dd2_on_node_(n_s)  t_dir_y_node(n_t)_dd3_on_node_(n_s)
  // t_dir_z_node(n_t)_dd1_on_node_(n_s)  t_dir_z_node(n_t)_dd2_on_node_(n_s)  t_dir_z_node(n_t)_dd3_on_node_(n_s)
}


}

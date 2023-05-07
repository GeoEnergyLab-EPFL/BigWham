//
// This file part of BigWham
//
// Created by Brice Lecampion on 23.05.20.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2020.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_EXE_ELASTIC3DT0_ELEMENT_H
#define BIGWHAM_EXE_ELASTIC3DT0_ELEMENT_H

#include <il/StaticArray2D.h>
#include <il/StaticArray.h>
#include <il/Array.h>
#include <il/Array2D.h>

#include "core/elastic_properties.h"

namespace bie {
//
//    il::Array2D<double> traction_influence_3DT0(
//            bie::FaceData &elem_data_s, // source element
//            bie::FaceData &elem_data_r, // receiver element
//            bie::ElasticProperties const &elas_, // elastic properties
//            il::int_t local_global); // 0 if local-local, 1 if global-global
//
//    il::Array2D<double> displacement_influence_3DT0(
//            bie::FaceData &elem_data_s, // source element
//            bie::FaceData &elem_data_r, // receiver element
//            bie::ElasticProperties const &elas_, // elastic properties
//            il::int_t local_global); // 0 if local-local, 1 if global-global

il::StaticArray2D<double, 3, 6>
StressesKernelT0(il::StaticArray<double, 3> &x,
                 il::StaticArray2D<double, 3, 3> &xv, double &G, double &nu);

// Fundamental displacement kernel = displacement influence coefficients
il::StaticArray2D<double, 3, 3> DisplacementKernelT0(il::Array2D<double> &x,
                                                     il::Array2D<double> &xv,
                                                     double &nu);

// function that modify the kernel to be expressed either in local-local or in
// global-global
il::Array2D<double> change_local_global(const il::Array2D<double> &A,
                                        il::int_t local_global,
                                        const il::Array2D<double> &R_source,
                                        const il::Array2D<double> &R_receiver);

// compute the stress tensor in the global reference system for a given source
// element and dd vector
//    il::Array<double> point_stress_3DT0(il::Array<double> &observ_pt, FaceData
//    &elem_data_s, // source element
//                                        il::Array<double> &dd,
//                                        ElasticProperties const &elas_ //
//                                        elastic properties
//    );

// il::Array<double> point_displacement_3DT0(
//     il::Array<double> &observ_pt,    // coordinates
//     FaceData &elem_data_s,           // source element
//     il::Array<double> &dd,           // dislacement discontinuities components
//     ElasticProperties const &elas_); // elastic properties

// generic integrals -- why the heck these functiomns are interfaced ??

// By order of appearance in stress influence coefficients due to DD1

double i5_Xi(il::StaticArray<double, 3> &delta, il::StaticArray<double, 3> &d,
             il::StaticArray<double, 3> &sinAlpha);

double i7_Xi_Xi_Xi(il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                   il::StaticArray<double, 3> &sinAlpha,
                   il::StaticArray<double, 3> &cosAlpha,
                   il::StaticArray<double, 3> &D,
                   il::StaticArray<double, 3> &Lambda,
                   il::StaticArray<double, 3> &delta);

double i7_Xi_Zeta_Zeta(il::StaticArray<double, 3> &q,
                       il::StaticArray<double, 3> &d,
                       il::StaticArray<double, 3> &sinAlpha,
                       il::StaticArray<double, 3> &cosAlpha,
                       il::StaticArray<double, 3> &D,
                       il::StaticArray<double, 3> &Lambda,
                       il::StaticArray<double, 3> &delta);

double i7_Xi(il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
             il::StaticArray<double, 3> &delta,
             il::StaticArray<double, 3> &sinAlpha);

double i7_Xi_Xi_Zeta(il::StaticArray<double, 3> &q,
                     il::StaticArray<double, 3> &d,
                     il::StaticArray<double, 3> &sinAlpha,
                     il::StaticArray<double, 3> &cosAlpha,
                     il::StaticArray<double, 3> &D,
                     il::StaticArray<double, 3> &Lambda,
                     il::StaticArray<double, 3> &delta);

double i5_Zeta(il::StaticArray<double, 3> &delta, il::StaticArray<double, 3> &d,
               il::StaticArray<double, 3> &cosAlpha);

double i7_Xi_Xi_Aux(double &eta, il::StaticArray<double, 3> &cosAlpha,
                    il::StaticArray<double, 3> &Lambda,
                    il::StaticArray<double, 3> &sinAlpha,
                    il::StaticArray<double, 3> &q,
                    il::StaticArray<double, 3> &d,
                    il::StaticArray<double, 3> &D,
                    il::StaticArray<double, 3> &delta);

double i5_Zeta_Zeta_Aux(il::StaticArray<double, 3> &L,
                        il::StaticArray<double, 3> &sinAlpha,
                        il::StaticArray<double, 3> &q,
                        il::StaticArray<double, 3> &d,
                        il::StaticArray<double, 3> &delta,
                        il::StaticArray<double, 3> &cosAlpha);

double i7_Xi_Zeta(il::StaticArray<double, 3> &sinAlpha,
                  il::StaticArray<double, 3> &Lambda,
                  il::StaticArray<double, 3> &cosAlpha,
                  il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                  il::StaticArray<double, 3> &D,
                  il::StaticArray<double, 3> &delta);

double i5_Xi_Zeta(il::StaticArray<double, 3> &L,
                  il::StaticArray<double, 3> &sinAlpha,
                  il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
                  il::StaticArray<double, 3> &delta,
                  il::StaticArray<double, 3> &cosAlpha);

// By order of appearance in stress influence coefficients due to DD2

double i7_Zeta_Zeta_Zeta(il::StaticArray<double, 3> &q,
                         il::StaticArray<double, 3> &d,
                         il::StaticArray<double, 3> &sinAlpha,
                         il::StaticArray<double, 3> &cosAlpha,
                         il::StaticArray<double, 3> &D,
                         il::StaticArray<double, 3> &Lambda,
                         il::StaticArray<double, 3> &delta);

double i7_Zeta(il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
               il::StaticArray<double, 3> &delta,
               il::StaticArray<double, 3> &cosAlpha);

double i7_Zeta_Zeta_Aux(double &eta, il::StaticArray<double, 3> &cosAlpha,
                        il::StaticArray<double, 3> &Lambda,
                        il::StaticArray<double, 3> &sinAlpha,
                        il::StaticArray<double, 3> &q,
                        il::StaticArray<double, 3> &d,
                        il::StaticArray<double, 3> &D,
                        il::StaticArray<double, 3> &delta);

double i5_Xi_Xi_Aux(il::StaticArray<double, 3> &L,
                    il::StaticArray<double, 3> &sinAlpha,
                    il::StaticArray<double, 3> &q,
                    il::StaticArray<double, 3> &d,
                    il::StaticArray<double, 3> &delta,
                    il::StaticArray<double, 3> &cosAlpha);

// By order of appearance in stress influence coefficients due to DD3

double i5_Aux(il::StaticArray<double, 3> &q, il::StaticArray<double, 3> &d,
              il::StaticArray<double, 3> &delta);

double i7_Aux(double &eta, il::StaticArray<double, 3> &q,
              il::StaticArray<double, 3> &d, il::StaticArray<double, 3> &D,
              il::StaticArray<double, 3> &delta);

// generic (and auxiliary) integrals first appearing for displacement influence
// coefficients

// By order of appearance in displacement influence coefficients due to DD1

double i3_Xi(il::StaticArray<double, 3> &chi,
             il::StaticArray<double, 3> &sinAlpha);

// By order of appearance in displacement influence coefficients due to DD2

double i3_Zeta(il::StaticArray<double, 3> &chi,
               il::StaticArray<double, 3> &cosAlpha);
} // namespace bie

#endif // BIGWHAM_EXE_ELASTIC3DT0_ELEMENT_H

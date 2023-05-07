//
// This file part of BigWham
//
// Created by Brice Lecampion on 04.02.19.
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details. 
//


#include <il/StaticArray2D.h>
#include "core/oldies/FaceData.h"
#include "core/elastic_properties.h"


namespace bie{
il::StaticArray2D<double, 3, 6> StressesKernelR0(
    double& xx, double& yy, double& zz, double& a, double& b, double& G,
    double& nu) ;



il::Array2D<double> traction_influence_3DR0(
            bie::FaceData &elem_data_s, // source element
            bie::FaceData &elem_data_r, // receiver element
            bie::ElasticProperties const &elas_, // elastic properties
            il::int_t I_want_global_DD,
            il::int_t I_want_global_traction) ;


il::Array2D<double> DisplacementKernelR0(
        double& xx, double& yy, double& zz, double& a, double& b,
        double& nu) ;


il::Array2D<double> displacement_influence_3DR0(
            bie::FaceData &elem_data_s, // source element
            bie::FaceData &elem_data_r, // receiver element
            bie::ElasticProperties const &elas_, // elastic properties
            il::int_t I_want_global_DD,
            il::int_t I_want_global_displacement) ;

il::Array<double> point_stress_3DR0(
            il::Array<double> &observ_pt, // coordinates
            FaceData &elem_data_s, // source element
            il::Array<double> &dd, // dislacement discontinuities components
            ElasticProperties const &elas_); // elastic properties

il::Array<double> point_displacement_3DR0(
            il::Array<double> &observ_pt, // coordinates
            FaceData &elem_data_s, // source element
            il::Array<double> &dd, // dislacement discontinuities components
            ElasticProperties const &elas_); // elastic properties

}

//
// Created by Alexis Sáez Uribe on 30/05/2022.
//

#ifndef BIGWHAM_ELASTICAXISYM3DP0_ELEMENT_H
#define BIGWHAM_ELASTICAXISYM3DP0_ELEMENT_H

#pragma once

#include <il/StaticArray2D.h>
#include <il/StaticArray.h>

#include <src/core/ElasticProperties.h>
#include <src/core/Mesh2D.h>
#include <src/core/SegmentData.h>

namespace bie {

    il::StaticArray2D<double, 2, 2> traction_influence_AxiSym3DP0(
            SegmentData &source_elt, SegmentData &receiver_elt, const ElasticProperties &Elas);

    double stress_disk_dislocation( double rObs, double rSrc );

}

#endif //BIGWHAM_ELASTICAXISYM3DP0_ELEMENT_H

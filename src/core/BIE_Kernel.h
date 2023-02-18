//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 12.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland, Geo-Energy Laboratory, 2016-2023.  All rights reserved.
// See the LICENSE.TXT file for more details.
//


#ifndef BIGWHAM_BIE_KERNEL_H
#define BIGWHAM_BIE_KERNEL_H
#pragma once

#include <string>

#include <src/core/BoundaryElement.h>

namespace bie{


template<typename T,class Es,class Er>
class BIE_Kernel {
protected:

    il::int_t dof_dimension_{1};
    il::int_t dim_{2};

public:

    //  constructor
    BIE_Kernel()  {};
    ~BIE_Kernel() {};

    // methods
    virtual std::vector<T>  influence(Es source_elt,il::int_t i_s,Er receiver_elt, il::int_t i_r) const =0 ;

    il::int_t getDofDimension() const  {return dof_dimension_;};
    il::int_t getSpatialDimension() const {return dim_;};

};


}
#endif //BIGWHAM_BIE_KERNEL_H

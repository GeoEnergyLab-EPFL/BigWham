//
// This file is part of BigWham.
//
// Created by Francois Fayard - 2018
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
#pragma once

#include <il/Array2D.h>

namespace il {

template <typename T>
struct LowRank {
  il::Array2D<T> A;
  il::Array2D<T> B;
};

}  // namespace il
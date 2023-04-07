//
// This file is part of BigWham.
//
// Created by Francois Fayard - 2018
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
#pragma once

#include <il/core.h>

namespace bie {

enum class HMatrixType { FullRank, LowRank, Hierarchical, FullLu };

struct SubHMatrix {
  il::Range range0;
  il::Range range1;
  bie::HMatrixType type;
};

}  // namespace bie

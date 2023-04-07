//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 08.09.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_TOHPATTERN_H
#define BIGWHAM_TOHPATTERN_H

#include <il/Array2D.h>
#include <il/Tree.h>

#include "hmat/hmatrix/HMatrixType.h"

namespace bie {

// structure for storing the h-mat pattern - note irrespective of the number of
// dofs per nodes
struct HPattern {
  il::Array2D<il::int_t> FRB_pattern; // full rank block pattern
  il::Array2D<il::int_t> LRB_pattern; // low rank block pattern
  // pattern are stored as info a block k at pattern(0-5,k)
  //  spot,i_begin,j_begin,i_end,j_end,flag (0 for full rank, rank for low rank)
  il::int_t n_B{};   // number of blocks in the matrix pattern
  il::int_t n_FRB{}; // number of full rank blocks in the matrix pattern
  il::int_t n_LRB{}; // number of low rank blocks in the matrix pattern

  il::int_t nr{}; // total number of rows in the matrix pattern // be careful
                  // this needs to be set-up separately from the creation
  il::int_t nc{}; // total number of colums in the matrix pattern// be careful
                  // this needs to be set-up separately from the creation
};

// function to get the matrix pattern from the binary cluster tree
HPattern createPattern(const il::Tree<bie::SubHMatrix, 4> &tree);

} // namespace bie

#endif // BIGWHAM_TOHPATTERN_H

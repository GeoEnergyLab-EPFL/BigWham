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
#include <il/Array2DView.h>

namespace bie {

template <typename T>
class MatrixGenerator {
 public:
  virtual il::int_t size(il::int_t d) const = 0;
  virtual il::int_t blockSize() const = 0;
  virtual il::int_t sizeAsBlocks(il::int_t d) const = 0;
  virtual void set(il::int_t b0, il::int_t b1, il::io_t,il::Array2DEdit<T> M) const = 0;
};

template <typename T>
il::Array2D<T> toArray2D(const bie::MatrixGenerator<T>& M) {
  il::Array2D<T> ans{M.size(0), M.size(1)};
  M.set(0, 0, il::io, ans.Edit());
  return ans;
}

}

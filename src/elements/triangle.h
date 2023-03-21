//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 26.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_TRIANGLE_H
#define BIGWHAM_TRIANGLE_H

#include "elements/polygon.h"

namespace bie {

template <int p> class Triangle : public Polygon<p> {
private:
  const double beta_ =
      1.5 *
      0.166666666666667; // collocation points' location parameter for linear
  const double beta1_ = 0.35;  // 1.5 * 0.091576213509771 related to nodes at vertices
                         // (see documentation)
  const double beta2_ = 0.35;  // 1.5 * 0.10810301816807 related to middle-edge nodes
                         // (see documentation)

public:
  Triangle() : Polygon<p>() {
    this->num_vertices_ = 3;
    switch (p) {
    case 0:
      this->num_nodes_ = 1;
      break;
    case 1:
      this->num_nodes_ = this->spatial_dimension_;
      break;
    case 2:
      this->num_nodes_ = 2 * this->spatial_dimension_;
      break;
    }
  }
  ~Triangle() {}

  virtual void set_collocation_points() override;
  virtual void set_nodes() override;
};
} // namespace bie
#endif // BIGWHAM_TRIANGLE_H

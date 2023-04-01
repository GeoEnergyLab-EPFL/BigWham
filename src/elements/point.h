//
// This file is part of BigWham.
//
// Created by Ankit on 29.March.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_POINT_H
#define BIGWHAM_POINT_H

#include "boundary_element.h"

namespace bie {
// dim = 2: for 2 dimensional problems
// dim = 3: for 3 dimensional problems
template <int dim> class Point : public BoundaryElement {
public:
  Point() : BoundaryElement(dim, 0) {
    this->num_vertices_ = 1;
    this->num_nodes_ = 1;
    this->num_collocation_points_ = this->num_nodes_;
    this->vertices_.Resize(this->num_vertices_, dim);
    this->collocation_points_.Resize(this->num_collocation_points_, dim);
    this->nodes_.Resize(this->num_nodes_, dim);
  }
  ~Point() {}

  virtual void SetElement(const il::Array2D<double> &vertices) override;
  virtual void SetRotationMatrices() override;
  virtual il::Array<double>
  ConvertToGlobal(const il::Array<double> &x) const override {
    return x;
  }
  virtual il::Array<double>
  ConvertToLocal(const il::Array<double> &x) const override {
    return x;
  }

protected:
  virtual void SetCollocationPoints();
  virtual void SetNodes();
};

template <int dim>
inline void Point<dim>::SetElement(const il::Array2D<double> &xv) {
  IL_ASSERT(xv.size(0) == num_vertices_);
  this->vertices_ = xv;
  for (int i; i < dim; i++) {
    centroid_[i] = xv(0, i);
  }
  this->SetRotationMatrices();
  this->SetCollocationPoints();
  this->SetNodes();
}

template <int dim> inline void Point<dim>::SetNodes() {
  this->nodes_ = this->collocation_points_;
}

template <int dim> inline void Point<dim>::SetCollocationPoints() {
  this->collocation_points_ = this->vertices_;
}

template <int dim> inline void Point<dim>::SetRotationMatrices() {
  // rotation matrix from e_i to e'_i is by definition R_ij= e'_i . e_j
  for (il::int_t i = 0; i < spatial_dimension_; i++) {
    rotation_matrix_(i, i) = 1.0;
  }

  for (il::int_t i = 0; i < spatial_dimension_; i++) {
    rotation_matrix_t_(i, i) = 1.0;
  }
}

} // namespace bie

#endif // BIGWHAM_POINT_H

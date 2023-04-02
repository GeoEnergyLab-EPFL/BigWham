//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 26.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIGWHAM_SEGMENT_H
#define BIGWHAM_SEGMENT_H

#include <il/Array2D.h>

#include "elements/boundary_element.h"
#include "il/container/2d/Array2D.h"

namespace bie {

template <int p> class Segment : public BoundaryElement {

protected:
  void SetNodes();
  void SetCollocationPoints();

public:
  Segment() : BoundaryElement(2, p) {
    this->num_vertices_ = 2;
    this->num_nodes_ = p + 1;
    this->num_collocation_points_ = this->num_nodes_;

    this->vertices_.Resize(num_vertices_, 2);
    this->collocation_points_.Resize(num_collocation_points_, 2);
    this->nodes_.Resize(num_nodes_, 2);
  }
  ~Segment() {}

  virtual void SetElement(const il::Array2D<double> &vertices) override;
  virtual void SetRotationMatrices() override;

};
/* -------------------------------------------------------------------------- */

template <int p> inline void Segment<p>::SetRotationMatrices() {
  // rotation matrix from e_i to e'_i is by definition R_ij= e'_i . e_j
  for (il::int_t i = 0; i < spatial_dimension_; i++) {
    rotation_matrix_(0, i) = this->tangent1_[i];
    rotation_matrix_(1, i) = this->normal_[i];
  }

  for (il::int_t i = 0; i < spatial_dimension_; i++) {
    rotation_matrix_t_(i, 0) = this->tangent1_[i];
    rotation_matrix_t_(i, 1) = this->normal_[i];
  }
}
/* -------------------------------------------------------------------------- */

template <int p>
inline void Segment<p>::SetElement(const il::Array2D<double> &xv) {
  IL_ASSERT(xv.size(0) == num_vertices_);
  for (il::int_t j = 0; j < spatial_dimension_; j++) {
    this->centroid_[j] =
        0.0; // always reset centroid when setting the coordinates
    for (il::int_t i = 0; i < num_vertices_; i++) {
      this->vertices_(i, j) = xv(i, j);
    }
  }
  for (il::int_t j = 0; j < spatial_dimension_; j++) {
    for (il::int_t i = 0; i < num_vertices_; i++) {
      this->centroid_[j] =
          this->centroid_[j] + this->vertices_(i, j) / num_vertices_;
    }
  }
  for (il::int_t j = 0; j < spatial_dimension_; j++) {
    this->tangent1_[j] = this->vertices_(1, j) - this->vertices_(0, j);
    this->tangent2_[j] = this->vertices_(num_vertices_ - 1, j) - this->vertices_(0, j);
  }
  this->size_ = std::sqrt(il::dot(this->tangent1_, this->tangent1_));
  // unit s and t
  for (il::int_t k = 0; k < spatial_dimension_; ++k) {
    this->tangent1_[k] = this->tangent1_[k] / size_;
    this->tangent2_[k] = this->tangent2_[k] / size_;
  }
  // normal;
  this->normal_[0] = -1. * this->tangent1_[1];
  this->normal_[1] = this->tangent1_[0];

  this->SetRotationMatrices();
  this->SetCollocationPoints();
  this->SetNodes();
}
/* -------------------------------------------------------------------------- */

// Zero order segment
template <> inline void Segment<0>::SetCollocationPoints() {
  // 0 order element: collocation at centroid
  // std::cout << " in set collo ! \n";
  il::Array2D<double> col{1, 2, 0.};
  for (il::int_t j = 0; j < 2; j++) {
    col(0, j) = this->centroid_[j];
  }
  this->collocation_points_ = col;
}
/* -------------------------------------------------------------------------- */

// Linear Segment
template <> inline void Segment<1>::SetCollocationPoints() {
  // on ref element x in [-1,1]   y=0
  // collocation @ -1/sqrt(2), 1/sqrt(2)
  il::Array2D<double> x_col{2, 2, 0.0};
  il::Array<double> xaux{2, 0.0};
  x_col(0, 0) = -1. / sqrt(2.);
  x_col(1, 0) = 1. / sqrt(2.);
  // auto Rt = this->rotationMatrix_T();
  for (int i = 0; i < 2; ++i) {
    xaux[0] = (size_)*x_col(i, 0) / 2.;
    xaux[1] = (size_)*x_col(i, 1) / 2.;
    xaux = this->ConvertToGlobal(xaux.view()); // il::dot(Rt, xaux);
    x_col(i, 0) = xaux[0] + this->centroid_[0];
    x_col(i, 1) = xaux[1] + this->centroid_[1];
  }
  // std::cout << x_col(0, 0) << " " << x_col(0, 1) << " " << x_col(1, 0) << " "
  //           << x_col(1, 1) << "\n";
  this->collocation_points_ = x_col;
}
/* -------------------------------------------------------------------------- */

template <int p> inline void Segment<p>::SetNodes() {
  this->nodes_ = this->collocation_points_; // by default nodes = collocation
                                            // points for 0 element
};

} // namespace bie

#endif // BIGWHAM_SEGMENT_H

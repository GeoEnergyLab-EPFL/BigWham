//
// This file is part of BigWham.
//
// Created by Brice Lecampion on  29.March.2023
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
//

#ifndef BIGWHAM_BIE_MATRIX_GENERATOR_H
#define BIGWHAM_BIE_MATRIX_GENERATOR_H

#include "hmat/arrayFunctor/matrix_generator.h"
#include "hmat/hierarchical_representation.h"

#include "core/bie_kernel.h"
#include "core/elastic_properties.h"
#include "core/mesh.h"
#include "il/core/core.h"

namespace bie {

template <typename T> class BieMatrixGenerator : public MatrixGenerator<T> {

private:
  const std::shared_ptr<Mesh> mesh_src_;
  const std::shared_ptr<Mesh> mesh_rec_;
  const std::shared_ptr<BieKernel<T>> bie_kernel_;

  il::int_t
      block_size_;  // dimension of the kernel (3 for 3D elasticity problems)
  il::int_t size0_; // total rows of matrix
  il::int_t size1_; // total cols of matrix
  il::int_t num_row_points_; // size0_ / block_size_
  il::int_t num_col_points_; // size1_ / block_size_

public:
  BieMatrixGenerator(const std::shared_ptr<Mesh> &mesh_src,
                     const std::shared_ptr<Mesh> &mesh_rec,
                     const std::shared_ptr<BieKernel<T>> &bie_kernel,
                     const std::shared_ptr<HRepresentation> &hr);
  virtual il::int_t size(il::int_t d) const override;
  virtual il::int_t blockSize() const override;
  virtual il::int_t sizeAsBlocks(il::int_t d) const override;
  virtual void set(il::int_t b0, il::int_t b1, il::io_t,
                   il::Array2DEdit<T> M) const override;
};

template <typename T>
inline BieMatrixGenerator<T>::BieMatrixGenerator(
    const std::shared_ptr<Mesh> &mesh_src,
    const std::shared_ptr<Mesh> &mesh_rec,
    const std::shared_ptr<BieKernel<T>> &bie_kernel,
    const std::shared_ptr<HRepresentation> &hr)
    : mesh_src_(mesh_src), mesh_rec_(mesh_rec), bie_kernel_(bie_kernel) {
  this->hr_ = hr;
  num_row_points_ = this->mesh_rec_->num_collocation_points();
  num_col_points_ = this->mesh_src_->num_collocation_points();
  block_size_ = this->bie_kernel_->dof_dimension();
  size0_ = num_row_points_ * block_size_;
  size1_ = num_col_points_ * block_size_;
  // std::cout << "BieMatrixGenerator: shape = " << size0_  
  // << " X  " << size1_ << std::endl;
}

template <typename T>
inline il::int_t BieMatrixGenerator<T>::size(il::int_t d) const {
  il::int_t size;
  if (d == 0) {
    size = size0_;
  } else if (d == 1) {
    size = size1_;
  }
  return size;
}

template <typename T>
inline il::int_t BieMatrixGenerator<T>::blockSize() const {
  return block_size_;
}

template <typename T>
inline il::int_t BieMatrixGenerator<T>::sizeAsBlocks(il::int_t d) const {
  il::int_t num;
  if (d == 0) {
    num = num_row_points_;
  } else if (d == 1) {
    num = num_col_points_;
  }
  return num;
}

template <typename T>
inline void BieMatrixGenerator<T>::set(il::int_t b0, il::int_t b1, il::io_t,
                                       il::Array2DEdit<T> M) const

{
  IL_EXPECT_MEDIUM(M.size(0) % blockSize() == 0);
  IL_EXPECT_MEDIUM(M.size(1) % blockSize() == 0);
  IL_EXPECT_MEDIUM(b0 + M.size(0) / blockSize() <= num_row_points_);
  IL_EXPECT_MEDIUM(b1 + M.size(1) / blockSize() <= num_col_points_);

  il::int_t jj = M.size(1) / blockSize();
#pragma omp parallel if (M.size(1) / blockSize() > 200)
  {
#pragma omp for
    for (il::int_t j1 = 0; j1 < M.size(1) / blockSize(); ++j1) {

      il::int_t k1 = b1 + j1;
      // j1 source node
      // from k1 - permute back to original mesh ordering using permutation
      // of
      // the clusters.
      il::int_t old_k1 = this->hr_->permutation_1_[k1];
      il::int_t e_k1 = this->mesh_src_->GetElementId(old_k1);
      il::int_t is_l = this->mesh_src_->GetElementCollocationId(old_k1);

      auto source_element = this->mesh_src_->GetElement(e_k1);

      for (il::int_t j0 = 0; j0 < M.size(0) / blockSize(); ++j0) {
        il::int_t k0 = b0 + j0;
        il::int_t old_k0 = this->hr_->permutation_0_[k0];
        il::int_t e_k0 =
            this->mesh_rec_->GetElementId(old_k0); //  receiver element
        il::int_t ir_l = this->mesh_rec_->GetElementCollocationId(old_k0);

        auto receiver_element = this->mesh_rec_->GetElement(e_k0);
        std::vector<double> st = this->bie_kernel_->influence(
            *source_element, is_l, *receiver_element,
            ir_l); // column major
        // std::cout << "kernel size =" << st.size() << std::endl;
        // std::cout << "DOF dimension =" << dof_dimension_ << std::endl;
        IL_EXPECT_FAST(st.size() == block_size_ * block_size_);
        il::int_t k = 0;
        for (il::int_t j = 0; j < block_size_; j++) {
          for (il::int_t i = 0; i < block_size_; i++) {
            M(j0 * block_size_ + i, j1 * block_size_ + j) = st[k];
            k++;
          }
        }
      }
    }
  }
}
} // namespace bie
#endif // BIGWHAM_BIE_MATRIX_GENERATOR_H

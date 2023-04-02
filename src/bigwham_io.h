//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 15.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications ::January 31. 2023 - cleaning up and using the new code
// interface.

#ifndef BIGWHAM_BIGWHAMIO_H
#define BIGWHAM_BIGWHAMIO_H

#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Dynamic.h>
#include <il/Map.h>

#include <hmat/hmatrix/hmat.h>

#include "core/be_mesh.h"
// #include "elements/rectangle.h"
#include "elasticity/fullspace_iso_sp3d_segment/bie_elastostatic_sp3d.h"
#include "elements/segment.h"
#include "elements/triangle.h"

#include "core/bie_kernel.h"
#include "core/elastic_properties.h"
#include "elasticity/bie_elastostatic.h"
#include "hmat/bie_matrix_generator.h"

#include "elasticity/fullspace_iso_axisymm_flat_ring_unidirectional/bie_elastostatic_axi3d0.h"
#include "elasticity/fullspace_iso_sp3d_segment/bie_elastostatic_sp3d.h"
// #include "hmat/square_matrix_generator.h"
// #include "elasticity/FsIso2dSegment/BIE_elastostatic_segment_0_impls.h"
// #include "elasticity/FsIso2dSegment/BIE_elastostatic_segment_1_impls.h"
// #include "elasticity/3d/bie_elastostatic_triangle_0_impls.h"
// #include "elasticity/FsIsoAxiFlatRingUnidirectional/ElasticAxi3DP0_element.h"
// #include <elasticity/FsIsoSp3dSegment/BieElastostaticSp3d.h>

// utilities for switch with string in C++17
// https://learnmoderncpp.com/2020/06/01/strings-as-switch-case-labels/
inline constexpr auto hash_djb2a(const std::string_view sv) {
  unsigned long hash{5381};
  for (unsigned char c : sv) {
    hash = ((hash << 5) + hash) ^ c;
  }
  return hash;
}

inline constexpr auto operator"" _sh(const char *str, size_t len) {
  return hash_djb2a(std::string_view{str, len});
}

//////////////////////////// utility for mesh object creation from std::vector
template <class El>
std::shared_ptr<bie::Mesh> createMeshFromVect(int spatial_dimension,
                                              int n_vertex_elt,
                                              const std::vector<double> &coor,
                                              const std::vector<int> &conn) {
  il::int_t npoints = coor.size() / spatial_dimension;
  il::int_t nelts = conn.size() / spatial_dimension;
  il::Array2D<double> Coor{npoints, spatial_dimension, 0.}; //
  il::Array2D<il::int_t> Conn{nelts, n_vertex_elt, 0};
  // populate mesh  ...
  int index = 0;
  for (il::int_t i = 0; i < Coor.size(0); i++) {
    for (il::int_t j = 0; j < Coor.size(1); j++) {
      Coor(i, j) = coor[index];
      index++;
    }
  }
  index = 0;
  for (il::int_t i = 0; i < Conn.size(0); i++) {
    for (il::int_t j = 0; j < Conn.size(1); j++) {
      Conn(i, j) = conn[index];
      index++;
    }
  }
  // BEMesh<El> mesh(Coor, Conn);
  auto mesh = std::make_shared<bie::BEMesh<El>>(Coor, Conn);
  mesh->ConstructMesh();
  std::cout << "in create mesh - done "
            << "\n";
  return mesh;
}
/* -------------------------------------------------------------------------- */

//////////////////////////// the 'infamous' Bigwhamio class
class Bigwhamio {
private:
  bie::Hmat<double> hmat_; // the  Hmat object
  std::shared_ptr<bie::Mesh> mesh_;
  std::shared_ptr<bie::BieKernel<double>> ker_obj_;

  il::Array<il::int_t>
      permutation_; // permutation list of the collocation points
  il::Array2D<double> collocation_points_; //  collocation points coordinates ?

  int dimension_;     // spatial dimension
  int dof_dimension_; // number of dof per nodes / collocation points

  bool is_built_; // True if the class instance is built

  // H-matrix parameters
  int max_leaf_size_;
  double eta_;
  double epsilon_aca_;
  // kernel
  std::string kernel_;

  // statistics
  double h_representation_time_;
  double hmat_time_;

public:
  // default constructor
  Bigwhamio() {
    dimension_ = 0;
    dof_dimension_ = 0;
    is_built_ = false;
    eta_ = 0.0;
    epsilon_aca_ = 0.001;
    max_leaf_size_ = 100;
    kernel_ = "none";
    h_representation_time_ = 0.;
    hmat_time_ = 0.;
  }

  ~Bigwhamio() = default;

  /* --------------------------------------------------------------------------
   */
  void set(const std::vector<double> &coor, const std::vector<int> &conn,
           const std::string &kernel, const std::vector<double> &properties,
           const int max_leaf_size, const double eta, const double eps_aca) {
    // coor and conn are assumed to be passed in row-major storage format
    kernel_ = kernel;
    // switch depending on Kernels for mesh building
    max_leaf_size_ = max_leaf_size;
    eta_ = eta;
    epsilon_aca_ = eps_aca;
    bie::ElasticProperties elas(properties[0], properties[1]);
    std::cout << " Now setting things for kernel ... " << kernel_
              << " with properties size " << properties.size() << "\n";
    il::Timer tt;

    if (kernel_ == "S3DP0") {
      IL_ASSERT(properties.size() == 3);
    } else {
      IL_ASSERT(properties.size() == 2);
    }
    switch (hash_djb2a(kernel_)) {
    case "2DP0"_sh: {
      dimension_ = 2;
      int nvertices_per_elt_ = 2;
      using EltType = bie::Segment<0>;
      mesh_ = createMeshFromVect<EltType>(dimension_, nvertices_per_elt_, coor,
                                          conn);
      ker_obj_ = std::make_shared<
          bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
          elas, dimension_);
      break;
    }
    case "S3DP0"_sh: {
      dimension_ = 2;
      int nvertices_per_elt_ = 2;
      using EltType = bie::Segment<0>;
      mesh_ = createMeshFromVect<EltType>(dimension_, nvertices_per_elt_, coor,
                                          conn);
      ker_obj_ = std::make_shared<bie::BieElastostaticSp3d<
          EltType, EltType, bie::ElasticKernelType::H>>(elas, dimension_);

      il::Array<double> prop{1, properties[2]};
      ker_obj_->set_kernel_properties(prop);
      break;
    }
    case "2DP1"_sh: {
      dimension_ = 2;
      int nvertices_per_elt_ = 2;
      using EltType = bie::Segment<1>;
      mesh_ = createMeshFromVect<EltType>(dimension_, nvertices_per_elt_, coor,
                                          conn);
      ker_obj_ = std::make_shared<
          bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
          elas, dimension_);
      break;
    }
    case "3DT0"_sh: {
      dimension_ = 3;
      int nvertices_per_elt_ = 3;
      using EltType = bie::Triangle<0>;
      mesh_ = createMeshFromVect<EltType>(dimension_, nvertices_per_elt_, coor,
                                          conn);
      ker_obj_ = std::make_shared<
          bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
          elas, dimension_);
      break;
    }
    case "Axi3DP0"_sh: {
      dimension_ = 2;
      int nvertices_per_elt_ = 2;
      using EltType = bie::Segment<0>;
      mesh_ = createMeshFromVect<EltType>(dimension_, nvertices_per_elt_, coor,
                                          conn);
      ker_obj_ =
          std::make_shared<bie::ElasticAxiSymmRingKernel>(elas, dimension_);
      break;
    }
    default: {
      std::cout << "wrong inputs -abort \n";
      il::abort();
    }
    }
    tt.Start();
    auto hr = HRepresentationSquareMatrix(mesh_, max_leaf_size, eta);
    tt.Stop();
    collocation_points_ =
        mesh_->collocation_points(); // be careful returning it in original
                                     // ordering.  note this is only for
                                     // the output function
                                     // getCollocationPoints
                                     // ... could be deleted possibly
    h_representation_time_ = tt.time();
    tt.Reset();
    tt.Start();
    bie::BieMatrixGenerator<double> M(mesh_, mesh_, ker_obj_, hr);
    hmat_.toHmat(M, epsilon_aca_);
    tt.Stop();
    permutation_ = hr->permutation_0_;
    hmat_time_ = tt.time();
    if (hmat_.isBuilt()) {
      is_built_ = true;
      dof_dimension_ = hmat_.dofDimension();
      std::cout << "HMAT --> built \n";
      double test_cr = hmat_.compressionRatio();
      std::cout << "HMAT set"
                << ", CR = " << test_cr << ", eps_aca = " << epsilon_aca_
                << ", eta = " << eta_ << "\n";
      // std::cout << "H-mat construction time = :  " << hmat_time_ << "\n";
    } else {
      is_built_ = false;
    }

    std::cout << "BigWhamIO ENDED\n";
  }
  /* --------------------------------------------------------------------------
   */

  bool isBuilt() { return is_built_; };
  /* --------------------------------------------------------------------------
   */

  void hmatDestructor() {
    // this function will free the memory and set the hmat obj to its initial
    // status prior to initialization this will avoid ownership specifications
    // at binding time
    this->hmat_.hmatMemFree();
  };
  /* --------------------------------------------------------------------------
   */

  double getHmatTime() const { return hmat_time_; };
  /* --------------------------------------------------------------------------
   */

  double getPatternTime() const { return h_representation_time_; };
  /* --------------------------------------------------------------------------
   */

  std::vector<double> getCollocationPoints() {
    IL_EXPECT_FAST(is_built_);
    IL_EXPECT_FAST(collocation_points_.size(1) == dimension_);
    il::int_t npoints = collocation_points_.size(0);
    std::vector<double> flat_col;
    flat_col.assign(npoints * dimension_, 0.);
    int index = 0;
    for (il::int_t i = 0; i < collocation_points_.size(0); i++) {
      for (il::int_t j = 0; j < collocation_points_.size(1); j++) {
        flat_col[index] = collocation_points_(i, j);
        index++;
      }
    }
    return flat_col;
  };
  /* --------------------------------------------------------------------------
   */

  std::vector<long> getPermutation() {
    IL_EXPECT_FAST(is_built_);
    std::vector<long> permut;
    permut.assign(permutation_.size(), 0);
    for (il::int_t i = 0; i < permutation_.size(); i++) {
      permut[i] = permutation_[i];
    }
    return permut;
  }
  /* --------------------------------------------------------------------------
   */

  double getCompressionRatio() {
    IL_EXPECT_FAST(is_built_);
    return hmat_.compressionRatio();
  }
  /*
   */

  std::string getKernel() { return kernel_; }
  /* --------------------------------------------------------------------------
   */

  int getSpatialDimension() const { return dimension_; }
  /* --------------------------------------------------------------------------
   */

  int getProblemDimension() const { return dof_dimension_; }
  /* --------------------------------------------------------------------------
   */

  long matrixSize(int k) { return hmat_.size(k); };
  /* --------------------------------------------------------------------------
   */

  std::vector<long> getHpattern() {
    // API function to output the hmatrix pattern
    //  as flattened list via a pointer
    //  the numberofblocks is also returned (by reference)
    //
    //  the pattern matrix is formatted as
    // row = 1 block : i_begin,j_begin, i_end,j_end,FLAG,entry_size
    // with FLAG=0 for full rank and FLAG=1 for low rank
    //
    // we output a flatten row-major order std::vector

    IL_EXPECT_FAST(is_built_);

    bie::HPattern pattern = hmat_.pattern();

    long numberofblocks = pattern.n_B;
    long len = 6 * numberofblocks;
    std::cout << "number of blocks " << numberofblocks << "\n";

    std::vector<long> patternlist(len, 0);

    int index = 0;
    //  starts with full rank
    for (il::int_t j = 0; j < pattern.n_FRB; j++) {
      // check is low rank or not
      //  il::Array2DView<double> A = hmat_.asFullRank(s);
      patternlist[index++] = pattern.FRB_pattern(1, j);
      patternlist[index++] = pattern.FRB_pattern(2, j);
      patternlist[index++] = pattern.FRB_pattern(3, j);
      patternlist[index++] = pattern.FRB_pattern(4, j);
      patternlist[index++] = 0;
      patternlist[index++] =
          (pattern.FRB_pattern(4, j) - pattern.FRB_pattern(2, j)) *
          (pattern.FRB_pattern(3, j) -
           pattern.FRB_pattern(1, j)); // size of that sub blocks
    }
    // then low ranks
    for (il::int_t j = 0; j < pattern.n_LRB; j++) {

      patternlist[index++] = pattern.LRB_pattern(1, j);
      patternlist[index++] = pattern.LRB_pattern(2, j);
      patternlist[index++] = pattern.LRB_pattern(3, j);
      patternlist[index++] = pattern.LRB_pattern(4, j);
      patternlist[index++] = 1;
      patternlist[index++] = pattern.LRB_pattern(5, j); // the rank
    }
    // return a row major flatten vector
    return patternlist;
  }
  /* --------------------------------------------------------------------------
   */

  void getFullBlocks(std::vector<double> &val_list,
                     std::vector<int> &pos_list) {
    // return the full dense block entries of the hmat as
    // flattened lists
    // val_list(i) = H(pos_list(2*i),pos_list(2*i+1));
    // output in the original dof state (accounting for the permutation)

    IL_EXPECT_FAST(is_built_);

    il::Array<double> values{};
    il::Array<int> positions{};
    hmat_.fullBlocksOriginal(il::io, values, positions);
    //    std::cout << " checking values size" << values.size() <<  "\n";
    val_list.reserve(values.size());
    for (il::int_t i = 0; i < values.size(); i++) {
      val_list.push_back(values[i]);
    }
    pos_list.reserve(positions.size());
    for (il::int_t i = 0; i < positions.size(); i++) {
      pos_list.push_back(positions[i]);
    }
    std::cout << "number of entries " << val_list.size() << " - "
              << pos_list.size() << "\n";
    std::cout << " End of Bigwhamio getFullBlocks \n";
  }
  /* --------------------------------------------------------------------------
   */

  void getDiagonal(std::vector<double> &val_list) {
    // return the diagonal of the h-matrix
    // output in the original dof state (accounting for the permutation)

    IL_EXPECT_FAST(is_built_);
    val_list = hmat_.diagonalOriginal();

    std::cout << " End of Bigwhamio getDiagonal() \n";
  }
  /* --------------------------------------------------------------------------
   */

  std::vector<double> matvect(const std::vector<double> &x) {
    // in the original / natural ordering
    IL_EXPECT_FAST(this->is_built_);
    IL_EXPECT_FAST(hmat_.size(0) == hmat_.size(1));
    IL_EXPECT_FAST(hmat_.size(1) == x.size());
    std::vector<double> y = hmat_.matvecOriginal(x);
    return y;
  }
  /* --------------------------------------------------------------------------
   */

  std::vector<double> hdotProductInPermutted(const std::vector<double> &x) {
    // in the permutted state.
    IL_EXPECT_FAST(this->is_built_);
    IL_EXPECT_FAST(hmat_.size(0) == hmat_.size(1));
    IL_EXPECT_FAST(hmat_.size(1) == x.size());
    std::vector<double> y = hmat_.matvec(x);
    return y;
  }
  /* -------------------------------------------------------------------------
   */
  std::vector<double> ConvertToGlobal(const std::vector<double> &x_local) {
    // Input: x in original state (not permutted)
    // Output: in original state (not permutted)
    return mesh_->ConvertToGlobal(x_local.view());
  }
  /* -------------------------------------------------------------------------
   */
  std::vector<double> ConvertToLocal(const std::vector<double> &x_global) {
    // Input: x in original state (not permutted)
    // Output: in original state (not permutted)
    return mesh_->ConvertToLocal(x_global.view());
  }

}; // end class bigwhamio

#endif // BIGWHAM_BIGWHAM_H

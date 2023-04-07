//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 08.09.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications 5.2.21: Moving to std::unique_ptr (C. Peruzzo)
#if defined(__clang__) && !defined(FMT_ICC_VERSION)
#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#endif

#ifndef BIGWHAM_HMAT_H
#define BIGWHAM_HMAT_H

#if defined(IL_OPENMP)
#include <omp.h>
#endif

#include <cmath>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#if defined(BIGWHAM_HDF5)
#include "hdf5.h"
#endif

#if defined(HAS_ITT)
#include <ittnotify.h>
#endif

#include "hmat/hierarchical_representation.h"
#include "hmat/hmatrix/LowRank.h"
#include "hmat/hmatrix/h_pattern.h"

#include "hmat/arrayFunctor/matrix_generator.h"
#include "hmat/compression/adaptiveCrossApproximation.h"

namespace bie {

template <typename T> class Hmat {
  // this is a new Hmat class wich does not contains the matrix-generator
  // (neither the mesh etc.) construction from the pattern built from the block
  // cluster tree openmp parallel construction openmp parallel mat_vect dot
  // product (non-permutted way)
private:
  std::shared_ptr<HRepresentation> hr_;

  // shall we store the permutation(s) in that object ?

  il::int_t dof_dimension_;            //  dof per collocation points
  il::StaticArray<il::int_t, 2> size_; // size of tot mat (row, cols)

  std::vector<std::unique_ptr<bie::LowRank<T>>>
      low_rank_blocks_; // vector of low rank blocks
  std::vector<std::unique_ptr<il::Array2D<T>>>
      full_rank_blocks_; // vector of full rank blocks

#if defined(IL_OPENMP)
  int frb_chunk_size_{1};
  int lrb_chunk_size_{1};
#endif

  bool isBuilt_{false};
  bool isBuilt_LR_{false};
  bool isBuilt_FR_{false};
  bool is_square_{true};

#if defined(HAS_ITT)
  static __itt_domain *itt_domain;
  static __itt_string_handle *itt_task_build;
  static __itt_string_handle *itt_task_buildfrb;
  static __itt_string_handle *itt_task_buildlrb;
  static __itt_string_handle *itt_task_matvec;
  static __itt_string_handle *itt_task_matvec_frb;
  static __itt_string_handle *itt_task_matvec_lrb;
#endif

public:
  Hmat() = default;
  ~Hmat() = default;
  /* --------------------------------------------------------------------------
   */

  // delete the memory pointed by low_rank_blocks_ and  full_rank_blocks_
  void hmatMemFree() {
    for (il::int_t i = 0; i < hr_->pattern_.n_FRB; i++) {
      this->full_rank_blocks_[i].reset();
    }
    for (il::int_t i = 0; i < hr_->pattern_.n_LRB; i++) {
      this->low_rank_blocks_[i].reset();
    }
  };
  /* --------------------------------------------------------------------------
   */

  Hmat(const bie::MatrixGenerator<T> &matrix_gen, double epsilon_aca) {
    this->toHmat(matrix_gen, epsilon_aca);
  }
  /* --------------------------------------------------------------------------
   */

  Hmat(const std::string &filename) {
    il::Timer tt;
    tt.Start();
    this->readFromFile(filename);
    tt.Stop();
    std::cout << "Reading of hmat done in " << tt.time() << "\n";
    std::cout << "Compression ratio - " << this->compressionRatio() << "\n";
    std::cout << "Hmat object - built "
              << "\n";
  }
  /* --------------------------------------------------------------------------
   */

  void toHmat(const bie::MatrixGenerator<T> &matrix_gen,
              const double epsilon_aca) {
    this->hr_ = matrix_gen.hr();
    this->is_square_ = hr_->is_square_;
    il::Timer tt;
    tt.Start();
    this->build(matrix_gen, epsilon_aca);
    tt.Stop();
    std::cout << "Creation of hmat done in " << tt.time() << "\n";
    std::cout << "Compression ratio - " << this->compressionRatio() << "\n";
    std::cout << "Hmat object - built "
              << "\n";
  }
  /* --------------------------------------------------------------------------
   */

  void writeToFile(const std::string &filename) {
#if defined(BIGWHAM_HDF5)
    auto file_id =
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    auto hmat_gid =
        H5Gcreate(file_id, "/hmat", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    auto writeAttribute = [](auto &&name, auto &&gid, auto &&value) {
      auto aid = H5Screate(H5S_SCALAR);
      auto attr =
          H5Acreate(gid, name, H5T_NATIVE_LLONG, aid, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr, H5T_NATIVE_LLONG, &value);
      H5Aclose(attr);
      H5Sclose(aid);
    };

    auto writeArray = [](auto &&name, auto &&gid, auto &&array) {
      hsize_t dims[1]{hsize_t(array.size())};
      auto dataspace_id = H5Screate_simple(1, dims, NULL);
      auto dataset_id =
          H5Dcreate(gid, std::string(name).c_str(), H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id,
               H5P_DEFAULT, array.data());
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
    };

    auto writeArray2D = [](auto &&name, auto &&gid, auto &&array) {
      hsize_t dims[2]{
          hsize_t(array.size(0)),
          hsize_t(array.size(
              1))}; // explicit conversion to avoid the compiler warning
      auto dataspace_id = H5Screate_simple(2, dims, NULL);
      auto dataset_id =
          H5Dcreate(gid, std::string(name).c_str(), H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id,
               H5P_DEFAULT, array.data());
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
    };

    writeAttribute("dof_dimension", hmat_gid, dof_dimension_);
    writeAttribute("m", hmat_gid, size_[0]);
    writeAttribute("n", hmat_gid, size_[1]);

    auto permutation_gid = H5Gcreate(hmat_gid, "permutation", H5P_DEFAULT,
                                     H5P_DEFAULT, H5P_DEFAULT);
    writeArray("perm_row", permutation_gid, this->hr_->permutation_0_);
    writeArray("perm_col", permutation_gid, this->hr_->permutation_1_);
    H5Gclose(permutation_gid);

    auto pattern_gid =
        H5Gcreate(hmat_gid, "pattern", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    writeAttribute("n_FRB", pattern_gid, hr_->pattern_.n_FRB);
    writeAttribute("n_LRB", pattern_gid, hr_->pattern_.n_LRB);
    writeArray2D("FRB", pattern_gid, hr_->pattern_.FRB_pattern);
    writeArray2D("LRB", pattern_gid, hr_->pattern_.LRB_pattern);
    H5Gclose(pattern_gid);

    auto frb_gid =
        H5Gcreate(hmat_gid, "FRB", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    il::int_t i_frb = 0;
    for (auto &&frb : full_rank_blocks_) {
      writeArray2D("frb_" + std::to_string(i_frb), frb_gid, *frb);
      ++i_frb;
    }
    H5Gclose(frb_gid);

    auto lrbs_gid =
        H5Gcreate(hmat_gid, "LRB", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    il::int_t i_lrb = 0;
    for (auto &&lrb : low_rank_blocks_) {
      std::string group = "lrb_" + std::to_string(i_lrb);
      auto lrb_gid = H5Gcreate(lrbs_gid, group.c_str(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
      writeArray2D("A", lrb_gid, lrb->A);
      writeArray2D("B", lrb_gid, lrb->B);
      H5Gclose(lrb_gid);
      ++i_lrb;
    }
    H5Gclose(lrbs_gid);
    H5Gclose(hmat_gid);
    H5Fclose(file_id);
#endif
  }
  /* --------------------------------------------------------------------------
   */

  void readFromFile(const std::string &filename) {
#if defined(BIGWHAM_HDF5)
    auto file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    auto hmat_gid = H5Gopen(file_id, "/hmat", H5P_DEFAULT);

    auto readAttribute = [](auto &&name, auto &&gid, auto &&value) {
      auto attr = H5Aopen(gid, name, H5P_DEFAULT);
      H5Aread(attr, H5T_NATIVE_LLONG, &value);
      H5Aclose(attr);
    };

    auto readArray = [](auto &&name, auto &&gid, auto &&array) {
      hsize_t dims[1];
      auto dataset_id = H5Dopen(gid, std::string(name).c_str(), H5P_DEFAULT);

      auto dataspace_id = H5Dget_space(dataset_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      array.Resize(dims[0]);
      H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              array.Data());
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    };

    auto readArray2D = [](auto &&name, auto &&gid, auto &&array) {
      hsize_t dims[2];
      auto dataset_id = H5Dopen(gid, std::string(name).c_str(), H5P_DEFAULT);

      auto dataspace_id = H5Dget_space(dataset_id);
      H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

      array.Resize(dims[0], dims[1]);
      H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              array.Data());
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    };

    readAttribute("dof_dimension", hmat_gid, dof_dimension_);
    readAttribute("m", hmat_gid, size_[0]);
    readAttribute("n", hmat_gid, size_[1]);

    this->hr_ = std::make_shared<HRepresentation>();

    auto permutation_gid = H5Gopen(hmat_gid, "permutation", H5P_DEFAULT);
    readArray("perm_row", permutation_gid, hr_->permutation_0_);
    readArray("perm_col", permutation_gid, hr_->permutation_1_);
    H5Gclose(permutation_gid);

    auto pattern_gid = H5Gopen(hmat_gid, "pattern", H5P_DEFAULT);
    readAttribute("n_FRB", pattern_gid, hr_->pattern_.n_FRB);
    readAttribute("n_LRB", pattern_gid, hr_->pattern_.n_LRB);

    readArray2D("FRB", pattern_gid, hr_->pattern_.FRB_pattern);
    readArray2D("LRB", pattern_gid, hr_->pattern_.LRB_pattern);
    H5Gclose(pattern_gid);

    std::cout << "Reading HMat from \"" << filename << "\" - " << size_[0]
              << "x" << size_[1] << " - " << dof_dimension_ << std::endl;
    std::cout << " Number of blocks = "
              << hr_->pattern_.n_FRB + hr_->pattern_.n_LRB << std::endl;
    std::cout << " Number of full blocks = " << hr_->pattern_.n_FRB
              << std::endl;

    auto frbs_gid = H5Gopen(hmat_gid, "FRB", H5P_DEFAULT);
    full_rank_blocks_.resize(hr_->pattern_.n_FRB);
    il::int_t i_frb = 0;
    il::int_t i_frb_10_per = hr_->pattern_.n_FRB * 0.1;
    for (auto &&frb : full_rank_blocks_) {
      if (i_frb % i_frb_10_per == 0)
        std::cout << "." << std::flush;
      frb = std::make_unique<il::Array2D<T>>();
      readArray2D("frb_" + std::to_string(i_frb), frbs_gid, *frb);
      ++i_frb;
    }
    std::cout << std::endl;
    H5Gclose(frbs_gid);

    std::cout << " Number of low rank blocks = " << hr_->pattern_.n_LRB
              << std::endl;
    auto lrbs_gid = H5Gopen(hmat_gid, "LRB", H5P_DEFAULT);
    low_rank_blocks_.resize(hr_->pattern_.n_LRB);
    il::int_t i_lrb = 0;
    il::int_t i_lrb_10_per = hr_->pattern_.n_LRB * 0.1;
    for (auto &&lrb : low_rank_blocks_) {
      if (i_lrb % i_lrb_10_per == 0)
        std::cout << "." << std::flush;
      lrb = std::make_unique<LowRank<T>>();
      std::string group = "lrb_" + std::to_string(i_lrb);
      auto lrb_gid = H5Gopen(lrbs_gid, group.c_str(), H5P_DEFAULT);
      readArray2D("A", lrb_gid, lrb->A);
      readArray2D("B", lrb_gid, lrb->B);
      H5Gclose(lrb_gid);
      ++i_lrb;
    }
    std::cout << std::endl;
    H5Gclose(lrbs_gid);

    H5Gclose(hmat_gid);
    H5Fclose(file_id);

    isBuilt_FR_ = isBuilt_LR_ = isBuilt_ = true;
#endif
  }
  /* --------------------------------------------------------------------------
   */

  void buildFR(const bie::MatrixGenerator<T> &matrix_gen) {
    // construction of the full rank blocks
    std::cout << " Loop on full blocks construction  \n";
    std::cout << " N full blocks " << hr_->pattern_.n_FRB << " \n";

    full_rank_blocks_.resize(hr_->pattern_.n_FRB);
#pragma omp parallel for schedule(static, frb_chunk_size_)
    for (il::int_t i = 0; i < hr_->pattern_.n_FRB; i++) {
      il::int_t i0 = hr_->pattern_.FRB_pattern(1, i);
      il::int_t j0 = hr_->pattern_.FRB_pattern(2, i);
      il::int_t iend = hr_->pattern_.FRB_pattern(3, i);
      il::int_t jend = hr_->pattern_.FRB_pattern(4, i);

      const il::int_t ni = matrix_gen.blockSize() * (iend - i0);
      const il::int_t nj = matrix_gen.blockSize() * (jend - j0);

      std::unique_ptr<il::Array2D<T>> a =
          std::make_unique<il::Array2D<T>>(ni, nj);
      matrix_gen.set(i0, j0, il::io, a->Edit());
      full_rank_blocks_[i] = std::move(a);
    }
    isBuilt_FR_ = true;
  }
  /* --------------------------------------------------------------------------
   */

  void buildLR(const bie::MatrixGenerator<T> &matrix_gen,
               const double epsilon) {
    // constructing the low rank blocks
    dof_dimension_ = matrix_gen.blockSize();
    std::cout << " Loop on low rank blocks construction  \n";
    std::cout << " N low rank blocks " << hr_->pattern_.n_LRB << " \n";
    std::cout << "dof_dimension: " << dof_dimension_ << "\n";

#if defined(_OPENMP)
    auto nthreads = omp_get_max_threads();
#endif
    low_rank_blocks_.resize(hr_->pattern_.n_LRB);

    auto buildLRByDim = [&](auto &&intergral_dim) {
      constexpr il::int_t dim = std::decay_t<decltype(intergral_dim)>::value;
#pragma omp parallel for schedule(static, lrb_chunk_size_)
      for (il::int_t i = 0; i < hr_->pattern_.n_LRB; i++) {
        il::int_t i0 = hr_->pattern_.LRB_pattern(1, i);
        il::int_t j0 = hr_->pattern_.LRB_pattern(2, i);
        il::int_t iend = hr_->pattern_.LRB_pattern(3, i);
        il::int_t jend = hr_->pattern_.LRB_pattern(4, i);
        il::Range range0{i0, iend};
        il::Range range1{j0, jend};

        // we need 7a LRA generator virtual template similar to the Matrix
        // generator... here we have an if condition for the LRA call dependent
        // on dof_dimension_
        auto lra = bie::adaptiveCrossApproximation<dim>(matrix_gen, range0,
                                                        range1, epsilon);

        // store the rank in the low_rank pattern
        hr_->pattern_.LRB_pattern(5, i) = lra->A.size(1);
        low_rank_blocks_[i] =
            std::move(lra); // lra_p does not exist after such call
      }
    };

    switch (matrix_gen.blockSize()) {
    case 1:
      buildLRByDim(std::integral_constant<int, 1>{});
      break;
    case 2:
      buildLRByDim(std::integral_constant<int, 2>{});
      break;
    case 3:
      buildLRByDim(std::integral_constant<int, 3>{});
      break;
    }
    isBuilt_LR_ = true;
  }
  //-----------------------------------------------------------------------------
  // filling up the h-matrix sub-blocks
  void build(const bie::MatrixGenerator<T> &matrix_gen, const double epsilon) {
    // #if defined(HAS_ITT)
    //     __itt_task_begin(itt_domain, __itt_null, __itt_null, itt_task_build);
    // #endif

    dof_dimension_ = matrix_gen.blockSize();
    size_[0] = matrix_gen.size(0);
    size_[1] = matrix_gen.size(1);
#if defined(HAS_ITT)
    __itt_task_begin(itt_domain, __itt_null, __itt_null, itt_task_buildfrb);
#endif
    buildFR(matrix_gen);
#if defined(HAS_ITT)
    __itt_task_end(itt_domain);
    __itt_task_begin(itt_domain, __itt_null, __itt_null, itt_task_buildlrb);
#endif
    buildLR(matrix_gen, epsilon);
#if defined(HAS_ITT)
    __itt_task_end(itt_domain);
#endif

    isBuilt_ = isBuilt_FR_ && isBuilt_LR_;
    // #if defined(HAS_ITT)
    //     __itt_task_end(itt_domain);
    // #endif
  }
  //-----------------------------------------------------------------------------
  bool isBuilt() const { return isBuilt_; };
  //
  il::int_t size(int k) const { return size_[k]; };
  //
  bie::HPattern pattern() {
    return this->hr_->pattern_;
  }; // returning the Hmat pattern

  il::int_t dofDimension() const { return dof_dimension_; };

  //-----------------------------------------------------------------------------
  // getting the nb of entries of the hmatrix
  il::int_t nbOfEntries() {
    IL_EXPECT_FAST(isBuilt_);
    il::int_t n = 0;
    for (il::int_t i = 0; i < hr_->pattern_.n_FRB; i++) {
      auto &a = *full_rank_blocks_[i];
      n += a.size(0) * a.size(1);
    }
    for (il::int_t i = 0; i < hr_->pattern_.n_LRB; i++) {
      il::Array2DView<double> a = (*low_rank_blocks_[i]).A.view();
      il::Array2DView<double> b = (*low_rank_blocks_[i]).B.view();
      n += a.size(0) * a.size(1) + b.size(0) * b.size(1);
    }
    return n;
  }
  //-----------------------------------------------------------------------------
  // getting the compression ratio of the hmatrix (double which is <=1)
  double compressionRatio() {
    auto nb_elts = static_cast<double>(nbOfEntries());
    return nb_elts / static_cast<double>(size_[0] * size_[1]);
  }

  //--------------------------------------------------------------------------
  // H-Matrix vector multiplication without permutation
  // in - il:Array<T>
  // out - il:Array<T>
  il::Array<T> matvec(const il::Array<T> &x) {
#if defined(HAS_ITT)
    __itt_frame_begin_v3(itt_domain, nullptr);
#endif

    IL_EXPECT_FAST(x.size() == size_[1]);
    il::Array<T> y{size_[0], 0.};

#if defined(_OPENMP)
    auto nthreads = omp_get_max_threads();
    il::Array2D<T> yprivate_storage{size_[0], nthreads};
#endif

#if defined(_OPENMP)
    static bool first_time = true;
#endif

#pragma omp parallel shared(first_time)
    {
#if defined(_OPENMP)
      auto thread_num = omp_get_thread_num();

      il::ArrayEdit<T> yprivate =
          yprivate_storage.Edit(il::Range{0, size_[0]}, thread_num);
      for (il::int_t i = 0; i < size_[0]; ++i) {
        yprivate[i] = 0.;
      }
#else
      il::ArrayEdit<T> yprivate = y.Edit();
#endif
#if defined(HAS_ITT)
      __itt_task_begin(itt_domain, __itt_null, __itt_null, itt_task_matvec_frb);
#endif

#pragma omp for nowait schedule(static, frb_chunk_size_)
      for (il::int_t i = 0; i < hr_->pattern_.n_FRB; i++) {
        il::int_t i0 = hr_->pattern_.FRB_pattern(1, i);
        il::int_t j0 = hr_->pattern_.FRB_pattern(2, i);
        il::int_t iend = hr_->pattern_.FRB_pattern(3, i);
        il::int_t jend = hr_->pattern_.FRB_pattern(4, i);

        il::Array2DView<T> a = (*full_rank_blocks_[i]).view();
        il::ArrayView<T> xs =
            x.view(il::Range{j0 * dof_dimension_, jend * dof_dimension_});
        il::ArrayEdit<T> ys = yprivate.Edit(
            il::Range{i0 * dof_dimension_, iend * dof_dimension_});

        il::blas(1.0, a, xs, 1.0, il::io, ys);
      }

#if defined(HAS_ITT)
      __itt_task_end(itt_domain);
      __itt_task_begin(itt_domain, __itt_null, __itt_null, itt_task_matvec_lrb);
#endif

#pragma omp for nowait schedule(static, lrb_chunk_size_)
      for (il::int_t ii = 0; ii < hr_->pattern_.n_LRB; ii++) {
        auto i0 = hr_->pattern_.LRB_pattern(1, ii);
        auto j0 = hr_->pattern_.LRB_pattern(2, ii);
        auto iend = hr_->pattern_.LRB_pattern(3, ii);
        auto jend = hr_->pattern_.LRB_pattern(4, ii);

        auto a = low_rank_blocks_[ii]->A.view();
        auto b = low_rank_blocks_[ii]->B.view();

        auto xs = x.view(il::Range{j0 * dof_dimension_, jend * dof_dimension_});
        auto ys = yprivate.Edit(
            il::Range{i0 * dof_dimension_, iend * dof_dimension_});
        const il::int_t r = a.size(1);
        il::Array<double> tmp{r, 0.0};

        il::blas(1.0, b, il::Dot::None, xs, 0.0, il::io,
                 tmp.Edit()); // Note here we have stored b (not b^T)
        il::blas(1.0, a, tmp.view(), 1.0, il::io, ys);
      }

#if defined(HAS_ITT)
      __itt_task_end(itt_domain);
#endif
#if defined(_OPENMP)
      il::int_t j = 0;
#pragma omp for schedule(static)
      for (il::int_t j = 0; j < y.size(); j++) {
        auto yview =
            yprivate_storage.view(il::Range{j, j + 1}, il::Range{0, nthreads});
        for (il::int_t i = 0; i < nthreads; ++i) {
          y[j] += yview(0, i);
        }
      }
#endif
    }

#if defined(HAS_ITT)
    __itt_frame_end_v3(itt_domain, nullptr);
#endif
    return y;
  }
  /* --------------------------------------------------------------------------
   */

  // H-Matrix vector multiplication with permutation for rectangular matrix
  // cases (only 1 permutation) in & out as std::vector todo : write another one
  // for the case of 2 permutations (rect. mat cases (for source != receivers)
  std::vector<T> matvecOriginal(const std::vector<T> &x) {

    il::Array<T> z{static_cast<il::int_t>(x.size())};
    // permutation of the dofs according to the re-ordering sue to clustering
    il::int_t ncolpoints = this->size(1) / dof_dimension_;
    il::int_t nrowpoints = this->size(0) / dof_dimension_;
    for (il::int_t i = 0; i < ncolpoints; i++) {
      for (int j = 0; j < dof_dimension_; j++) {
        z[dof_dimension_ * i + j] = x[dof_dimension_ * hr_->permutation_1_[i] + j];
      }
    }
    il::Array<T> y = this->matvec(z);
    std::vector<T> yout;
    yout.assign(y.size(), 0.);
    // permut back
    for (il::int_t i = 0; i < nrowpoints; i++) {
      for (int j = 0; j < dof_dimension_; j++) {
        yout[dof_dimension_ * hr_->permutation_0_[i] + j] = y[dof_dimension_ * i + j];
      }
    }
    return yout;
  }
  /* --------------------------------------------------------------------------
   */

  // matvect in and outs as std::vector
  std::vector<T> matvec(const std::vector<T> &x) {
    il::Array<T> xil{static_cast<il::int_t>(x.size())};
    // todo find a better way to convert il::Array to std::vect and vice versa !
    for (long i = 0; i < xil.size(); i++) {
      xil[i] = x[i];
    }
    il::Array<T> yil = this->matvec(xil);
    std::vector<T> y;
    y.reserve(static_cast<long>(yil.size()));
    for (long i = 0; i < yil.size(); i++) {
      y.push_back(yil[i]);
    }
    return y;
  }
  /* --------------------------------------------------------------------------
   */

  std::vector<T> diagonal() {
    // return diagonal in permutted state....
    IL_EXPECT_FAST(isBuilt_FR_);
    il::int_t diag_size = il::max(size_[0], size_[1]);
    std::vector<T> diag(static_cast<long>(diag_size), 0.);

    for (il::int_t k = 0; k < hr_->pattern_.n_FRB; k++) {
      il::Array2D<double> aux = *full_rank_blocks_[k];
      il::int_t i0 = hr_->pattern_.FRB_pattern(1, k);
      il::int_t j0 = hr_->pattern_.FRB_pattern(2, k);
      il::int_t iend = hr_->pattern_.FRB_pattern(3, k);
      il::int_t jend = hr_->pattern_.FRB_pattern(4, k);
      // check if it intersects the diagonal
      //
      bool in_lower = (i0 > j0) && (i0 > jend) && (iend > j0) && (iend > jend);
      bool in_upper = (i0 < j0) && (i0 < jend) && (iend < j0) && (iend < jend);
      if ((!in_lower) && (!in_upper)) // this fb intersect the diagonal....
      {
        for (il::int_t j = 0; j < aux.size(1); j++) {
          for (il::int_t i = 0; i < aux.size(0); i++) {
            if ((i + dof_dimension_ * i0) ==
                (j + dof_dimension_ * j0)) { // on diagonal !
              diag[(i + dof_dimension_ * i0)] = aux(i, j);
            }
          }
        }
      }
    }
    return diag;
  }
  /* --------------------------------------------------------------------------
   */

  std::vector<T> diagonalOriginal() {
    // return diagonal in original state....
    il::int_t diag_size = il::max(size_[0], size_[1]);
    il::int_t ncolpoints = diag_size / dof_dimension_;
    std::vector<T> diag_raw = this->diagonal();
    std::vector<T> diag(diag_raw.size(), 0.);
    // permut back
    for (il::int_t i = 0; i < ncolpoints; i++) {
      for (int j = 0; j < dof_dimension_; j++) {
        diag[dof_dimension_ * hr_->permutation_1_[i] + j] =
            diag_raw[dof_dimension_ * i + j];
      }
    }
    return diag;
  }
  /* --------------------------------------------------------------------------
   */

  void fullBlocksOriginal(il::io_t, il::Array<T> &val_list,
                          il::Array<int> &pos_list) {
    // return the full blocks in the permutted Original dof state
    // in the val_list and pos_list 1D arrays.
    IL_EXPECT_FAST(isBuilt_FR_);
    IL_EXPECT_FAST(hr_->permutation_1_.size() * dof_dimension_ == size_[1]);
    //  compute the number of  entries in the whole full rank blocks
    int nbfentry = 0;
    for (il::int_t i = 0; i < hr_->pattern_.n_FRB; i++) {
      il::Array2DView<double> aux = ((*full_rank_blocks_[i]).view());
      nbfentry = nbfentry + static_cast<int>(aux.size(0) * aux.size(1));
    }

    // prepare outputs
    pos_list.Resize(nbfentry * 2);
    val_list.Resize(nbfentry);

    il::Array<int> permutDOF{dof_dimension_ * hr_->permutation_1_.size(), 0};
    IL_EXPECT_FAST(permutDOF.size() == size_[0]);
    for (il::int_t i = 0; i < hr_->permutation_1_.size(); i++) {
      for (il::int_t j = 0; j < dof_dimension_; j++) {
        permutDOF[i * dof_dimension_ + j] =
            hr_->permutation_1_[i] * dof_dimension_ + j;
      }
    }
    // loop on full rank and get i,j and val
    int nr = 0;
    int npos = 0;
    for (il::int_t k = 0; k < hr_->pattern_.n_FRB; k++) {
      il::Array2D<double> aux = *full_rank_blocks_[k];
      il::int_t i0 = hr_->pattern_.FRB_pattern(1, k);
      il::int_t j0 = hr_->pattern_.FRB_pattern(2, k);
      il::int_t index = 0;
      for (il::int_t j = 0; j < aux.size(1); j++) {
        for (il::int_t i = 0; i < aux.size(0); i++) {
          pos_list[npos + 2 * index] = permutDOF[(i + dof_dimension_ * i0)];
          pos_list[npos + 2 * index + 1] = permutDOF[(j + dof_dimension_ * j0)];
          val_list[nr + index] = aux(i, j);
          index++;
        }
      }
      nr = nr + static_cast<int>(aux.size(0) * aux.size(1));
      npos = npos + static_cast<int>(2 * aux.size(0) * aux.size(1));
    }
    std::cout << "done Full Block: nval " << val_list.size() << " / "
              << pos_list.size() / 2 << " n^2 "
              << (this->size_[0]) * (this->size_[1]) << "\n";
  }
};

#if defined(HAS_ITT)
template <class T>
__itt_domain *Hmat<T>::itt_domain = __itt_domain_create("HMat");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_build = __itt_string_handle_create("Build.Build");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_buildfrb = __itt_string_handle_create("Build.BuildFRB");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_buildlrb = __itt_string_handle_create("Build.BuildLRB");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_matvec = __itt_string_handle_create("MatVec");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_matvec_frb = __itt_string_handle_create("MatVec.FRB");
template <class T>
__itt_string_handle *
    Hmat<T>::itt_task_matvec_lrb = __itt_string_handle_create("MatVec.LRB");
#endif

} // namespace bie
#endif // BIGWHAM_HMAT_H

#if defined(__clang__) && !defined(FMT_ICC_VERSION)
#pragma clang diagnostic pop
#endif

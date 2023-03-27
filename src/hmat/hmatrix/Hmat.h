//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 08.09.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications 5.2.21: Moving to std::unique_ptr (C. Peruzzo)
#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#ifndef BIGWHAM_HMAT_H
#define BIGWHAM_HMAT_H

#pragma once
#if defined(IL_OPENMP)
#  include <omp.h>
#endif

#include "core/hierarchical_representation.h"
#include "hmat/hmatrix/LowRank.h"
#include "hmat/hmatrix/toHPattern.h"

#include "hmat/compression/adaptiveCrossApproximation.h"
#include "hmat/arrayFunctor/MatrixGenerator.h"

#include <omp.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <string>

#if defined(BIGWHAM_HDF5)
#include "hdf5.h"
#endif

// #if defined(HAS_ITT)
// #include <ittnotify.h>
// #endif

#include <vector>

namespace bie {

template <typename T>
class Hmat {
// this is a new Hmat class wich does not contains the matrix-generator (neither the mesh etc.)
// construction from the pattern built from the block cluster tree
// openmp parallel construction
// openmp parallel mat_vect dot product (non-permutted way)
 private:

  bie::HPattern pattern_;

  // shall we store the permutation(s) in that object ?

  il::int_t dof_dimension_; //  dof per collocation points
  il::StaticArray<il::int_t,2> size_; // size of tot mat (row, cols)

  std::vector<std::unique_ptr<bie::LowRank<T>>> low_rank_blocks_; // vector of low rank blocks
  std::vector<std::unique_ptr<il::Array2D<T>>>  full_rank_blocks_; // vector of full rank blocks

  bool isBuilt_= false;
  bool isBuilt_LR_=false;
  bool isBuilt_FR_=false;

  bool is_square_=true;

  public:

   Hmat()= default;
   ~Hmat()= default;

   // delete the memory pointed by low_rank_blocks_ and  full_rank_blocks_
   void hmatMemFree(){
       for (il::int_t i = 0; i < pattern_.n_FRB; i++) {
           this->full_rank_blocks_[i].reset();
       }
       for (il::int_t i = 0; i < pattern_.n_LRB; i++) {
           this->low_rank_blocks_[i].reset();
       }
   };

//   // simple constructor from pattern  -> todo remove
//   Hmat(const bie::HPattern& pattern){
//     pattern_=pattern;
//   };

   // direct constructor
   Hmat(const bie::MatrixGenerator<T>& matrix_gen,const bie::HRepresentation& h_r, double epsilon_aca){
       pattern_=h_r.pattern_;
       is_square_=h_r.is_square_;
       // construction directly
       il::Timer tt;
       tt.Start();
       this->build(matrix_gen,epsilon_aca);
       tt.Stop();
       std::cout << "Creation of hmat done in "  << tt.time() <<"\n";std::cout << "Compression ratio - " << this->compressionRatio() <<"\n";
       std::cout << "Hmat object - built " << "\n";
   }

  Hmat(const std::string & filename){
       // construction directly
       il::Timer tt;
       tt.Start();
       this->readFromFile(filename);
       tt.Stop();
       std::cout << "Reading of hmat done in "  << tt.time() <<"\n";std::cout << "Compression ratio - " << this->compressionRatio() <<"\n";
       std::cout << "Hmat object - built " << "\n";
   }


    // Main constructor
    void toHmat(const bie::MatrixGenerator<T>& matrix_gen,const bie::HRepresentation& h_r,const double epsilon_aca){
        pattern_=h_r.pattern_;
        is_square_=h_r.is_square_;
        // construction directly
        il::Timer tt;
        tt.Start();
        this->build(matrix_gen,epsilon_aca);
        tt.Stop();
        std::cout << "Creation of hmat done in "  << tt.time() <<"\n";std::cout << "Compression ratio - " << this->compressionRatio() <<"\n";
        std::cout << "Hmat object - built " << "\n";
    }

  void writeToFile(const std::string & filename) {
#if defined(BIGWHAM_HDF5)
    auto file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    auto hmat_gid = H5Gcreate(file_id, "/hmat", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    auto writeAttribute = [](auto && name, auto && gid, auto && value) {
      auto aid  = H5Screate(H5S_SCALAR);
      auto attr = H5Acreate(gid, name, H5T_NATIVE_LLONG, aid, H5P_DEFAULT, H5P_DEFAULT);
      H5Awrite(attr, H5T_NATIVE_LLONG, &value);
      H5Aclose(attr);
      H5Sclose(aid);
    };

    auto writeArray2D = [](auto && name, auto && gid, auto && array) {
      hsize_t dims[2] = {array.size(0), array.size(1)};
      auto dataspace_id = H5Screate_simple(2, dims, NULL);
      auto dataset_id = H5Dcreate(gid, std::string(name).c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, H5P_DEFAULT, array.data());
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
    };

    writeAttribute("dof_dimension", hmat_gid, dof_dimension_);
    writeAttribute("m", hmat_gid, size_[0]);
    writeAttribute("n", hmat_gid, size_[1]);

    auto pattern_gid = H5Gcreate(hmat_gid, "pattern", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    writeAttribute("n_FRB", pattern_gid, pattern_.n_FRB);
    writeAttribute("n_LRB", pattern_gid, pattern_.n_LRB);
    writeArray2D("FRB", pattern_gid, pattern_.FRB_pattern);
    writeArray2D("LRB", pattern_gid, pattern_.LRB_pattern);
    H5Gclose(pattern_gid);

    auto frb_gid = H5Gcreate(hmat_gid, "FRB", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    il::int_t i_frb = 0;
    for(auto && frb : full_rank_blocks_) {
      writeArray2D("frb_" + std::to_string(i_frb), frb_gid, *frb);
      ++i_frb;
    }
    H5Gclose(frb_gid);

    auto lrbs_gid = H5Gcreate(hmat_gid, "LRB", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    il::int_t i_lrb = 0;
    for(auto && lrb : low_rank_blocks_) {
      std::string group = "lrb_" + std::to_string(i_lrb);
      auto lrb_gid = H5Gcreate(lrbs_gid, group.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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

  void readFromFile(const std::string & filename) {
#if defined(BIGWHAM_HDF5)
    auto file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    auto hmat_gid = H5Gopen(file_id, "/hmat", H5P_DEFAULT);

    auto readAttribute = [](auto && name, auto && gid, auto && value) {
      auto attr = H5Aopen(gid, name, H5P_DEFAULT);
      H5Aread(attr, H5T_NATIVE_LLONG, &value);
      H5Aclose(attr);
    };

    auto readArray2D = [](auto && name, auto && gid, auto && array) {
      hsize_t dims[2];
      auto dataset_id = H5Dopen(gid, std::string(name).c_str(), H5P_DEFAULT);

      auto dataspace_id = H5Dget_space (dataset_id);
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

    auto pattern_gid = H5Gopen(hmat_gid, "pattern", H5P_DEFAULT);
    readAttribute("n_FRB", pattern_gid, pattern_.n_FRB);
    readAttribute("n_LRB", pattern_gid, pattern_.n_LRB);

    readArray2D("FRB", pattern_gid, pattern_.FRB_pattern);
    readArray2D("LRB", pattern_gid, pattern_.LRB_pattern);
    H5Gclose(pattern_gid);

    std::cout << "Reading HMat from \"" << filename << "\" - " << size_[0] << "x" << size_[1] << " - " << dof_dimension_  << std::endl;

    std::cout << " Number of blocks = " << pattern_.n_FRB + pattern_.n_LRB  << std::endl;

    std::cout << " Number of full blocks = " << pattern_.n_FRB  << std::endl;
    auto frbs_gid = H5Gopen(hmat_gid, "FRB", H5P_DEFAULT);
    full_rank_blocks_.resize(pattern_.n_FRB);
    il::int_t i_frb = 0;
    for(auto && frb : full_rank_blocks_) {
      frb = std::make_unique<il::Array2D<T>>();
      readArray2D("frb_" + std::to_string(i_frb), frbs_gid, *frb);
      ++i_frb;
    }
    H5Gclose(frbs_gid);

    std::cout << " Number of low rank blocks = " << pattern_.n_LRB  << std::endl;
    auto lrbs_gid = H5Gopen(hmat_gid, "LRB", H5P_DEFAULT);
    low_rank_blocks_.resize(pattern_.n_LRB);
    il::int_t i_lrb = 0;
    for(auto && lrb : low_rank_blocks_) {
      lrb = std::make_unique<LowRank<T>>();
      std::string group = "lrb_" + std::to_string(i_lrb);
      auto lrb_gid = H5Gopen(lrbs_gid, group.c_str(), H5P_DEFAULT);
      readArray2D("A", lrb_gid, lrb->A);
      readArray2D("B", lrb_gid, lrb->B);
      H5Gclose(lrb_gid);
      ++i_lrb;
    }
    H5Gclose(lrbs_gid);

    H5Gclose(hmat_gid);
    H5Fclose(file_id);

    isBuilt_FR_ = isBuilt_LR_ = isBuilt_ = true;
#endif
  }


   // -----------------------------------------------------------------------------
  void buildFR(const bie::MatrixGenerator<T>& matrix_gen){
    // construction of the full rank blocks
    std::cout << " Loop on full blocks construction  \n";
    std::cout << " N full blocks "<< pattern_.n_FRB << " \n";

    full_rank_blocks_.resize(pattern_.n_FRB);
#pragma omp parallel
    {
      std::vector<std::unique_ptr<il::Array2D<T>>> private_full_rank_blocks;

#if defined(_OPENMP)
      auto nthreads = omp_get_num_threads();
      int chunk_size = std::max(1, int(pattern_.n_FRB / nthreads / 100));
#endif
#pragma omp for nowait schedule(static,chunk_size)
      for (il::int_t i = 0; i < pattern_.n_FRB; i++) {
        il::int_t i0 = pattern_.FRB_pattern(1, i);
        il::int_t j0 = pattern_.FRB_pattern(2, i);
        il::int_t iend = pattern_.FRB_pattern(3, i);
        il::int_t jend = pattern_.FRB_pattern(4, i);

        const il::int_t ni = matrix_gen.blockSize() * (iend - i0);
        const il::int_t nj = matrix_gen.blockSize() * (jend - j0);

        std::unique_ptr<il::Array2D<T>> a =std::make_unique<il::Array2D<T>>  (ni, nj);
        matrix_gen.set(i0, j0, il::io, a->Edit());
        full_rank_blocks_[i] = std::move(a);
      }
    }
    isBuilt_FR_=true;
  }
////////////////////////////////////////////////////////////////////////////////
///
/// \param matrix_gen
/// \param epsilon
  void buildLR(const bie::MatrixGenerator<T>& matrix_gen,const double epsilon){
    // constructing the low rank blocks
    dof_dimension_=matrix_gen.blockSize();
    std::cout << " Loop on low rank blocks construction  \n";
    std::cout << " N low rank blocks "<< pattern_.n_LRB << " \n";
    std::cout << "dof_dimension: "<< dof_dimension_ <<"\n";

#if defined(_OPENMP)
    auto nthreads = omp_get_max_threads();
#endif

    low_rank_blocks_.resize(pattern_.n_LRB);
#pragma omp parallel
    {

#if defined(_OPENMP)
      int chunk_size = std::max(1, int(pattern_.n_LRB / nthreads / 100));
#endif

#pragma omp for nowait schedule(static, chunk_size)
      for (il::int_t i = 0; i < pattern_.n_LRB; i++) {
        il::int_t i0 = pattern_.LRB_pattern(1, i);
        il::int_t j0 = pattern_.LRB_pattern(2, i);
        il::int_t iend = pattern_.LRB_pattern(3, i);
        il::int_t jend = pattern_.LRB_pattern(4, i);
        il::Range range0{i0, iend}, range1{j0, jend};

        // we need a LRA generator virtual template similar to the Matrix generator...
        // here we have an if condition for the LRA call dependent on dof_dimension_
        bie::LowRank<T> lra;
        if (matrix_gen.blockSize()==1){
          lra = bie::adaptiveCrossApproximation<1>(matrix_gen, range0, range1, epsilon);
        } else if (matrix_gen.blockSize()==2){
          lra = bie::adaptiveCrossApproximation<2>(matrix_gen, range0, range1, epsilon);
        } else if (matrix_gen.blockSize()==3){
          lra = bie::adaptiveCrossApproximation<3>(matrix_gen, range0, range1, epsilon);
        } else {
          IL_UNREACHABLE;
        }
        //std::unique_ptr<bie::LowRank<T>> lra_p( new bie::LowRank<T>( std::move(lra) ) );

        // store the rank in the low_rank pattern
        pattern_.LRB_pattern(5, i) = lra.A.size(1);
        low_rank_blocks_[i] = std::make_unique<bie::LowRank<T>>(lra); // lra_p does not exist after such call
      }
    }
    isBuilt_LR_=true;
  }

  //-----------------------------------------------------------------------------
  // filling up the h-matrix sub-blocks
  void build(const bie::MatrixGenerator<T>& matrix_gen,const double epsilon){
    dof_dimension_ =matrix_gen.blockSize();
    size_[0]=matrix_gen.size(0);
    size_[1]=matrix_gen.size(1);
    buildFR(matrix_gen);
    buildLR(matrix_gen,epsilon);
    isBuilt_= isBuilt_FR_ && isBuilt_LR_;
  }
  //-----------------------------------------------------------------------------
  bool isBuilt() const {return isBuilt_;};
//
  il::int_t size(int k) const {return size_[k];};
//
  bie::HPattern pattern(){return pattern_;}; //returning the Hmat pattern

  il::int_t dofDimension() const {return dof_dimension_;};

//-----------------------------------------------------------------------------
  // getting the nb of entries of the hmatrix
  il::int_t nbOfEntries(){
    IL_EXPECT_FAST(isBuilt_);
    il::int_t n=0;
    for (il::int_t i=0;i<pattern_.n_FRB;i++){
      auto & a = *full_rank_blocks_[i];
      n+=a.size(0)*a.size(1);
    }
    for (il::int_t i=0;i<pattern_.n_LRB;i++) {
      il::Array2DView<double> a = (*low_rank_blocks_[i]).A.view();
      il::Array2DView<double> b = (*low_rank_blocks_[i]).B.view();
      n+=a.size(0)*a.size(1)+b.size(0)*b.size(1);
    }
    return n;
  }
  //-----------------------------------------------------------------------------
 // getting the compression ratio of the hmatrix (double which is <=1)
  double compressionRatio(){
    auto nb_elts = static_cast<double>(nbOfEntries());
    return nb_elts / static_cast<double>(size_[0]*size_[1]);
  }

  //--------------------------------------------------------------------------
  // H-Matrix vector multiplication without permutation
  // in - il:Array<T>
  // out - il:Array<T>
  il::Array<T> matvec(const il::Array<T> & x){
    IL_EXPECT_FAST(x.size()==size_[1]);
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

      il::ArrayEdit<T> yprivate = yprivate_storage.Edit(il::Range{0, size_[0]}, thread_num);
      for (il::int_t i = 0; i < size_[0]; ++i) {
        yprivate[i] = 0.;
      }
#else
      il::ArrayEdit<T> yprivate = y.Edit();
#endif

#if defined(_OPENMP)
      static bool first_time = true;
      int chunk_size = std::max(1, int(pattern_.n_FRB / nthreads / 100));
#pragma omp single
      if (first_time) {
        std::printf("  FRB - chunk size: %d\n", chunk_size);
      }
#endif

#pragma omp for nowait schedule(static, chunk_size)
      for (il::int_t i = 0; i < pattern_.n_FRB; i++) {
          il::int_t i0=pattern_.FRB_pattern(1,i);
          il::int_t j0=pattern_.FRB_pattern(2,i);
          il::int_t iend=pattern_.FRB_pattern(3,i);
          il::int_t jend=pattern_.FRB_pattern(4,i);

          il::Array2DView<T> a = (*full_rank_blocks_[i]).view();
          il::ArrayView<T> xs = x.view(il::Range{j0* dof_dimension_, jend* dof_dimension_});
          il::ArrayEdit<T> ys = yprivate.Edit(il::Range{i0*dof_dimension_, iend* dof_dimension_});

          il::blas(1.0, a, xs, 1.0, il::io, ys);
      }

#if defined(_OPENMP)
      chunk_size = std::max(1, int(pattern_.n_LRB / nthreads/ 100));
#pragma omp single
      if (first_time) {
        std::printf("  LRB - chunk size: %d\n", chunk_size);
        first_time = false;
      }
#endif

#pragma omp for nowait schedule(static, chunk_size)
      for (il::int_t ii = 0; ii < pattern_.n_LRB; ii++) {
        auto i0 = pattern_.LRB_pattern(1, ii);
        auto j0 = pattern_.LRB_pattern(2, ii);
        auto iend = pattern_.LRB_pattern(3, ii);
        auto jend = pattern_.LRB_pattern(4, ii);

        auto a = low_rank_blocks_[ii]->A.view();
        auto b = low_rank_blocks_[ii]->B.view();

        auto xs = x.view(il::Range{j0 * dof_dimension_, jend * dof_dimension_});
        auto ys = yprivate.Edit(il::Range{i0 * dof_dimension_, iend * dof_dimension_});
        const il::int_t r = a.size(1);
        il::Array<double> tmp{r, 0.0};

        il::blas(1.0, b, il::Dot::None, xs, 0.0, il::io,
                 tmp.Edit());  // Note here we have stored b (not b^T)
        il::blas(1.0, a, tmp.view(), 1.0, il::io, ys);
      }

#if defined(_OPENMP)
      il::int_t j = 0;
#pragma omp for schedule(static)
      for (il::int_t j = 0; j < y.size(); j++) {
        auto yview = yprivate_storage.view(il::Range{j, j+1}, il::Range{0, nthreads});
        for(il::int_t i = 0; i < nthreads; ++i) {
          y[j] += yview(0, i);
        }
      }
#endif
    }

    return y;
  }

  //--------------------------------------------------------------------------
  // H-Matrix vector multiplication with permutation for rectangular matrix cases (only 1 permutation)
  // in & out as std::vector
  // todo : write another one for the case of 2 permutations (rect. mat cases (for source != receivers)
  std::vector<T> matvecOriginal(const il::Array<il::int_t> & permutation,const std::vector<T> & x){

    il::Array<T> z{static_cast<il::int_t>(x.size())};
    // permutation of the dofs according to the re-ordering sue to clustering
    il::int_t ncolpoints = this->size(1)/dof_dimension_;
    for (il::int_t i = 0; i < ncolpoints; i++) {
      for (int j = 0; j < dof_dimension_; j++) {
        z[dof_dimension_ * i + j] = x[dof_dimension_ * permutation[i] + j];
      }
    }
    il::Array<T> y = this->matvec(z);
    std::vector<T> yout;
    yout.assign(y.size(), 0.);
    // permut back
    for (il::int_t i = 0; i < ncolpoints; i++) {
      for (int j = 0; j < dof_dimension_; j++) {
        yout[dof_dimension_ * permutation[i] + j] = y[dof_dimension_ * i + j];
      }
    }
    return yout;
  }
  ////////////////////////////////////////////////
  // matvect in and outs as std::vector
  std::vector<T> matvec(const std::vector<T> & x){
    il::Array<T> xil{static_cast<il::int_t>(x.size())};
  // todo find a better way to convert il::Array to std::vect and vice versa  !
    for (long i=0;i<xil.size();i++){
      xil[i]=x[i];
    }
    il::Array<T> yil=this->matvec(xil);
    std::vector<T> y;
    y.reserve(static_cast<long>(yil.size()));
    for (long i=0;i<yil.size();i++){
      y.push_back(yil[i]);
    }
    return y;
  }

/////-----------------------------------------------------------------
  std::vector<T> diagonal(){
      // return diagonal in permutted state....
      IL_EXPECT_FAST(isBuilt_FR_);
      il::int_t diag_size = il::max(size_[0],size_[1]);
      std::vector<T> diag(static_cast<long>(diag_size),0.);

      for (il::int_t k = 0;k < pattern_.n_FRB; k++) {
          il::Array2D<double> aux = *full_rank_blocks_[k];
          il::int_t i0 = pattern_.FRB_pattern(1, k);
          il::int_t j0 = pattern_.FRB_pattern(2, k);
          il::int_t iend = pattern_.FRB_pattern(3, k);
          il::int_t jend = pattern_.FRB_pattern(4, k);
          // check if it intersects the diagonal
          //
          bool in_lower = (i0 > j0) && (i0 > jend) && (iend > j0) && (iend > jend);
          bool in_upper = (i0 < j0) && (i0 < jend) && (iend < j0) && (iend < jend);
          if ( (!in_lower) && (!in_upper) ) // this fb intersect the diagonal....
          {
              for (il::int_t j = 0; j < aux.size(1); j++) {
                  for (il::int_t i = 0; i < aux.size(0); i++) {
                      if ((i+dof_dimension_*i0) == (j+dof_dimension_*j0) ) { // on diagonal !
                          diag[(i+dof_dimension_*i0)] = aux(i, j);
                      }
                  }
              }
          }
      }
    return diag;
  }
/////
  std::vector<T> diagonalOriginal(const il::Array<il::int_t> & permutation){
      // return diagonal in original state....
      il::int_t diag_size = il::max(size_[0],size_[1]);
      il::int_t ncolpoints= diag_size/dof_dimension_;
      std::vector<T> diag_raw=this->diagonal();
      std::vector<T> diag(diag_raw.size(), 0.);
      // permut back
      for (il::int_t i = 0; i < ncolpoints; i++) {
            for (int j = 0; j < dof_dimension_; j++) {
                diag[dof_dimension_ * permutation[i] + j] = diag_raw[dof_dimension_ * i + j];
            }
        }
      return diag;
  }


  //--------------------------------------------------------------------------
  void fullBlocksOriginal(const il::Array<il::int_t> & permutation, il::io_t, il::Array<T> &val_list,
                          il::Array<int> &pos_list){
// return the full blocks in the permutted Original dof state
// in the val_list and pos_list 1D arrays.
    IL_EXPECT_FAST(isBuilt_FR_);
    IL_EXPECT_FAST(permutation.size()*dof_dimension_==size_[1]);
    //  compute the number of  entries in the whole full rank blocks
    int nbfentry=0;
    for (il::int_t i=0;i<pattern_.n_FRB;i++) {
      il::Array2DView<double>  aux=((*full_rank_blocks_[i]).view());
      nbfentry=nbfentry+static_cast<int>(aux.size(0)*aux.size(1)) ;
    }

    // prepare outputs
    pos_list.Resize(nbfentry * 2);
    val_list.Resize(nbfentry);

    il::Array<int> permutDOF{dof_dimension_*permutation.size(),0};
    IL_EXPECT_FAST(permutDOF.size()==size_[0]);
    for (il::int_t i=0;i<permutation.size();i++){
      for (il::int_t j=0;j< dof_dimension_;j++){
        permutDOF[i* dof_dimension_ +j]=permutation[i]* dof_dimension_ +j;
      }
    }
     // loop on full rank and get i,j and val
    int nr=0; int npos=0;
    for (il::int_t k = 0;k < pattern_.n_FRB; k++) {
      il::Array2D<double>  aux=*full_rank_blocks_[k];
      il::int_t i0 = pattern_.FRB_pattern(1, k);
      il::int_t j0 = pattern_.FRB_pattern(2, k);
      il::int_t index=0;
      for (il::int_t j=0;j<aux.size(1);j++){
        for (il::int_t i=0;i<aux.size(0);i++){
          pos_list[npos+2*index]=permutDOF[(i+dof_dimension_*i0)];
          pos_list[npos+2*index+1]=permutDOF[(j+dof_dimension_*j0)];
          val_list[nr+index]=aux(i,j);
          index++;
        }
      }
      nr=nr + static_cast<int>(aux.size(0)*aux.size(1));
      npos=npos+static_cast<int>(2*aux.size(0)*aux.size(1));
    }
    std::cout << "done Full Block: nval " << val_list.size() << " / " << pos_list.size()/2
              << " n^2 " << (this->size_[0]) * (this->size_[1]) << "\n";
  }
};

}
#endif

#pragma clang diagnostic pop

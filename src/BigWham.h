//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 15.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications :: Nov. 18, 2021 - interfacing with the new Hmat

#pragma once

#include <iostream>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Dynamic.h>
#include <il/Map.h>

#include <hmat/cluster/cluster.h>
#include <hmat/hmatrix/Hmat.h>

#include <src/core/ElasticProperties.h>
#include <src/core/Mesh2D.h>
#include <src/core/Mesh3D.h>
#include <src/elasticity/PostProcessDDM_2d.h>
#include <src/elasticity/PostProcessDDM_3d.h>

// kernels.
#include <elasticity/2d/ElasticHMatrix2DP0.h>
#include <elasticity/2d/ElasticHMatrix2DP1.h>
#include <elasticity/3d/ElasticHMatrix3DR0.h>
#include <elasticity/3d/ElasticHMatrix3DR0displ.h>
#include <elasticity/3d/ElasticHMatrix3DR0_mode1Cartesian.h>
#include <elasticity/3d/ElasticHMatrix3DT0.h>
#include <elasticity/3d/ElasticHMatrix3DT0displ.h>
#include <elasticity/3d/ElasticHMatrix3DT6.h>
#include <elasticity/3d/ElasticHMatrix3DR0_modes2and3Cartesian.h>
#include <elasticity/3d/ElasticHMatrix3DT0_modes2and3.h>
#include <elasticity/2d/ElasticHMatrixAxi3DP0.h>


class Bigwhamio {
 private:

  bie::Hmat<double>  h_;  // the  Hmat object

  il::Array<il::int_t> permutation_;  // permutation list of the collocation points
  il::Array2D<double> collocationPoints_;  //  collocation points coordinates

  int dimension_;      // spatial dimension
  int dof_dimension_;  // number of dof per nodes / collocation points

  bool isBuilt_;  // True if the class instance is built

  // H-matrix parameters
  int max_leaf_size_;
  double eta_;
  double epsilon_aca_;
  // kernel
  std::string kernel_;

  // statistics
  double block_clstr_crtion_time_;
  double binary_clstr_tree_time_;
  double hmat_time_;

 public:
  //---------------------------------------------------------------------------
  Bigwhamio() { // default constructor
    dimension_ = 0;
    dof_dimension_ = 0;
    isBuilt_ = false;
    eta_ = 0.0;
    epsilon_aca_ = 0.001;
    max_leaf_size_ = 100;
    kernel_ = "none";

    block_clstr_crtion_time_ = 0.;
    binary_clstr_tree_time_ = 0.;
    hmat_time_ = 0.;
  };

  ~Bigwhamio() = default;

  void set(const std::vector<double>& coor, const std::vector<int>& conn,
           const std::string& kernel, const std::vector<double>& properties,
           const int max_leaf_size, const double eta, const double eps_aca) {
    // coor and conn are assumed to be passed in row-major storage format
    kernel_ = kernel;

    // switch depending on Kernels for mesh building
    max_leaf_size_ = max_leaf_size;
    eta_ = eta;
    epsilon_aca_ = eps_aca;

    std::cout << " Now setting things for kernel ... " << kernel_ << "\n";
    il::Timer tt;

    // if on kernel name - separating 2D and 3D kernels,
    // duplicating code for simplicity
    if ((kernel_ == "2DP1") || (kernel_ == "S3DP0") || (kernel_ == "Axi3DP0")) {
      // step 1 - create the mesh object
      dimension_ = 2;
      dof_dimension_ = 2;

      IL_ASSERT(coor.size() % dimension_ == 0);
      IL_ASSERT(conn.size() % dimension_ == 0);

      std::cout << "Number of nodes " << coor.size() / dimension_ << " .. mod"
                << (coor.size() % dimension_) << "\n";
      std::cout << " Number of elts " << conn.size() / dimension_ << "\n";

      il::int_t nvertex = coor.size() / dimension_;
      il::int_t nelts = conn.size() / dimension_;
      il::int_t nnodes_elts = dimension_;
      il::Array2D<double> Coor{nvertex, dimension_, 0.};  // columm major order
      il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};

      // set interpolation order
      int p = 0;
      if (kernel_ == "2DP1") {
        p = 1;
      }
      std::cout << " interpolation order  " << p << "\n";
      // populate mesh (loops could be optimized - passage row-major to
      // col-major)
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

      bie::Mesh mesh2d(Coor, Conn, p);
      std::cout << "... mesh done"  << "\n";
      std::cout << "Number elts " << mesh2d.numberOfElts() << "\n";
      collocationPoints_ = mesh2d.getCollocationPoints();
      std::cout << "Creating cluster tree - number of collocation pts" << collocationPoints_.size(0) << "\n";

      tt.Start();
      const bie::Cluster cluster =
          bie::cluster(max_leaf_size_, il::io, collocationPoints_);
      tt.Stop();
      std::cout << "Cluster tree creation time :  " << tt.time() << "\n";
      tt.Reset();
      permutation_ = cluster.permutation;

      // elastic properties
      std::cout << " properties vector size " << properties.size() << "\n";

      if (kernel_ == "2DP1" || kernel_ == "Axi3DP0") {
        IL_ASSERT(properties.size() == 2);
      } else if (kernel_ == "S3DP0") {
        IL_ASSERT(properties.size() == 3);
      }
      bie::ElasticProperties elas(properties[0], properties[1]);

      // now we setting up the matrix generator
      tt.Start();
      if (kernel_ == "2DP1")  // or maybe use switch
      {
        std::cout << "Kernel Isotropic ELasticity 2D P1 segment \n";
        const bie::ElasticHMatrix2DP1<double> M{collocationPoints_,permutation_, mesh2d, elas};

        h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);

      } else if (kernel_ == "S3DP0") {
        std::cout << "Kernel Isotropic ELasticity Simplified_3D (2D) P0 segment \n";
        const bie::ElasticHMatrix2DP0<double> M{collocationPoints_, permutation_, mesh2d, elas, properties[2]};
        h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
      }
      else if (kernel_ == "Axi3DP0") {
          std::cout << "Axisymmetric 3D (2D) P0 ring dislocation element  \n";
          const bie::ElasticHMatrixAxi3DP0<double> M{collocationPoints_, permutation_, mesh2d, elas};
          h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
      }
    }

    else if (kernel_ == "3DT6" || kernel_ == "3DR0_displ" ||
             kernel_ == "3DR0" || kernel_ == "3DT0" || kernel_ == "3DT0_displ" || kernel_ == "3DR0opening" || kernel_ == "3DR0shear" || kernel_ == "3DT0shear") {
      // step 1 - create the mesh object
      dimension_ = 3;
      il::int_t nnodes_elts = 0;  // n of nodes per element
      int p = 0;                  // interpolation order
      if (kernel_ == "3DT6") {
        dof_dimension_ = 3;
        nnodes_elts = 3;
        p = 2;
      } else if (kernel_ == "3DT0" || kernel_ == "3DT0_displ") {
        dof_dimension_ = 3;
        nnodes_elts = 3;
        p = 0;
      } else if (kernel_ == "3DR0_displ" || kernel_ == "3DR0") {
        dof_dimension_ = 3;
        nnodes_elts = 4;
        p = 0;
      } else if (kernel_ == "3DR0shear") {
          dof_dimension_ = 2;
          nnodes_elts = 4;
          p = 0;
      } else if (kernel_ == "3DT0shear") {
          dof_dimension_ = 2;
          nnodes_elts = 3;
          p = 0;
      } else if (kernel_ == "3DR0opening") {
        dof_dimension_ = 1;
        nnodes_elts = 4;
        p = 0;
      }
      else {
        std::cout << "Invalid kernel name ---\n";
        il::abort();
      };

      IL_ASSERT(conn.size() % nnodes_elts == 0);
      IL_ASSERT(coor.size() % dimension_ == 0);

      std::cout << " Number of nodes " << coor.size() / dimension_ << " .. mod " << (coor.size() % dimension_) << "\n";
      std::cout << " Number of elts " << conn.size() / nnodes_elts << "\n";
      std::cout << " Interpolation order  " << p << "\n";

      il::int_t nelts = conn.size() / nnodes_elts;
      il::int_t nvertex = coor.size() / dimension_;

      il::Array2D<double> Coor{nvertex, dimension_, 0.};  // columm major order
      il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};

      // populate mesh (loops could be optimized - passage row-major to
      // col-major)
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

      bie::Mesh3D mesh3d(Coor, Conn, p);
      std::cout << "... mesh done" << "\n";
      std::cout << " Number elts " << mesh3d.numberOfElts() << "\n";

      collocationPoints_ = mesh3d.getCollocationPoints();
      std::cout << " Coll points dim " << collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
      std::cout << "Creating cluster tree - number of collocation pts: "  << collocationPoints_.size(0) << "\n";

      tt.Start();
      const bie::Cluster cluster =bie::cluster(max_leaf_size_, il::io, collocationPoints_);
      tt.Stop();
      std::cout << "Cluster tree creation time :  " << tt.time() << "\n";
      binary_clstr_tree_time_ = tt.time();
      tt.Reset();

      permutation_ = cluster.permutation;
      tt.Start();
      std::cout << "Creating Block Cluster  Tree - \n";
      const il::Tree<bie::SubHMatrix, 4> block_tree =bie::hmatrixTreeIxI(collocationPoints_, cluster.partition,eta);
      tt.Stop();
      std::cout << "hmatrix   Block Cluster creation time :  " << tt.time() << "\n";
      block_clstr_crtion_time_ = tt.time();
      tt.Reset();
      std::cout << "coll points dim " << collocationPoints_.size(0) << " - "  << collocationPoints_.size(1) << "\n";

      // elastic properties
      std::cout << " properties vector size " << properties.size() << "\n";

      IL_ASSERT(properties.size() == 2);
      bie::ElasticProperties elas(properties[0], properties[1]);

      tt.Start();
      if (kernel_ == "3DT6")  // or maybe use switch
      {
        std::cout
            << "Hypersingular Kernel Isotropic ELasticity 3D T6 (quadratic) triangle \n";
        std::cout << "coll points dim " << collocationPoints_.size(0) << " - "
                  << collocationPoints_.size(1) << "\n";
        const bie::ElasticHMatrix3DT6<double> M{
            collocationPoints_, permutation_, mesh3d, elas, 0, 0};
        h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);

      } else if (kernel_ == "3DT0") {
        std::cout   << "Hypersingular Kernel Isotropic Elasticity 3D T0  triangle \n";
        std::cout << "coll points dim " << collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
        const bie::ElasticHMatrix3DT0<double> M{collocationPoints_, permutation_, mesh3d, elas,0};  // local_global = 0 if local-local, 1 if global-global
        h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
      } else if (kernel_ == "3DT0_displ"){
          std::cout   << "Singular Kernel Isotropic Elasticity 3D T0  triangle \n";
          std::cout << "coll points dim " << collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
          const bie::ElasticHMatrix3DT0displ<double> M{collocationPoints_, permutation_, mesh3d, elas,0};  // local_global = 0 if local-local, 1 if global-global
          h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
      } else if (kernel_ == "3DR0_displ" || kernel_ == "3DR0" || kernel_ == "3DR0opening" ||  kernel_ == "3DR0shear" || kernel_ == "3DT0shear") {
        std::cout << " Kernel Isotropic ELasticity 3D R0 (constant) rectangle \n";
        std::cout << "coll points dim " << collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
        if (kernel_ == "3DR0") {
          // DD to traction HMAT
          std::cout << "\n Kernel: "<< kernel_ << " " <<  "< Hypersingular traction kernel >"<< "\n  ";
          const bie::ElasticHMatrix3DR0<double> M{collocationPoints_, permutation_, mesh3d, elas, 0, 0};
          h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
        } else if (kernel_ == "3DR0_displ") {
          // DD to displacement HMAT
          std::cout << "\n Kernel: "<< kernel_ << " " <<  "< Singular displacement kernel >"<< "\n  ";
          const bie::ElasticHMatrix3DR0displ<double> M{collocationPoints_, permutation_, mesh3d, elas, 0, 0};
          h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
        } else if (kernel_ == "3DR0opening") {
          // DD to displacement HMAT
          std::cout << "\n Kernel: "<< kernel_ << " " <<  "< Hypersingular traction kernel >"<< "\n  ";
          const bie::ElasticHMatrix3DR0_mode1Cartesian<double> M{collocationPoints_, permutation_, mesh3d, elas};
          h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
        }
        else if (kernel_ == "3DR0shear") {
            // DD to displacement HMAT
            std::cout << "\n Kernel: "<< kernel_ << " " <<  "< Hypersingular traction kernel >"<< "\n  ";
            const bie::ElasticHMatrix3DR0_modes2and3Cartesian<double> M{collocationPoints_, permutation_, mesh3d, elas,0,0};
            h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
        }
        else if (kernel_ == "3DT0shear") {
            // DD to displacement HMAT
            std::cout << "\n Kernel: "<< kernel_ << " " <<  "< Hypersingular traction kernel >"<< "\n  ";
            const bie::ElasticHMatrix3DT0_modes2and3<double> M{collocationPoints_, permutation_, mesh3d, elas,0};
            h_.toHmat(M,cluster, collocationPoints_,  eta_,epsilon_aca_);
        }
      }

    } else {
      std::cout << "Invalid kernel name ---\n";
      return;
    }

    tt.Stop();
    std::cout << "HMAT --> built \n";
    std::cout << "coll points dim " << collocationPoints_.size(0) << " - "
              << collocationPoints_.size(1) << "\n";
    std::cout << "H mat set : CR = " << h_.compressionRatio()
              << " eps_aca " << epsilon_aca_ << " eta " << eta_ << "\n";
    std::cout << "H-mat time = :  " << tt.time() << "\n";
    hmat_time_ = tt.time();
    std::cout << "coll points dim " << collocationPoints_.size(0) << " - "
              << collocationPoints_.size(1) << "\n";

    if (h_.isBuilt()) {
      isBuilt_ = true;
    } else {
      isBuilt_ = false;
    }

    std::cout << "end of set() of bigwhamio object \n";
  }

  bool isBuilt() { return isBuilt_; };

  void hmatDestructor(){
      // this function will free the memory and set the hmat obj to its initial status prior to initialization
      // this will avoid ownership specifications at binding time
      //this->h_.~Hmat<double>();
      this->h_.hmatMemFree();
      };

  //---------------------------------------------------------------------------
  //  get and other methods below
  double getHmatTime(){ return hmat_time_;};
  double getBlockClstrTime() { return block_clstr_crtion_time_;};
  double getBinaryClstrTime() { return binary_clstr_tree_time_;};

  std::vector<double> getCollocationPoints() {
    IL_EXPECT_FAST(isBuilt_);
    std::cout << "beginning of getCollocationPoints bigwham \n";
    std::cout << " spatial dim :" << dimension_  << " collocation dim size :" << collocationPoints_.size(1) << "\n";
    std::cout << " collocation npoints :" << collocationPoints_.size(0) << "\n";
    std::cout << "coll points dim " << collocationPoints_.size(0) << " - "  << collocationPoints_.size(1) << "\n";

    IL_EXPECT_FAST(collocationPoints_.size(1) == dimension_);

    il::int_t npoints = collocationPoints_.size(0);

    std::vector<double> flat_col;
    flat_col.assign(npoints * dimension_, 0.);
    int index = 0;
    for (il::int_t i = 0; i < collocationPoints_.size(0); i++) {
      for (il::int_t j = 0; j < collocationPoints_.size(1); j++) {
        flat_col[index] = collocationPoints_(i, j);
        index++;
      }
    }
    std::cout << "end of getCollocationPoints bigwham \n";
    return flat_col;
  };

  //---------------------------------------------------------------------------
  std::vector<int> getPermutation() {
    IL_EXPECT_FAST(isBuilt_);
    std::cout << "coll points dim " << collocationPoints_.size(0) << " - "
              << collocationPoints_.size(1) << "\n";

    std::vector<int> permut;
    permut.assign(permutation_.size(), 0);
    for (il::int_t i = 0; i < permutation_.size(); i++) {
      permut[i] = permutation_[i];
    }
    return permut;
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  double getCompressionRatio() {
    IL_EXPECT_FAST(isBuilt_);
    return h_.compressionRatio();
  }

  std::string getKernel() { return kernel_; }

  int getSpatialDimension() const { return dimension_; }

  int getProblemDimension() const {return dof_dimension_;}

  long matrixSize(int k) { return h_.size(k); };

  //---------------------------------------------------------------------------
  std::vector<int> getHpattern() {
    // API function to output the hmatrix pattern
    //  as flattened list via a pointer
    //  the numberofblocks is also returned (by reference)
    //
    //  the pattern matrix is formatted as
    // row = 1 block : i_begin,j_begin, i_end,j_end,FLAG,entry_size
    // with FLAG=0 for full rank and FLAG=1 for low rank
    //
    // we output a flatten row-major order std::vector

    IL_EXPECT_FAST(isBuilt_);

    int fullRankPatternSize1, lowRankPatternSize1;

    bie::HPattern pattern = h_.pattern();

    long numberofblocks = pattern.n_B;
    long len = 6 * numberofblocks;
    std::cout << "number of blocks " << numberofblocks << "\n";

    std::vector<int> patternlist(len, 0);

    int index = 0;
    //  starts with full rank
    for (il::int_t j = 0; j < pattern.n_FRB; j++) {
      // check is low rank or not
    //  il::Array2DView<double> A = h_.asFullRank(s);
      patternlist[index++] = pattern.FRB_pattern(1, j);
      patternlist[index++] = pattern.FRB_pattern(2, j);
      patternlist[index++] = pattern.FRB_pattern(3, j)  ;
      patternlist[index++] = pattern.FRB_pattern(4, j) ;
      patternlist[index++] = 0;
      patternlist[index++] = (pattern.FRB_pattern(4, j)-pattern.FRB_pattern(2, j))  * (pattern.FRB_pattern(3, j)-pattern.FRB_pattern(1, j)) ;   // size of that sub blocks
    }
    // then low ranks
    for (il::int_t j = 0; j < pattern.n_LRB; j++) {

      patternlist[index++] = pattern.LRB_pattern(1, j);
      patternlist[index++] = pattern.LRB_pattern(2, j);
      patternlist[index++] = pattern.LRB_pattern(3, j) ;
      patternlist[index++] = pattern.LRB_pattern(4, j)  ;
      patternlist[index++] = 1;
      patternlist[index++] = pattern.LRB_pattern(5, j)  ; // the rank
    }
    // return a row major flatten vector
    return patternlist;
  }

  //---------------------------------------------------------------------------
  void getFullBlocks(std::vector<double>& val_list,
                     std::vector<int>& pos_list) {
    // return the full dense block entries of the hmat as
    // flattened lists
    // val_list(i) = H(pos_list(2*i),pos_list(2*i+1));
    // output in the original dof state (accounting for the permutation)

    IL_EXPECT_FAST(isBuilt_);

    il::Array<double> values{};
    il::Array<int> positions{};
    h_.fullBlocksOriginal(permutation_,il::io,values,positions);
//    std::cout << " checking values size" << values.size() <<  "\n";

    val_list.reserve(values.size());
    for(il::int_t i=0;i<values.size();i++){
        val_list.push_back(values[i]);
    }
    pos_list.reserve(positions.size());
    for(il::int_t i=0;i<positions.size();i++){
        pos_list.push_back(positions[i]);
    }

    std::cout << "number of entries " << val_list.size() << " - "<< pos_list.size() <<  "\n";
//    std::cout << " end val " << val_list[val_list.size()-1] << " pos" << pos_list[pos_list.size()-2] <<"-" << pos_list[pos_list.size()-1]  <<"\n ";
//    std::cout << " end val " << values[values.size()-1] << " pos" << positions[positions.size()-2] << "-" <<positions[positions.size()-1] <<"\n ";
    std::cout << " End of Bigwhamio getFullBlocks \n";
  }

  // ---------------------------------------------------------------------------
  std::vector<double> hdotProduct(const std::vector<double>& x) {
    IL_EXPECT_FAST(this->isBuilt_);
    IL_EXPECT_FAST(h_.size(0) == h_.size(1));
    IL_EXPECT_FAST(h_.size(1) == x.size());

    std::vector<double> y=h_.matvecOriginal(permutation_,x);

    return y;
  }

  // ---------------------------------------------------------------------------
  std::vector<double> hdotProductInPermutted(const std::vector<double>& x) {
    IL_EXPECT_FAST(this->isBuilt_);
    IL_EXPECT_FAST(h_.size(0) == h_.size(1));
    IL_EXPECT_FAST(h_.size(1) == x.size());
    std::vector<double> y=h_.matvec_stdvect(x);
    return y;
  }

  //---------------------------------------------------------------------------
  int getNodesPerElem() {
    int nnodes_elts = 0;
    if (kernel_ == "3DT6" || kernel_ == "3DT0") {
      nnodes_elts = 3;
    } else if (kernel_ == "3DR0_displ" || kernel_ == "3DR0" || kernel_ =="3DR0opening") {
      nnodes_elts = 4;
    } else if (kernel_ == "2DP1" || kernel_ == "Axi3DP0") {
      nnodes_elts = 2;
    } else if (kernel_ == "S3DP0") {
      nnodes_elts = 3;
    } else {
      std::cout << "Invalid kernel name ---\n";
      il::abort();
    };
    return nnodes_elts;
  }
  //---------------------------------------------------------------------------
  int getInterpOrder() {
    int p = -1;
    if (kernel_ == "3DT6") {
      p = 2;
    } else if (kernel_ == "3DT0" || kernel_ == "Axi3DP0") {
      p = 0;
    } else if (kernel_ == "3DR0_displ" || kernel_ == "3DR0" ||  kernel_ == "3DT0_displ" || (kernel_ == "S3DP0") || kernel_ =="3DR0opening" ) {
      p = 0;
    } else if (kernel_ == "2DP1") {
      p = 1;
    } else {
      std::cout << "Invalid kernel name ---\n";
      il::abort();
    };
    return p;
  }
  //---------------------------------------------------------------------------
  il::Array2D<il::int_t> getConn(const std::vector<int>& conn)  {
    int nnodes_elts = getNodesPerElem();
    il::int_t nelts = conn.size() / nnodes_elts;

    il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};
    int index = 0;
    for (il::int_t i = 0; i < Conn.size(0); i++) {
      for (il::int_t j = 0; j < Conn.size(1); j++) {
        Conn(i, j) = conn[index];
        index++;
      }
    }
    return Conn;
  }
  //---------------------------------------------------------------------------
  il::Array2D<double> getCoor(const std::vector<double>& coor) const {
    il::int_t nvertex = coor.size() / dimension_;
    il::Array2D<double> Coor{nvertex, dimension_, 0.};  // columm major order
    // populate mesh (loops could be optimized - passage row-major to col-major)
    int index = 0;
    for (il::int_t i = 0; i < Coor.size(0); i++) {
      for (il::int_t j = 0; j < Coor.size(1); j++) {
        Coor(i, j) = coor[index];
        index++;
      }
    }
    return Coor;
  }

  //---------------------------------------------------------------------------
  std::vector<double> computeStresses(
      const std::vector<double>& solution, const std::vector<double>& obsPts,
      const int npts,  // do we really need it? we know dimension + size of inputs
      const std::vector<double>& properties, const std::vector<double>& coor,
      const std::vector<int>& conn, const bool are_dd_global) {
    /* BE CAREFUL 2D CASES NEVER TESTED! */

    // PURPOSE: compute stresses at list of points (of size npts )
    //          from a solution vector.
    // INPUT:   "solution" a flattened list containing the solution in terms of
    // DDs
    //          "obsPts" a flattened list containing the observation points
    //          coordinates [ x(1), y(1), z(1), ... ,x(npts), y(npts), z(npts) ]
    //          "npts" the number of points
    //          "properties" is a vector of size 2 or 3 depending on the kernel
    //          It should contain at least
    //          - properties[0] = YoungModulus
    //          - properties[1] = PoissonRatio
    //          - properties[2] = fracture height (mandatory only for "S3DP0"
    //          kernel) "coor" are the coordinates of the vertices of the mesh
    //          "conn" is the connectivity matrix vertices to elements of the
    //          mesh
    // OUTPUT:  a flattened list containing the stress at each required point

    std::cout << " Computing stress tensor at obs points...\n";

    IL_EXPECT_FAST(this->isBuilt_);
    // note solution MUST be of length = number of dofs !

    il::Array2D<double> pts{npts, dimension_};
    il::int_t numberofunknowns = solution.size();
    il::Array<double> solu{numberofunknowns};
    il::Array2D<double> stress;
    bie::ElasticProperties elas(properties[0], properties[1]);

    int p = getInterpOrder();  // interpolation order

    il::Array2D<il::int_t> Conn = getConn(conn);
    il::Array2D<double> Coor = getCoor(coor);
    for (il::int_t i = 0; i < numberofunknowns; i++) {
      solu[i] = solution[i];
    }
    switch (dimension_) {
      case 2: {
        std::cout << "\n WARNING: computeStresses never tested !!\n";

        il::int_t index = 0;
        for (il::int_t i = 0; i < npts; i++) {
          pts(i, 0) = obsPts[index++];
          pts(i, 1) = obsPts[index++];
        }

        std::cout << " compute stress - " << kernel_ << "\n";
        if (kernel_ == "2DP1") {
          bie::Mesh mesh2d(Coor, Conn, p);
          stress = bie::computeStresses2D(pts, mesh2d, elas, solu,
                                          bie::point_stress_s2d_dp1_dd, 0.);
        } else if (kernel_ == "S3DP0") {
          bie::Mesh mesh2d(Coor, Conn, p);
          stress = bie::computeStresses2D(pts, mesh2d, elas, solu,
                                          bie::point_stress_s3d_dp0_dd,
                                          properties[2]);
        }
        break;
      }
      case 3: {
        /*
            not implemented yet for 3DT6
        */
        il::int_t index = 0;
        for (il::int_t i = 0; i < npts; i++) {
          pts(i, 0) = obsPts[index++];
          pts(i, 1) = obsPts[index++];
          pts(i, 2) = obsPts[index++];
        }

        std::cout << "\n compute stress - " << kernel_ << "\n";
        if (kernel_ == "3DT6") {
          bie::Mesh3D mesh3d(Coor, Conn, p);
          std::cout << "\n WARNING: not implemented !!\n";
          il::abort();
        } else if (kernel_ == "3DT0") {
          bie::Mesh3D mesh3d(Coor, Conn, p, false);
          stress = bie::computeStresses3D(
              pts, mesh3d, elas, solu, bie::point_stress_3DT0, are_dd_global);
        } else if (kernel_ == "3DR0" || kernel_ == "3DR0_displ") {
          bie::Mesh3D mesh3d(Coor, Conn, p, false);
          stress = bie::computeStresses3D(pts, mesh3d, elas, solu, bie::point_stress_3DR0, are_dd_global);
        }
        break;
      }
    }

    std::vector<double> stress_out(stress.size(0) * stress.size(1));

    il::int_t index = 0;
    for (il::int_t i = 0; i < stress.size(0); i++) {
      for (il::int_t j = 0; j < stress.size(1); j++) {
        stress_out[index] = stress(i, j);
        index = index + 1;
      }  // loop on the stress components
    }    // loop on the observation points

    return stress_out;
  }

  //---------------------------------------------------------------------------
  std::vector<double> computeDisplacements(
      std::vector<double>& solution, std::vector<double>& obsPts, int npts,
      const std::vector<double>& properties, const std::vector<double>& coor,
      const std::vector<int>& conn, bool are_dd_global) {
    // PURPOSE: compute displacements at list of points (of size npts )
    //          from a solution vector.
    // INPUT:   "solution" a flattened list containing the solution in terms of
    // DDs
    //          "obsPts" a flattened list containing the observation points
    //          coordinates [ x(1), y(1), z(1), ... ,x(npts), y(npts), z(npts) ]
    //          "npts" the number of points
    //          "properties" is a vector of size 2 or 3 depending on the kernel
    //          It should contain at least
    //          - properties[0] = YoungModulus
    //          - properties[1] = PoissonRatio
    //          - properties[2] = fracture height (mandatory only for "S3DP0"
    //          kernel) "coor" are the coordinates of the nodes of the mesh
    //          "conn" is the connectivity matrix node to elements of the mesh
    // OUTPUT:  a flattened list containing the displacements at each required
    // point

    IL_EXPECT_FAST(this->isBuilt_);
    // note solution MUST be of length = number of dofs !

    il::Array2D<double> pts{npts, dimension_};
    il::int_t numberofunknowns = solution.size();
    il::Array<double> solu{numberofunknowns};
    il::Array2D<double> displacements;
    bie::ElasticProperties elas(properties[0], properties[1]);

    int p = getInterpOrder();  // interpolation order

    il::Array2D<il::int_t> Conn = getConn(conn);
    il::Array2D<double> Coor = getCoor(coor);
    for (il::int_t i = 0; i < numberofunknowns; i++) {
      solu[i] = solution[i];
    }
    switch (dimension_) {
      case 2: {
        il::int_t index = 0;
        for (il::int_t i = 0; i < npts; i++) {
          pts(i, 0) = obsPts[index++];
          pts(i, 1) = obsPts[index++];
        }

        std::cout << " compute displacement - " << kernel_ << "\n";
        if (kernel_ == "2DP1") {
          bie::Mesh mesh2d(Coor, Conn, p);
          std::cout << "\n WARNING: not implemented !!\n";
          il::abort();
        } else if (kernel_ == "S3DP0") {
          bie::Mesh mesh2d(Coor, Conn, p);
          std::cout << "\n WARNING: not implemented !!\n";
          il::abort();
        }
        break;
      }
      case 3: {
        /*
            implemented only for constant DD
        */
        il::int_t index = 0;
        for (il::int_t i = 0; i < npts; i++) {
          pts(i, 0) = obsPts[index++];
          pts(i, 1) = obsPts[index++];
          pts(i, 2) = obsPts[index++];
        }
        std::cout << "\n compute displacement - " << kernel_ << "\n";
        if (kernel_ == "3DT6") {
          bie::Mesh3D mesh3d(Coor, Conn, p);
          std::cout << "\n WARNING: not implemented !!\n";
          il::abort();
        } else if (kernel_ == "3DR0_displ" || kernel_ == "3DR0") {
            bie::Mesh3D mesh3d(Coor, Conn, p, false);
            displacements = bie::computeDisplacements3D(pts, mesh3d, elas, solu, bie::point_displacement_3DR0,are_dd_global);
        } else if (kernel_ == "3DT0" || kernel_== "3DT0_displ"){
            bie::Mesh3D mesh3d(Coor, Conn, p, false);
            displacements = bie::computeDisplacements3D(pts, mesh3d, elas, solu, bie::point_displacement_3DT0,are_dd_global);
        }
        break;
      }
    }

    std::vector<double> displacements_out(displacements.size(0) *  displacements.size(1));

    il::int_t index = 0;
    for (il::int_t i = 0; i < displacements.size(0); i++) {
      for (il::int_t j = 0; j < displacements.size(1); j++) {
        displacements_out[index] = displacements(i, j);
        index = index + 1;
      }  // loop on the displacements components
    }    // loop on the observation points

    return displacements_out;
  }




};  // end class bigwhamio


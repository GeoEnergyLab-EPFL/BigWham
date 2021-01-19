//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 15.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2020.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications :: Nov. 12 2020

#include <iostream>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Dynamic.h>
#include <il/Map.h>
#include <il/SparseMatrixCSR.h>

#include <Hmat-lib/cluster/cluster.h>
#include <Hmat-lib/compression/toHMatrix.h>
#include <Hmat-lib/hmatrix/HMatrix.h>
#include <Hmat-lib/hmatrix/HMatrixUtils.h>
#include <Hmat-lib/linearAlgebra/blas/hdot.h>

#include <src/core/ElasticProperties.h>
#include <src/core/Mesh2D.h>
#include <src/core/Mesh3D.h>
#include <src/elasticity/PostProcessDDM.h>

// kernels.
#include <elasticity/2d/ElasticHMatrix2DP0.h>
#include <elasticity/2d/ElasticHMatrix2DP1.h>
#include <elasticity/3d/ElasticHMatrix3DT6.h>
#include <elasticity/3d/ElasticHMatrix3DR0.h>

//#pragma once

//#include <il/Gmres.h>
//#include <Hmat-lib/linearAlgebra/factorization/luDecomposition.h>
//#include <src/elasticity/jacobi_prec_assembly.h>  // for diagonal
//#include <src/solvers/HIterativeSolverUtilities.h>


class Bigwhamio
        {
          private:
          il::HMatrix<double> h_; // dd to traction
          il::HMatrix<double> hdispl_; // dd to displacements
          //   arrays storing the pattern (to speed up the hdot ...).
          il::Array2D<il::int_t> lr_pattern_;  // low rank block pattern
          il::Array2D<il::int_t> fr_pattern_;  // full rank block pattern
          il::Array2D<il::int_t> lr_pattern_hdispl_;  // low rank block pattern dd to displacement h
          il::Array2D<il::int_t> fr_pattern_hdispl_;  // full rank block pattern dd to displacement h

          il::Array<il::int_t> permutation_;  // permutation of the dof.

          il::Array2D<double> collocationPoints_;  //  collocation points coordinates

          int dimension_;      // spatial dimension
          int dof_dimension_;  // number of dof per nodes / collocation points

          bool isBuilt_;  // if the class instance is built
                          //  bool isLU_=false;

          // H-matrix parameters
          int max_leaf_size_;
          double eta_;
          double epsilon_aca_;
          //  double epsilon_lu_;
          std::string kernel_;

          public:
          //---------------------------------------------------------------------------

          Bigwhamio(){
              dimension_=0;dof_dimension_=0;
              isBuilt_= false;
              eta_=0.;
              epsilon_aca_=0.001;max_leaf_size_=1;
              kernel_="none";
          };
          ~Bigwhamio() = default;
          void set(const std::vector<double>& coor, const std::vector<int64_t>& conn,
                   const std::string& kernel, const std::vector<double>& properties,
                   const int max_leaf_size,const double eta,const double eps_aca)
                   {
                        // coor and conn are assumed to be passed in row-major storage format
                        kernel_ = kernel;

                        // switch depending on Kernels for mesh building
                        max_leaf_size_ = max_leaf_size;
                        eta_ = eta;
                        epsilon_aca_ = eps_aca;
                        std::cout << " Now setting things... " << kernel_ << "\n";
                        il::Timer tt;

                        // if on kernel name - separating 2D and 3D kernels,
                        // duplicating code for simplicity
                        if ((kernel_ == "2DP1") || (kernel_ == "S3DP0")) {
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
                          il::Array2D<double> Coor{nvertex, dimension_,0.}; // columm major order
                          il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};

                          // interpolation order
                          int p = 0;
                          if (kernel_ == "2DP1") {
                            p = 1;
                          }
                          std::cout << " interpolation order  " << p << "\n";
                          // populate mesh (loops could be optimized - passage row-major to col-major)
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
                          std::cout << "... mesh done"<< "\n";
                          std::cout << "Number elts " << mesh2d.numberOfElts() <<"\n";

                          collocationPoints_ = mesh2d.getCollocationPoints();
                          std::cout << "Creating cluster tree - number of collocation pts" << collocationPoints_.size(0) <<"\n";

                          tt.Start();
                          const il::Cluster cluster =
                              il::cluster(max_leaf_size_, il::io, collocationPoints_);
                          tt.Stop();
                          std::cout << "Cluster tree creation time :  " << tt.time() << "\n";
                          tt.Reset();

                          permutation_ = cluster.permutation;
                          tt.Start();
                          std::cout << "Creating hmatrix  Tree - \n";
                          il::Tree<il::SubHMatrix, 4> hmatrix_tree =
                              il::hmatrixTree(collocationPoints_, cluster.partition, eta_);
                          tt.Stop();
                          std::cout << "hmatrix  tree creation time :  " << tt.time() << "\n";
                          tt.Reset();

                          // elastic properties
                          std::cout <<" properties vector size " << properties.size() <<"\n";

                          if (kernel_ == "2DP1") {
                            IL_ASSERT(properties.size() == 2);
                          } else if (kernel_ == "S3DP0") {
                            IL_ASSERT(properties.size() == 3);
                          }
                          bie::ElasticProperties elas(properties[0], properties[1]);

                          // now we can populate the h- matrix
                          tt.Start();
                          if (kernel_ == "2DP1")  // or maybe use switch
                          {
                            std::cout << "Kernel Isotropic ELasticity 2D P1 segment \n";
                            const bie::ElasticHMatrix2DP1<double> M{collocationPoints_,
                                                                    permutation_, mesh2d, elas};
                            h_ = il::toHMatrix(M, hmatrix_tree, epsilon_aca_);  //

                          } else if (kernel_ == "S3DP0") {
                            std::cout
                                << "Kernel Isotropic ELasticity Simplified_3D (2D) P0 segment \n";

                            const bie::ElasticHMatrix2DP0<double> M{
                                collocationPoints_, permutation_, mesh2d, elas, properties[2]};
                            h_ = il::toHMatrix(M, hmatrix_tree, epsilon_aca_);  //
                          }
                          tt.Stop();
                          std::cout << "H-mat time = :  " << tt.time() << "\n";
                          std::cout << "H mat set : CR = " << il::compressionRatio(h_)
                                    << " eps_aca " << epsilon_aca_ << " eta " << eta_ << "\n";
                          tt.Reset();
                        } else if (kernel_=="3DT6" || kernel_=="3DR0") {
                          // check this  NOTE 1: the 3D mesh uses points and connectivity matrix that are
                          // transposed w.r. to the 2D mesh

                          // step 1 - create the mesh object
                          dimension_ = 3;
                          dof_dimension_ = 3;

                          il::int_t nnodes_elts = 0; // n of nodes per element
                          int p = 0; // interpolation order

                          if (kernel_=="3DT6") {nnodes_elts = 3; p = 2;}
                          else if (kernel_=="3DR0") {nnodes_elts = 4; p = 0;}
                          else {std::cout << "Invalid kernel name ---\n"; return; };

                          IL_ASSERT(conn.size() % nnodes_elts == 0);
                          IL_ASSERT(coor.size() % dimension_ == 0);

                          std::cout << " Number of nodes " << coor.size() / dimension_ << " .. mod "
                                    << (coor.size() % dimension_) << "\n";
                          std::cout << " Number of elts " << conn.size() / nnodes_elts << "\n";
                          std::cout << " Interpolation order  " << p << "\n";

                          il::int_t nelts = conn.size() / nnodes_elts;
                          il::int_t nvertex = coor.size() / dimension_;

                          il::Array2D<double> Coor{nvertex, dimension_,0.}; // columm major order
                          il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};


                          // populate mesh (loops could be optimized - passage row-major to col-major)
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
                          std::cout << "... mesh done"<< "\n";
                          std::cout << " Number elts " << mesh3d.numberElts() <<"\n";

                          collocationPoints_ = mesh3d.getCollocationPoints();

                          std::cout << " Coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
                          std::cout << "Creating cluster tree - number of collocation pts: " << collocationPoints_.size(0) <<"\n";

                          tt.Start();
                          const il::Cluster cluster = il::cluster(max_leaf_size_, il::io, collocationPoints_);
                          tt.Stop();
                          std::cout << "Cluster tree creation time :  " << tt.time() << "\n";
                          tt.Reset();

                          permutation_ = cluster.permutation;
                          tt.Start();
                          std::cout << "Creating hmatrix  Tree - \n";
                          il::Tree<il::SubHMatrix, 4> hmatrix_tree =
                              il::hmatrixTree(collocationPoints_, cluster.partition, eta_);
                          tt.Stop();
                          std::cout << "hmatrix  tree creation time :  " << tt.time() << "\n";
                          tt.Reset();
                          std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";

                          // elastic properties
                          std::cout <<" properties vector size " << properties.size() <<"\n";

                          IL_ASSERT(properties.size() == 2);
                          bie::ElasticProperties elas(properties[0], properties[1]);

                          // now we can populate the h- matrix
                          tt.Start();
                          if (kernel_ == "3DT6")  // or maybe use switch
                          {
                            std::cout << "Kernel Isotropic ELasticity 3D T6 (quadratic) triangle \n";
                            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
                            const bie::ElasticHMatrix3DT6<double> M{
                                collocationPoints_, permutation_, mesh3d, elas, 1, 1};
                            h_ = il::toHMatrix(M, hmatrix_tree, epsilon_aca_);  //
                            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
                          }
                          else if (kernel_ == "3DR0")
                          {
                            std::cout << "Kernel Isotropic ELasticity 3D R0 (constant) rectangle \n";
                            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";

                            // DD to traction HMAT
                            const bie::ElasticHMatrix3DR0<double> M{collocationPoints_, permutation_, mesh3d, elas, 0, 0,1};
                            h_ = il::toHMatrix(M, hmatrix_tree, epsilon_aca_);
                            std::cout << "DD to traction HMAT --> built \n";

                            // DD to displacement HMAT
                            const bie::ElasticHMatrix3DR0<double> Mdispl{collocationPoints_, permutation_, mesh3d, elas, 0, 0,0};
                            hdispl_ = il::toHMatrix(Mdispl, hmatrix_tree, epsilon_aca_);
                            std::cout << "DD to displacement HMAT --> built \n";

                            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";
                            std::cout << "H mat DD2traction set : CR = " << il::compressionRatio(h_)
                                        << " eps_aca " << epsilon_aca_ << " eta " << eta_ << "\n";
                            std::cout << "H mat DD2displacements set : CR = " << il::compressionRatio(hdispl_)
                                        << " eps_aca " << epsilon_aca_ << " eta " << eta_ << "\n";
                          }
                          tt.Stop();
                          std::cout << "H-mat time = :  " << tt.time() << "\n";
                          tt.Reset();
                          std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";

                        }
                        else {std::cout << "Invalid kernel name ---\n"; return;
                        };

                        tt.Start();

                        setHpattern("DD2traction");  // set Hpattern
                        if (kernel_ == "3DR0")
                        {
                            setHpattern("DD2displacements");  // set Hpattern
                            std::cout << "now saving the H-mat patterns for DD to displacements and DD to tractions...  ";
                        }
                        else{ std::cout << "now saving the H-mat pattern ...  ";}
                        tt.Stop();
                        std::cout << " in " << tt.time() << "\n";

                        if (h_.isBuilt()) {
                          isBuilt_ = true;
                        } else {
                          isBuilt_ = false;
                        }

                        std::cout << "end of set() of bigwhamio object \n";
                   }

         //---------------------------------------------------------------------------
          void setHpattern(std::string option)
          {
                // store the h pattern in a il::Array2D<double> for future use
                il::HMatrix<double> h;
                if(option == "DD2traction") {h = h_;}
                else if (option == "DD2displacements") {h = hdispl_;}
                else {std::cout << "Non valid option \n";}

                IL_EXPECT_FAST(h.isBuilt());
                il::Array2D<il::int_t> pattern = il::output_hmatPattern(h);
                //  separate full rank and low rank blocks to speed up the hdot
                il::Array2D<il::int_t> lr_patt{pattern.size(0), 0};
                il::Array2D<il::int_t> fr_patt{pattern.size(0), 0};
                lr_patt.Reserve(3, pattern.size(1));
                fr_patt.Reserve(3, pattern.size(1));
                il::int_t nfb = 0;
                il::int_t nlb = 0;
                for (il::int_t i = 0; i < pattern.size(1); i++) {
                    il::spot_t s(pattern(0, i));
                    if (h.isFullRank(s)) {
                    fr_patt.Resize(3, nfb + 1);
                    fr_patt(0, nfb) = pattern(0, i);
                    fr_patt(1, nfb) = pattern(1, i);
                    fr_patt(2, nfb) = pattern(2, i);
                    nfb++;
                    } else if (h.isLowRank(s)) {
                    lr_patt.Resize(3, nlb + 1);
                    lr_patt(0, nlb) = pattern(0, i);
                    lr_patt(1, nlb) = pattern(1, i);
                    lr_patt(2, nlb) = pattern(2, i);
                    nlb++;
                    } else {
                    std::cout << "error in pattern !\n";
                    il::abort();
                    }
                }
                if(option == "DD2traction") {
                    lr_pattern_ = lr_patt;
                    fr_pattern_ = fr_patt;}
                else if (option == "DD2displacements") {
                    lr_pattern_hdispl_ = lr_patt;
                    fr_pattern_hdispl_ = fr_patt;}
                else {std::cout << "Non valid option \n";}
          }

          bool isBuilt() {return isBuilt_;} ;

          //---------------------------------------------------------------------------
          //  get and other methods below
          std::vector<double> getCollocationPoints()
          {
            IL_EXPECT_FAST(isBuilt_);
            std::cout << "beginning of getCollocationPoints bigwham \n";
            std::cout << " spatial dim :" << dimension_ << " collocation dim size :" << collocationPoints_.size(1) << "\n";
            std::cout << " collocation npoints :" << collocationPoints_.size(0) << "\n";
            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";

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
          std::vector<int> getPermutation()
          {
            IL_EXPECT_FAST(isBuilt_);
            std::cout << "coll points dim "<< collocationPoints_.size(0) << " - " << collocationPoints_.size(1) << "\n";

            std::vector<int> permut;
            permut.assign(permutation_.size(), 0);
            for (il::int_t i = 0; i < permutation_.size(); i++) {
              permut[i] = permutation_[i];
            }
            return permut;
          }
          //---------------------------------------------------------------------------

          //---------------------------------------------------------------------------
          double getCompressionRatio(bool UseTractionKernel = true)
          {
            IL_EXPECT_FAST(isBuilt_);
              if (UseTractionKernel) {return il::compressionRatio(h_);}
              else {return il::compressionRatio(hdispl_);}
          }

          std::string getKernel()  {return  kernel_;}

          int getSpatialDimension() const  {return dimension_;}

          int matrixSize(int k, bool UseTractionKernel = true) {
              if (UseTractionKernel) {return  h_.size(k);}
              else {return  hdispl_.size(k);}};

          //---------------------------------------------------------------------------
             std::vector<int> getHpattern(bool UseTractionKernel = true)
          {
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

            if (UseTractionKernel){
                lowRankPatternSize1 =  lr_pattern_.size(1);
                fullRankPatternSize1 = fr_pattern_.size(1);
            }
            else {
                lowRankPatternSize1 =  lr_pattern_hdispl_.size(1);
                fullRankPatternSize1 = fr_pattern_hdispl_.size(1);
            };

            int numberofblocks = lowRankPatternSize1 + fullRankPatternSize1;
            int len = 6 * numberofblocks;
            std::cout << "number of blocks " << numberofblocks << "\n";

            std::vector<int> patternlist(len,0);

            int index = 0;
            //  starts with full rank

            if (UseTractionKernel){
                  for (il::int_t j = 0; j < fullRankPatternSize1; j++) {
                      il::spot_t s(fr_pattern_(0, j));
                      // check is low rank or not
                      il::Array2DView<double> A = h_.asFullRank(s);
                      //        std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," <<
                      //        pat_SPOT(2,i) <<
                      //                  "/ " << pat_SPOT(1,i)+A.size(0)-1 << ","
                      //                  <<pat_SPOT(2,i)+A.size(1)-1 << " -   "  << k << " - "
                      //                  << A.size(0)*A.size(1) << "\n";
                      patternlist[index++] = fr_pattern_(1, j);
                      patternlist[index++] = fr_pattern_(2, j);
                      patternlist[index++] = fr_pattern_(1, j) + A.size(0) - 1;
                      patternlist[index++] = fr_pattern_(2, j) + A.size(1) - 1;
                      patternlist[index++] = 0;
                      patternlist[index++] = A.size(0) * A.size(1);
                  }
            }
            else {
                  for (il::int_t j = 0; j < fullRankPatternSize1; j++) {
                      il::spot_t s(fr_pattern_hdispl_(0, j));
                      // check is low rank or not
                      il::Array2DView<double> A = h_.asFullRank(s);
                      //        std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," <<
                      //        pat_SPOT(2,i) <<
                      //                  "/ " << pat_SPOT(1,i)+A.size(0)-1 << ","
                      //                  <<pat_SPOT(2,i)+A.size(1)-1 << " -   "  << k << " - "
                      //                  << A.size(0)*A.size(1) << "\n";
                      patternlist[index++] = fr_pattern_hdispl_(1, j);
                      patternlist[index++] = fr_pattern_hdispl_(2, j);
                      patternlist[index++] = fr_pattern_hdispl_(1, j) + A.size(0) - 1;
                      patternlist[index++] = fr_pattern_hdispl_(2, j) + A.size(1) - 1;
                      patternlist[index++] = 0;
                      patternlist[index++] = A.size(0) * A.size(1);
                  }
            };


            // then low ranks
            if (UseTractionKernel) {
                for (il::int_t j = 0; j < lowRankPatternSize1; j++) {
                    il::spot_t s(lr_pattern_(0, j));
                    il::Array2DView<double> A = h_.asLowRankA(s);
                    il::Array2DView<double> B = h_.asLowRankB(s);
                    //        std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," <<
                    //        pat_SPOT(2,i) <<
                    //                  "/ " << pat_SPOT(1,i)+A.size(0)-1 << ","
                    //                  <<pat_SPOT(2,i)+B.size(0)-1
                    //                  << " - "  <<  k <<  " - "  <<
                    //                  A.size(0)*A.size(1)+B.size(0)*B.size(1) << "\n";
                    patternlist[index++] = lr_pattern_(1, j);
                    patternlist[index++] = lr_pattern_(2, j);
                    patternlist[index++] = lr_pattern_(1, j) + A.size(0) - 1;
                    patternlist[index++] = lr_pattern_(2, j) + B.size(0) - 1;
                    patternlist[index++] = 1;
                    patternlist[index++] = A.size(0) * A.size(1) + B.size(0) * B.size(1);
                }
            }
            else{
                for (il::int_t j = 0; j < lowRankPatternSize1; j++) {
                    il::spot_t s(lr_pattern_hdispl_(0, j));
                    il::Array2DView<double> A = h_.asLowRankA(s);
                    il::Array2DView<double> B = h_.asLowRankB(s);
                    //        std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," <<
                    //        pat_SPOT(2,i) <<
                    //                  "/ " << pat_SPOT(1,i)+A.size(0)-1 << ","
                    //                  <<pat_SPOT(2,i)+B.size(0)-1
                    //                  << " - "  <<  k <<  " - "  <<
                    //                  A.size(0)*A.size(1)+B.size(0)*B.size(1) << "\n";
                    patternlist[index++] = lr_pattern_hdispl_(1, j);
                    patternlist[index++] = lr_pattern_hdispl_(2, j);
                    patternlist[index++] = lr_pattern_hdispl_(1, j) + A.size(0) - 1;
                    patternlist[index++] = lr_pattern_hdispl_(2, j) + B.size(0) - 1;
                    patternlist[index++] = 1;
                    patternlist[index++] = A.size(0) * A.size(1) + B.size(0) * B.size(1);
                }
            }
            // return a row major flatten vector
            return patternlist;
          }

          //---------------------------------------------------------------------------
          void getFullBlocks(std::vector<double> & val_list,std::vector<int> & pos_list)
          {
            // return the full dense block entries of the hmat as
            // flattened lists
            // val_list(i) = H(pos_list(2*i),pos_list(2*i+1));

            IL_EXPECT_FAST(isBuilt_);

            il::int_t numberofunknowns = h_.size(1);
            il::Array2C<il::int_t> pos{0, 2};
            il::Array<double> val{};
            val.Reserve(numberofunknowns * 4);
            pos.Reserve(numberofunknowns * 4, 2);

            output_hmatFullBlocks(this->h_, val, pos);

            std::cout << "done Full Block: nval " << val.size() << " / " << pos.size(0)
                      << " n^2 " << numberofunknowns * numberofunknowns << "\n";

            IL_EXPECT_FAST((val.size()) == (pos.size(0)));
            IL_EXPECT_FAST(pos.size(1) == 2);

            // outputs
            pos_list.resize((pos.size(0)) * 2);
            val_list.resize(pos.size(0));

            int index = 0;
            for (il::int_t i = 0; i < pos.size(0); i++) {
              pos_list[index++] = pos(i, 0);
              pos_list[index++] = pos(i, 1);
            }
            index = 0;
            for (il::int_t i = 0; i < pos.size(0); i++) {
              val_list[index++] = val[i];
            }

          }


          // ---------------------------------------------------------------------------
          std::vector<double> hdotProduct(const std::vector<double>& x, bool UseTractionKernel = true)
          {
              // UseTractionKernel:   it is true if you want to build the HMatrix for the DD to traction kernel
              //                      it is false if you want to build the HMatrix for the DD to displacement kernel
            IL_EXPECT_FAST(this->isBuilt_);
            if (UseTractionKernel) {
                IL_EXPECT_FAST(h_.size(0) == h_.size(1));
                IL_EXPECT_FAST(h_.size(1) == x.size());

                il::Array<double> z{h_.size(1), 0.};

                // permutation of the dofs according to the re-ordering sue to clustering
                il::int_t numberofcollocationpoints = collocationPoints_.size(0);

                for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
                    for (int j = 0; j < dof_dimension_; j++) {
                        z[dof_dimension_ * i + j] = x[dof_dimension_ * (permutation_[i]) + j];
                    }
                }

                z = il::dotwithpattern(h_, fr_pattern_, lr_pattern_, z);
                ////    z = il::dot(h_,z);

                std::vector<double> y;
                y.assign(z.size(), 0.);
                // permut back
                for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
                    for (int j = 0; j < dof_dimension_; j++) {
                        y[dof_dimension_ * (this->permutation_[i]) + j] =
                                z[dof_dimension_ * i + j];
                    }
                }
                return y;
            }
            else {
                IL_EXPECT_FAST(hdispl_.size(0) == hdispl_.size(1));
                IL_EXPECT_FAST(hdispl_.size(1) == x.size());

                il::Array<double> z{hdispl_.size(1), 0.};

                // permutation of the dofs according to the re-ordering sue to clustering
                il::int_t numberofcollocationpoints = collocationPoints_.size(0);

                for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
                    for (int j = 0; j < dof_dimension_; j++) {
                        z[dof_dimension_ * i + j] = x[dof_dimension_ * (permutation_[i]) + j];
                    }
                }

                z = il::dotwithpattern(hdispl_, fr_pattern_hdispl_, lr_pattern_hdispl_, z);
                ////    z = il::dot(h_,z);

                std::vector<double> y;
                y.assign(z.size(), 0.);
                // permut back
                for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
                    for (int j = 0; j < dof_dimension_; j++) {
                        y[dof_dimension_ * (this->permutation_[i]) + j] =
                                z[dof_dimension_ * i + j];
                    }
                }
                return y;
            }

          }
          //---------------------------------------------------------------------------
          std::vector<double> hdotProductInPermutted(const std::vector<double> & x, bool UseTractionKernel = true)
          {
            // UseTractionKernel:   it is true if you want to build the HMatrix for the DD to traction kernel
            //                      it is false if you want to build the HMatrix for the DD to displacement kernel
                IL_EXPECT_FAST(this->isBuilt_);

                if (UseTractionKernel) {
                    IL_EXPECT_FAST(h_.size(0) == h_.size(1));
                    IL_EXPECT_FAST(h_.size(1) == x.size());

                    il::Array<double> z{h_.size(1), 0.};
                    for (il::int_t i = 0; i < h_.size(1); i++) {
                        z[i] = x[i];
                    }

                    z = il::dotwithpattern(h_,fr_pattern_,lr_pattern_,z);
                    std::vector<double> y=x;
                    for (il::int_t i = 0; i < h_.size(1); i++) {
                        y[i] = z[i];
                    }
                    return y;
                }
                else {
                    IL_EXPECT_FAST(hdispl_.size(0) == hdispl_.size(1));
                    IL_EXPECT_FAST(hdispl_.size(1) == x.size());

                    il::Array<double> z{hdispl_.size(1), 0.};
                    for (il::int_t i = 0; i < hdispl_.size(1); i++) {
                        z[i] = x[i];
                    }

                    z = il::dotwithpattern(hdispl_,fr_pattern_hdispl_,lr_pattern_hdispl_,z);
                    std::vector<double> y=x;
                    for (il::int_t i = 0; i < hdispl_.size(1); i++) {
                        y[i] = z[i];
                    }
                    return y;
                }




          }


          //

          //  //---------------------------------------------------------------------------
          //  int h_IterativeSolve(double* y, bool permuted, int maxRestart, il::io_t,
          //                       int& nIts, double& rel_error, double* x0) {
          //    // api function for a GMRES iterative solver h_.x=y
          //    //  IMPORTANT: this GMRES  USE the  dot using  dotwithpattern
          //    //
          //    // inputs:
          //    // double * y : pointer to the RHS
          //    // bool permuted : true or false depending if y is already permutted
          //    // il::int_t maxRestart :: maximum number of restart its of the GMRES
          //    // il::io_t :: il flag - variables after are modifed by this function
          //    // nIts :: max number of allowed its of the GMRES, modified as the
          //    actual nb
          //    // of its
          //    // rel_error:: relative error of the GMRES, modifed as the actual
          //    // norm of current residuals
          //    // double * x0 : pointer to the initial guess,
          //    // modified as the actual solution out int :: 0 if status.Ok(),
          //    // output
          //    // int 0 for success, 1 for failure of gmres
          //    const il::int_t rows = h_.size(0);
          //    const il::int_t columns = h_.size(1);
          //    IL_EXPECT_FAST(columns == rows);
          //    IL_EXPECT_FAST(this->isHbuilt == 1);
          //    //
          //
          //    //
          //    il::Array<double> z{columns, 0.}, xini{columns, 0.};
          //
          //    // permut
          //
          //    // note we duplicate 2 vectors here....
          //    // if x is given as permutted or not
          //    if (!permuted) {
          //      for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
          //        for (int j = 0; j < this->dof_dimension; j++) {
          //          z[dof_dimension * i + j] =
          //              y[dof_dimension * (this->permutation_[i]) + j];
          //          xini[dof_dimension * i + j] =
          //              x0[dof_dimension * (this->permutation_[i]) + j];
          //        }
          //      }
          //    } else {
          //      for (int j = 0; j < columns; j++) {
          //        z[j] = y[j];
          //        xini[j] = x0[j];
          //      }
          //    }
          //
          //    // ensure hpattern is set for hdot
          //    if (isHpattern==0){
          //      setHpattern();
          //    }
          //    // prepare call to GMRES
          //    // Instantiation of Matrix type object for GMRES
          //    bie::Matrix<double> A(this->h_,this->fr_pattern_,this->lr_pattern_);
          //    il::Status status{};
          //
          //    // Jacobi preconditioner only so far
          //    bie::DiagPreconditioner<double> Prec(this->diagonal);
          //
          //    il::Gmres<double> gmres{A, Prec, maxRestart};
          //    // solution of the systme via GMRES
          //    gmres.Solve(z.view(), rel_error, nIts, il::io, xini.Edit(), status);
          //    std::cout << "end of Gmres solve, its # " << gmres.nbIterations()
          //              << " norm of residuals: " << gmres.normResidual() << "\n";
          //    // return norm and nits
          //    rel_error = gmres.normResidual();
          //    nIts = gmres.nbIterations();
          //
          //    // return solution in x0
          //    if (!permuted) {
          //      // permut back
          //      for (il::int_t i = 0; i < numberofcollocationpoints; i++) {
          //        for (int j = 0; j < this->dof_dimension; j++) {
          //          x0[dof_dimension * (this->permutation_[i]) + j] =
          //              xini[dof_dimension * i + j];
          //        }
          //      }
          //    } else {
          //      for (int j = 0; j < columns; j++) {
          //        x0[j] = xini[j];
          //      }
          //    }
          //
          //    if (status.Ok()) {
          //      return 0;
          //    } else {
          //      return 1;
          //    }
          //  }

          // todo: make it work in 2D but in 3D needs to be recoded
          // todo: make the displacement version as well

          //  //---------------------------------------------------------------------------
          //  double* computeStresses(const double* solution, const double* obsPts,
          //                          const int npts) {
          //    // compute stresses at list of points (of size npts )
          //    // from a solution vector.
          //
          //    IL_EXPECT_FAST(this->isMeshbuilt == 1);
          //    // note solution MUST be of length = number of dofs !
          //
          //    IL_EXPECT_FAST((obsPts != (double*)nullptr));
          //    IL_EXPECT_FAST((solution != (double*)nullptr));
          //
          //    il::Array2D<double> pts{npts, dimension_};
          //    il::Array<double> solu{numberofunknowns};
          //    il::Array2D<double> stress;
          //
          //    switch (dimension_) {
          //      case 2: {
          //        il::int_t index = 0;
          //        for (il::int_t i = 0; i < npts; i++) {
          //          pts(i, 0) = obsPts[index++];
          //          pts(i, 1) = obsPts[index++];
          //        }
          //        for (il::int_t i = 0; i < numberofunknowns; i++) {
          //          solu[i] = solution[i];
          //        }
          //        std::cout << " compute stress - " << this->kernel <<"\n";
          //        if (this->kernel == "2DP1") {
          //          stress = bie::computeStresses2D(
          //              pts, the_mesh_2D, elas_, solu, bie::point_stress_s2d_dp1_dd,
          //              frac_height_);
          //        } else if (this->kernel == "S3DP0") {
          //
          //          stress = bie::computeStresses2D(
          //              pts, the_mesh_2D, elas_, solu, bie::point_stress_s3d_dp0_dd,
          //              frac_height_);
          //        }
          //        break;
          //      }
          //      case 3: {
          //        std::cout << "Not implemented\n";
          //        il::abort();
          //      }
          //    }
          //    // transfert back as a pointer to a flattened list....
          //    double* stress_o = new double[stress.size(0) * stress.size(1)];
          //    il::int_t index =0;
          //    for (il::int_t i=0;i<stress.size(0);i++){
          //      stress_o[index++]=stress(i,0);
          //      stress_o[index++]=stress(i,1);
          //      stress_o[index++]=stress(i,2);
          //    }
          //
          //    return stress_o;
          //  }

        };  // end class bigwhamio

//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 22.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2020.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <string>
#include <random>
#include <chrono>

#include <il/Array2D.h>

#include <BigWham.h>

#include <hmat/cluster/cluster.h>
#include <hmat/compression/toHMatrix.h>

#include <hmat/hmatrix/toHPattern.h>
#include <hmat/hmatrix/Hmat.h>

#include <hmat/hmatrix/HMatrix.h>
#include <hmat/hmatrix/HMatrixUtils.h>
#include <hmat/linearAlgebra/blas/hdot.h>
//#include <hmat/linearAlgebra/factorization/luDecomposition.h>
#include <elasticity/2d/ElasticHMatrix2DP0.h>
#include <elasticity/2d/ElasticHMatrix2DP1.h>
#include <elasticity/3d/ElasticHMatrix3DT0.h>
#include <elasticity/PostProcessDDM_2d.h>
#include <elasticity/2d/FullMatrixAssembly2D.h>
#include <elasticity/3d/Elastic3DT0_element.h>
#include <core/ElasticProperties.h>
#include <core/FaceData.cpp>
#include <core/FaceData.h>
#include <_test/elastic3DR0_element_benchmark.h>
#include <elasticity/2d/elliptic_integral.hpp>
#include <elasticity/2d/ElasticAxi3DP0_element.h>
#include <elasticity/2d/ElasticHMatrixAxi3DP0.h>

// TEST 2DP1 ///////////////////////////////////////////////////////////////////////////////
int test2DP1(){

  std::cout << "-------------- test2DP1 ---------------------\n";
  // this routine is for testing different component of the library
  //  here 2DP1 kernel.

// simple 1D mesh
  il::int_t ne=40;

  il::Array2D<double> nodes{ne+1,2,0.};
  il::Array2D<il::int_t> conn{ne,2};
  for (il::int_t i=0;i<ne+1;i++){
    nodes(i,0)=-1.0+2.0*i/ne;
    nodes(i,1)=0.;
  }
 for (il::int_t i=0;i<ne;i++){
   conn(i,0)=i;
   conn(i,1)=i+1;
 }

  bie::Mesh test2(nodes,conn,1);
  il::Array2D<double> coll_points = test2.getCollocationPoints();

  il::int_t leaf_size=32; //test->max_leaf_size;
  il::Timer tt;
  tt.Start();
  const bie::Cluster cluster = bie::cluster(leaf_size, il::io, coll_points);

  const il::Tree<bie::SubHMatrix, 4> hmatrix_tree =
      bie::hmatrixTreeIxI(coll_points, cluster.partition, 0.0);
  tt.Stop();

  std::cout << "Time for cluster construction " << tt.time() <<"\n";
  tt.Reset();
  std::cout << cluster.partition.depth() <<"\n";

  bie::ElasticProperties elas_aux(1.0,0.2);

  bie::HMatrix<double> h_ ;
  const bie::ElasticHMatrix2DP1<double> M{coll_points, cluster.permutation,
                                            test2, elas_aux};

  std::cout << " create h mat " << M.size(0) <<"\n";

  h_= bie::toHMatrix(M, hmatrix_tree, 0.001);
  std::cout << " create h mat ended " << h_.isBuilt() <<"\n";
  std::cout << " compression ratio " << bie::compressionRatio(h_)<<"\n";

  double Ep=1.0/(1.0-0.2*0.2);
  double sig = 1.0;
  double a=1.0;
  double coef = 4.0 * sig / (Ep);
  // at collocation points
  il::Array<double> wsol_coll{coll_points.size(0), 0.};
  for (int i = 0; i < coll_points.size(0); ++i) {
    if (std::abs(coll_points(i,0)) < a) {
      wsol_coll[i] = coef * sqrt(pow(a, 2) - pow(coll_points(i,0), 2));
    }
  }
  //at corresponding nodes - works here due to simple mesh....
  il::Array<double> wsol_nodes{coll_points.size(0), 0.};
  double aux_x=0.; int j=0;
  for (int i = 0; i < conn.size(0); ++i) {

    aux_x=nodes(conn(i,0),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
    aux_x=nodes(conn(i,1),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
  }

  il::Array<double> xx{coll_points.size(0)*2};

  for (il::int_t i=0;i<xx.size()/2;i++){
     xx[2*i]=0.0;
     xx[2*i+1]=wsol_nodes[i];
  }

 il::Array< double> y=il::dot(h_,xx);

  // now testing the Bigwhamio class...
  Bigwhamio testbie;
  //Bigwhamio *test = new Bigwhamio();

  std::vector<double> f_coor;
  f_coor.assign(2*test2.numberOfNodes(),0.);
  for (il::int_t i=0;i<test2.numberOfNodes();i++){
    f_coor[2*i]=test2.coordinates(i,0);
    f_coor[2*i+1]=test2.coordinates(i,1);
  }

  std::vector<int> f_conn;
  f_conn.assign(2*test2.numberOfElts(),0);
  for (il::int_t i=0;i<test2.numberOfElts();i++){
    f_conn[2*i]=test2.connectivity(i,0);
    f_conn[2*i+1]=test2.connectivity(i,1);
  }

  std::vector<double> f_prop;
  f_prop.assign(2,0);
  f_prop[0]=elas_aux.getE();
  f_prop[1]=elas_aux.getNu();

  std::string ker="2DP1";

  std::cout << " now setting things in bigwhamio obj \n";

  testbie.set(f_coor,f_conn,ker,f_prop,leaf_size,0.,0.001);


  std::cout        <<  " C R :"<< testbie.getCompressionRatio() <<"\n";

  std::vector<double> x;
  x.assign(xx.size(),0.);
  for (il::int_t i=0;i<xx.size();i++){
    x[i]=xx[i];
  }

  std::vector<double> y2=testbie.hdotProduct(x);
  std::cout << " elastic dot product solution:: \n";
  for (il::int_t i=0;i<coll_points.size(0);i++){
    std::cout <<" y " << i  <<" " << y[2*i] << " - with obj - " << y2[2*i]  <<"\n";
    std::cout <<" y " << i  <<" " << y[2*i+1] << " - with obj - " << y2[2*i+1] <<"\n";
  }

  std::cout << "-------------- End of 2DP1 - tests ------------- " << "\n";

  return  0;

}
///////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////
int testS3DP0(){
// testing the S3DP0 kernel implementation on griffith crack
  std::cout << "-------------- testS3DP0 ---------------------\n";

  std::cout << " testing the S3DP0 kernel implementation on griffith crack" << "\n";
// simple 1D mesh
  il::int_t ne=140;

  il::Array2D<double> nodes{ne+1,2,0.};
  il::Array2D<il::int_t> conn{ne,2};

  for (il::int_t i=0;i<ne+1;i++){
    nodes(i,0)=-1.0+2.0*i/ne;
    nodes(i,1)=0.;
  }

  for (il::int_t i=0;i<ne;i++){
    conn(i,0)=i;
    conn(i,1)=i+1;
  }
  bie::Mesh Mesh0(nodes,conn,0);

  il::Array2D<double> coll_points = Mesh0.getCollocationPoints();

  il::int_t leaf_size=12;
  il::Timer tt;
  tt.Start();
  const bie::Cluster cluster = bie::cluster(leaf_size, il::io, coll_points);

  const il::Tree<bie::SubHMatrix, 4> hmatrix_tree =
      bie::hmatrixTreeIxI(coll_points, cluster.partition, 10.0);
  tt.Stop();
  std::cout << "Time for cluster construction " << tt.time() <<"\n";
  tt.Reset();
  std::cout << cluster.partition.depth() <<"\n";

  bie::ElasticProperties elas_aux(1.0,0.2);

  bie::HMatrix<double> h_ ;
  const bie::ElasticHMatrix2DP0<double> M{coll_points, cluster.permutation,
                                          Mesh0, elas_aux, 10000.};

//  std::cout << " in set h mat 2 " << cluster.permutation.size() << " e aca " << test->epsilon_aca <<"\n";

  std::cout << " create h mat " << M.size(0) <<"\n";
  h_= bie::toHMatrix(M, hmatrix_tree, 0.001);
  std::cout << " create h mat ended " << h_.isBuilt() <<"\n";
  std::cout << " compression ratio " << bie::compressionRatio(h_)<<"\n";

  double Ep=1.0/(1.0-0.2*0.2);
  double sig = 1.0;
  double a=1.0;

  double coef = 4.0 * sig / (Ep);

  // compute analyticla solution at collocation points
  il::Array<double> wsol_coll{coll_points.size(0), 0.};

  for (int i = 0; i < coll_points.size(0); ++i) {
    if (std::abs(coll_points(i,0)) < a) {
      wsol_coll[i] = coef * sqrt(pow(a, 2) - pow(coll_points(i,0), 2));
    }
  }
  //at corresponding nodes - works here due to simple mesh....

  il::Array<double> xx{coll_points.size(0)*2};

  for (il::int_t i=0;i<xx.size()/2;i++){
    xx[2*i]=0.0;
    xx[2*i+1]=wsol_coll[i];
  }

  il::Array< double> y=il::dot(h_,xx);

  std::cout << " Now creating similar BigWham instance \n";
  std::string kernelname = "S3DP0";

  // now testing the Bigwhamio class...
  Bigwhamio testbie;
  //Bigwhamio *test = new Bigwhamio();


  std::vector<double> f_coor;
  f_coor.assign(2*Mesh0.numberOfNodes(),0.);
  for (il::int_t i=0;i<Mesh0.numberOfNodes();i++){
    f_coor[2*i]=Mesh0.coordinates(i,0);
    f_coor[2*i+1]=Mesh0.coordinates(i,1);
  }

  std::vector<int> f_conn;
  f_conn.assign(2*Mesh0.numberOfElts(),0);
  for (il::int_t i=0;i<Mesh0.numberOfElts();i++){
    f_conn[2*i]=Mesh0.connectivity(i,0);
    f_conn[2*i+1]=Mesh0.connectivity(i,1);
  }

  std::vector<double> f_prop;
  f_prop.assign(3,0);
  f_prop[0]=elas_aux.getE();
  f_prop[1]=elas_aux.getNu();
  f_prop[2]=1000.;

  std::cout << " now setting things in bigwhamio obj \n";

  testbie.set(f_coor,f_conn,kernelname,f_prop,22,0.4,0.001);

  std::cout        <<  " C R :"<< testbie.getCompressionRatio() <<"\n";

  std::vector<double> x;
  x.assign(xx.size(),0.);
  for (il::int_t i=0;i<xx.size();i++){
    x[i]=xx[i];
  }

  std::vector<double> y2=testbie.hdotProduct(x);
  std::cout << " elastic dot product solution:: \n";
  for (il::int_t i=0;i<coll_points.size(0);i++){
    std::cout <<" y " << i  <<" " << y[2*i] << " - with obj - " << y2[2*i]  <<"\n";
    std::cout <<" y " << i  <<" " << y[2*i+1] << " - with obj - " << y2[2*i+1] <<"\n";
  }


  std::cout << " build ?" << testbie.isBuilt()
            <<  " CR :"<< testbie.getCompressionRatio() << " CR   " <<"\n";

  il::Array2D<il::int_t> pat_SPOT = bie::output_hmatPattern(h_);

  std::cout << "n blocks "  << " / " << pat_SPOT.size(1) << "\n";

  int k=0;
  for (il::int_t i=0;i<pat_SPOT.size(1);i++){
    il::spot_t s(pat_SPOT(0,i));
    k=2;
    if (h_.isFullRank(s)) {
      k=0;
      il::Array2DView<double > A =h_.asFullRank(s);
       std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," << pat_SPOT(2,i) <<
       "/ " << pat_SPOT(1,i)+A.size(0)-1 << "," <<pat_SPOT(2,i)+A.size(1)-1 <<
       " -   "  << k << " - "  << A.size(0)*A.size(1) << "\n";
    } else if(h_.isLowRank(s)) {
      k=1;
      il::Array2DView<double > A = h_.asLowRankA(s);
      il::Array2DView<double> B = h_.asLowRankB(s);
      std::cout << "block :" << i  << " | " << pat_SPOT(1,i) << "," << pat_SPOT(2,i) <<
                "/ " << pat_SPOT(1,i)+A.size(0)-1 << "," <<pat_SPOT(2,i)+B.size(0)-1
                << " - "  <<  k <<  " - "  << A.size(0)*A.size(1)+B.size(0)*B.size(1) << "\n";
    }
      std::cout <<" block " << i << " :: " << pat_SPOT(0,i) << " - " << h_.isHierarchical(s) <<"\n";
  }

  std::vector<double> val_list;
  std::vector<int> pos_list;

  testbie.getFullBlocks(val_list, pos_list);

  std::cout << "n full block entry" << val_list.size() <<"\n";
  std::cout << " hmat size " << testbie.matrixSize(0) <<"\n";

  il::Array2D<double> mypts{3,2,0.};
  mypts(0,0)=1.5;
  mypts(1,0)=2.;
  mypts(2,0)=2.5;

  il::Array2D<double> stress=bie::computeStresses2D(mypts,Mesh0,elas_aux,xx,bie::point_stress_s3d_dp0_dd,1000.);
  for (il::int_t i=0;i<3;i++){
    std::cout << "stresses " << stress(i,0) << " - " << stress(i,1) <<  " - " << stress(i,2) <<"\n";

  }
  std::cout << "----------end of S3DP0 test ---------------------\n";

  return 0;
}
///////////////////////////////////////////////////////////////////////////////


// TEST FULL MAT /////////////////////////////////////////////////////////////////////////////
int testFullMat()
{
  std::cout << "--------------- testFullMat ---------------------\n";
// simple 1D mesh
  il::int_t ne=100;

  il::Array2D<double> nodes{ne+1,2,0.};
  il::Array2D<il::int_t> conn{ne,2};
  for (il::int_t i=0;i<ne+1;i++){
    nodes(i,0)=-1.0+2.0*i/ne;
    nodes(i,1)=0.;
  }
  for (il::int_t i=0;i<ne;i++){
    conn(i,0)=i;
    conn(i,1)=i+1;
  }

  bie::Mesh mesh(nodes,conn,1);

  bie::ElasticProperties elas(1.0,0.2);

  il::Timer tt;
  tt.Start();
  std::cout << "start of assembly \n " ;
  il::Array2D<double> Mfull = serialFullMatrix2d(mesh,elas,bie::normal_shear_stress_kernel_dp1_dd_nodal,0.);
  tt.Stop();
  std::cout << "time of assembly " << tt.time() <<"\n";
  tt.Reset();

  il::Array<il::int_t> permut{mesh.numberCollocationPoints(),0};
  for (il::int_t i=0;i<mesh.numberCollocationPoints();i++){
    permut[i]=i;
  }
  tt.Start();
  il::Array2D<double> Mfull2 =
      parallelFullMatrix2d(mesh, elas, permut, 0.,
                           bie::normal_shear_stress_kernel_dp1_dd_nodal);
  tt.Stop();
  std::cout << "time of parallel assembly  " << tt.time();

  /// dot-product of linear system and checks
  double Ep=1.0/(1.0-0.2*0.2);
  double sig = 1.0;
  double a=1.0;
  double coef = 4.0 * sig / (Ep);
  // at collocation points
  il::Array<double> wsol_coll{mesh.numberCollocationPoints(), 0.};
  auto coll_points=mesh.getCollocationPoints();
  for (int i = 0; i < coll_points.size(0); ++i) {
    if (std::abs(coll_points(i,0)) < a) {
      wsol_coll[i] = coef * sqrt(pow(a, 2) - pow(coll_points(i,0), 2));
    }
  }
  //at corresponding nodes - works here due to simple mesh....
  il::Array<double> wsol_nodes{coll_points.size(0), 0.};
  double aux_x=0.; int j=0;
  for (int i = 0; i < conn.size(0); ++i) {

    aux_x=nodes(conn(i,0),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
    aux_x=nodes(conn(i,1),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
  }

  il::Array<double> xx{coll_points.size(0)*2};

  for (il::int_t i=0;i<xx.size()/2;i++){
    xx[2*i]=0.0;
    xx[2*i+1]=wsol_nodes[i];
  }

  il::Array< double> y1=il::dot(Mfull,xx);
  il::Array< double> y2=il::dot(Mfull2,xx);

  for (il::int_t i=0;i<coll_points.size(0);i++){
    std::cout <<" y (1) " << i  <<" / " << y1[2*i] <<" y (2) " << i  << " /  " << y2[2*i] <<"\n";
    std::cout <<" y (1) " << i  <<" / " << y1[2*i+1] <<" y (2) " << i  <<" / " << y2[2*i+1] <<"\n";
  }

  std::cout << "----------end of testFullMat ---------------------\n";

  return 0;

}
///////////////////////////////////////////////////////////////////////////////


// TEST H DOT /////////////////////////////////////////////////////////////////////////////
int testHdot() {
  std::cout << "--------------- testHdot ---------------------\n";

  // star cracks mesh - crack length unity
  il::int_t nfracs=8;
  il::int_t ne_per_frac=100;
  il::Array<double> rad{ne_per_frac+1,0.};
  il::Array<double> angle{nfracs,0.};
//
  for (il::int_t i=0;i<=ne_per_frac;i++){
    rad[i]=1.*i/ne_per_frac;
  }
  for (il::int_t i=0;i<nfracs;i++){
    angle[i]=2*(il::pi)*i/nfracs;
  }
//
  il::int_t ne=ne_per_frac*nfracs;
  il::Array2D<double> nodes{ne+1,2,0.};
  il::Array2D<il::int_t> conn{ne,2};
  il::int_t index=0;
   nodes(index,0)=0.;
   nodes(index,1)=0.;
  for (il::int_t i=0;i<nfracs;i++){
    for (il::int_t j=1;j<ne_per_frac+1;j++){
      index++;
      nodes(index,0)=rad[j]*cos(angle[i]);
      nodes(index,1)=rad[j]*sin(angle[i]);
    }
  }

  for (il::int_t i=0;i<ne;i++){
    conn(i,0)=i;
    conn(i,1)=i+1;
  }
  for (il::int_t j=0;j<nfracs;j++){
    // first node of each fract is the first nod of the mesh.
    conn(j*ne_per_frac,0)=0;
  }

  std::cout << " N unknowns: " << 4*ne <<"\n";
  bie::Mesh mesh(nodes,conn,1);

  bie::ElasticProperties elas(1.0,0.2);
  il::Array2D<double> coll_points = mesh.getCollocationPoints();

  il::int_t leaf_size=32;
  il::Timer tt;
  tt.Start();
  const bie::Cluster cluster = bie::cluster(leaf_size, il::io, coll_points);

  tt.Stop();
  std::cout << "Time for  cluster tree construction " << tt.time() <<"\n";
  std::cout << " cluster depth ..." <<  cluster.partition.depth() <<"\n";
  std::cout << " cluster - part " << cluster.permutation.size() << "\n";

  tt.Reset();
//  std::cout << " press enter to continue ...\n";
//  std::cin.ignore(); // pause while user do not enter return

  tt.Start();
  const il::Tree<bie::SubHMatrix, 4> hmatrix_tree =
      bie::hmatrixTreeIxI(coll_points, cluster.partition, 3.0);
  tt.Stop();
  std::cout << "Time for binary cluster tree construction " << tt.time() <<"\n";
  std::cout << " binary cluster depth ..." << hmatrix_tree.depth() << "\n";
  std::cout << " root - " << hmatrix_tree.root().index << "\n";
  tt.Reset();

  il::int_t nb=bie::nbBlocks(hmatrix_tree);
  std::cout << " Number of sub-matrix blocks: "  << nb <<  " \n";
  //std::cin.ignore(); // pause while user do not enter return
  il::int_t n_fullb=bie::nbFullBlocks(hmatrix_tree);
  std::cout << " Number of sub-matrix full blocks: "  << n_fullb <<  " \n";

  tt.Start();
  bie::HPattern my_patt=bie::createPattern(hmatrix_tree);
  std::cout << "Time for pattern construction " << tt.time() <<"\n";
  std::cout << " Number of sub-matrix full blocks: "  << my_patt.n_FRB <<  " \n";
  std::cout  << " n fb " <<  my_patt.FRB_pattern.size(1) <<"\n";
  std::cout  << " n lrb " <<  my_patt.LRB_pattern.size(1) <<"\n";
  for (il::int_t i=0;i<5;i++) {
    std::cout << " FRB : " << my_patt.FRB_pattern(0, i) << "-"
              << my_patt.FRB_pattern(1, i) << "-" << my_patt.FRB_pattern(5, i)
              << " \n";
    std::cout << " LRB : " << my_patt.LRB_pattern(0, i) << "-"
              << my_patt.LRB_pattern(1, i) << "-" << my_patt.LRB_pattern(5, i)
              << " \n";
  }
  tt.Reset();

  bie::HMatrix<double> h_ ;
  const bie::ElasticHMatrix2DP1<double> M{coll_points, cluster.permutation,
                                          mesh, elas};
  std::cout << " create h mat - size " << M.size(0) << " * " << M.size(1) <<"\n";
  tt.Start();
  h_= bie::toHMatrix(M, hmatrix_tree, 0.0001);
  tt.Stop();

  std::cout << " create h mat ended " << h_.isBuilt() <<  " in " << tt.time() <<  "\n";
  std::cout << " compression ratio " << bie::compressionRatio(h_)<<"\n";
  std::cout << " Compressed memory " << (bie::compressionRatio(h_)*8*(4*ne*4*ne)) <<"\n";
  std::cout << " dense case memory " << 8*(4*ne*4*ne) << "\n";
  std::cout <<  "number of blocks " << bie::numberofBlocks(h_) << "\n";
  std::cout << "number of full blocks " << bie::numberofFullBlocks(h_) <<"\n";
 // std::cout << " press enter to continue ...\n";
  //std::cin.ignore(); // pause while user do not enter return
  tt.Reset();
  tt.Start();
  il::Array2D<il::int_t> pattern=output_hmatPattern(h_);
  tt.Stop();
  std::cout << " time for getting pattern "  << tt.time() <<  " number of blocks "  << pattern.size(1) << "\n";
  std::cout << " first block "  << pattern(0,0) << "-" << pattern(2,0) << "\n";

  tt.Reset();
  /// dot-product of linear system and checks
  double Ep=1.0/(1.0-0.2*0.2);
  double sig = 1.0;
  double a=1.0;
  double coef = 4.0 * sig / (Ep);
  // at collocation points
  il::Array<double> wsol_coll{mesh.numberCollocationPoints(), 0.};
  for (int i = 0; i < coll_points.size(0); ++i) {
    if (std::abs(coll_points(i,0)) < a) {
      wsol_coll[i] = coef * sqrt(pow(a, 2) - pow(coll_points(i,0), 2));
    }
  }
  //at corresponding nodes - works here due to simple mesh....
  il::Array<double> wsol_nodes{coll_points.size(0), 0.};
  double aux_x=0.; int j=0;
  for (int i = 0; i < conn.size(0); ++i) {
    aux_x=nodes(conn(i,0),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
    aux_x=nodes(conn(i,1),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
  }

  il::Array<double> xx{coll_points.size(0)*2};

  for (il::int_t i=0;i<xx.size()/2;i++){
    xx[2*i]=0.0;
    xx[2*i+1]=wsol_nodes[i];
  }

  tt.Start();
  il::Array< double> y1=il::dot(h_,xx);
  tt.Stop();
  std::cout << " time for recursive hdot " << tt.time() <<"\n";
  tt.Reset();
  tt.Start();
  il::Array< double> y2= il::dotwithpattern_serial(h_, pattern, xx);
  tt.Stop();
  std::cout << " time for Non-recursive hdot serial " << tt.time() <<"\n";

  il::Array2D<il::int_t> lr_patt{pattern.size(0),0};
  il::Array2D<il::int_t> fr_patt{pattern.size(0),0};
  lr_patt.Reserve(3,pattern.size(1));
  fr_patt.Reserve(3,pattern.size(1));
  il::int_t nfb=0;il::int_t nlb=0;
  for (il::int_t i=0;i<pattern.size(1);i++){
    il::spot_t s(pattern(0,i));
    if (h_.isFullRank(s)){

      fr_patt.Resize(3,nfb+1);
      fr_patt(0,nfb)=pattern(0,i);
      fr_patt(1,nfb)=pattern(1,i);
      fr_patt(2,nfb)=pattern(2,i);
      nfb++;
    } else if (h_.isLowRank(s)){

      lr_patt.Resize(3,nlb+1);
      lr_patt(0,nlb)=pattern(0,i);
      lr_patt(1,nlb)=pattern(1,i);
      lr_patt(2,nlb)=pattern(2,i);
      nlb++;
    } else {
      std::cout <<"error in pattern !\n" ;
      il::abort();
    }
  }

  tt.Reset();
  tt.Start();

  il::Array< double> y3= il::dotwithpattern(h_, fr_patt, lr_patt, xx);
  tt.Stop();
  std::cout << " time for Non-recursive hdot (parallel_invoke) " << tt.time() <<"\n";
  std::cout << " norm y1 " << il::norm(y1,il::Norm::L2) << " y2 " << il::norm(y2,il::Norm::L2)
  <<" y3 " << il::norm(y3,il::Norm::L2) << "\n";

  std::cout << " n FR blocks " << fr_patt.size(1) << "  n LR block " << lr_patt.size(1) <<"\n";

  std::cout << "----------end of test hdot  ---------------------\n";

  return 0;

}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DR0 /////////////////////////////////////////////////////////////////////////////
int test3DR0() {
    std::cout << "-------------- test3DR0 ---------------------\n";

    // coordinates
    const std::vector<double> coor={-1.,-1.,0.,
                                    1.,-1.,0.,
                                    1.,1.,0.,
                                    -1.,1.,0.,
                                    -1.,2.,0.,
                                    1.,2.,0.};
    // connectivity
    const std::vector<int> conn = {0,1,2,3,
                                       3,2,5,4};

    const std::vector<double> properties = {100, 0.2}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 1000;
    const double eta = 0.;
    const double eps_aca = 0.001;

    // create displacement HMAT
    const std::string displacementKernel = "3DR0_displ";
    Bigwhamio displacementHMAT;
    displacementHMAT.set(coor,conn,displacementKernel,properties,
                         max_leaf_size, eta, eps_aca);


    const std::string tractionKernel = "3DR0";
    Bigwhamio tractionHMAT;
    tractionHMAT.set(coor,conn,tractionKernel,properties,
                     max_leaf_size, eta, eps_aca);

    // use the Hdot product
    const std::vector<double>  xx = {1.,2.,3.,4.,5.,6.};
    std::vector<double> x;
    std::cout << "Traction HMAT dot product \n" ;
    x = tractionHMAT.hdotProduct(xx);
    for(int i=0; i<x.size(); ++i) std::cout << x[i] << " ";
    std::cout << "\n" ;
    std::cout << "Displacement HMAT dot product \n" ;
    x = displacementHMAT.hdotProduct(xx);
    for(int i=0; i<x.size(); ++i) std::cout << x[i] << ' ';

    // compute stresses at a set of observation points
    std::vector<double> coorobsp={-10.,-10.,0.,
                                   20.,-20.,0.};
    std::vector<double> mysol = {1.,1.,1.,1.,1.,1.};
    std::vector<double> bb = tractionHMAT.computeStresses(mysol, coorobsp, 2, properties, coor, conn, true);

    for(int i=0; i<bb.size()/6; ++i) {
        std::cout << "\n stress at point #" << i << "\n ";
        for(int j=0; j<6; ++j){
            std::cout << bb[i*6+j] << " ";
        }
    }

    // compute displacements at a set of observation points

    std::vector<double> cc = displacementHMAT.computeDisplacements(mysol, coorobsp, 2, properties, coor, conn, true);

    for(int i=0; i<cc.size()/3; ++i) {
        std::cout << "\n displacement at point #" << i << "\n ";
        for(int j=0; j<3; ++j){
            std::cout << cc[i*3+j] << " ";
        }
    }

    std::cout << "\n----------end of test 3DR0  ---------------------\n";
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DR0_MODE2AND3/////////////////////////////////////////////////////////////////////////////
int test3DR0modes2and3() {
    std::cout << "-------------- test3DR0modes2and3 ---------------------\n";
    // this function tests if the implementation of the reduced 3DR0 element works

    // coordinates
    const std::vector<double> coor={-1.,-1.,0.,
                                    1.,-1.,0.,
                                    1.,1.,0.,
                                    -1.,1.,0.,
                                    -1.,2.,0.,
                                    1.,2.,0.};
    // connectivity
    const std::vector<int> conn = {0,1,2,3,
                                   3,2,5,4};

    const std::vector<double> properties = {100, 0.2}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 1000;
    const double eta = 0.;
    const double eps_aca = 0.001;



    const std::string tractionKernel = "3DR0";
    Bigwhamio tractionHMAT;
    tractionHMAT.set(coor,conn,tractionKernel,properties,
                     max_leaf_size, eta, eps_aca);

    const std::string tractionKernel_shearONLY = "3DR0shear";
    Bigwhamio tractionHMAT_shearONLY;
    tractionHMAT_shearONLY.set(coor,conn,tractionKernel_shearONLY,properties,max_leaf_size, eta, eps_aca);

    // use the Hdot product
    const std::vector<double>  xx = {1.,2.,0.,4.,5.,0.};
    std::vector<double> x;
    std::cout << "Traction HMAT dot product \n" ;
    x = tractionHMAT.hdotProduct(xx);
    for(int i=0; i<x.size(); ++i) std::cout << x[i] << " ";
    std::cout << "\n" ;

    const std::vector<double>  xxRed = {1.,2.,4.,5.};
    std::vector<double> xRed;
    std::cout << "Traction HMAT dot product \n" ;
    xRed = tractionHMAT_shearONLY.hdotProduct(xxRed);
    for(int i=0; i<xRed.size(); ++i) std::cout << xRed[i] << " ";
    std::cout << "\n" ;


    std::cout << "\n----------end of test 3DR0modes2and3  ---------------------\n";
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////

// TEST 3DR0_MODE2AND3/////////////////////////////////////////////////////////////////////////////
int test3DT0modes2and3() {
    std::cout << "-------------- test3DT0modes2and3 ---------------------\n";
    // this function tests if the implementation of the reduced 3DR0 element works

    // coordinates
    const std::vector<double> coor={-1.,-1.,0.,
                                    1.,-1.,0.,
                                    1.,1.,0.,
                                    -1.,1.,0.,
                                    -1.,2.,0.,
                                    1.,2.,0.};
    // connectivity
    const std::vector<int> conn = {0,1,3,1,2,3,
                                   3,2,4,2,5,4};

    const std::vector<double> properties = {100, 0.2}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 1000;
    const double eta = 0.;
    const double eps_aca = 0.001;



    const std::string tractionKernel = "3DT0";
    Bigwhamio tractionHMAT;
    tractionHMAT.set(coor,conn,tractionKernel,properties,
                     max_leaf_size, eta, eps_aca);

    const std::string tractionKernel_shearONLY = "3DT0shear";
    Bigwhamio tractionHMAT_shearONLY;
    tractionHMAT_shearONLY.set(coor,conn,tractionKernel_shearONLY,properties,max_leaf_size, eta, eps_aca);

    // use the Hdot product
    const std::vector<double>  xx = {1.,2.,0.,4.,5.,0.,6.,7.,0.,8.,9.,0.};
    std::vector<double> x;
    std::cout << "Traction HMAT dot product \n" ;
    x = tractionHMAT.hdotProduct(xx);
    for(int i=0; i<x.size(); ++i) std::cout << x[i] << " ";
    std::cout << "\n" ;

    const std::vector<double>  xxRed = {1.,2.,4.,5.,6.,7.,8.,9.};
    std::vector<double> xRed;
    std::cout << "Traction HMAT dot product \n" ;
    xRed = tractionHMAT_shearONLY.hdotProduct(xxRed);
    for(int i=0; i<xRed.size(); ++i) std::cout << xRed[i] << " ";
    std::cout << "\n" ;


    std::cout << "\n----------end of test 3DT0modes2and3  ---------------------\n";
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////

// TEST PERF 3DR0 ///////////////////////////////////////////////////////////////////////////////
int perf3DR0() {
    std::cout << "-------------- perf3DR0 ---------------------\n";

    Box_mesh_1 mymesh;

    const std::vector<double> properties = {100, 0.2}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 100;
    const double eta = 0.;
    const double eps_aca = 0.0001;

    auto t1 = std::chrono::high_resolution_clock::now();
    // create traction HMAT
    const std::string tractionKernel = "3DR0_displ";
    Bigwhamio tractionHMAT;
    tractionHMAT.set(mymesh.coor,mymesh.conn,tractionKernel,properties,
                     max_leaf_size, eta, eps_aca);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 -t1).count();
    std::cout<<duration;


    // create displacement HMAT
    const std::string displacementKernel = "3DR0_displ";
    Bigwhamio displacementHMAT;
    displacementHMAT.set(mymesh.coor,mymesh.conn,displacementKernel,properties,
                         max_leaf_size, eta, eps_aca);



    std::cout << "\n----------end of test 3DR0 perf ---------------------\n";
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DT0 ///////////////////////////////////////////////////////////////////////////////
int test3DT0() {

    std::cout << "--------------- test3DT0 ---------------------\n";

    il::Array2D<double> x{1,3,0.};
//    x[0] = 1.0;
//    x[1] = 0.0;
//    x[2] = 0.3;
    x(0,0) = 0.3;
    x(0,1) = 0.3;
    x(0,2) = 0.7;
    il::Array2D<double> xv{3,3,0.};
    xv(0,0) = 0.0;
    xv(0,1) = 0.0;
    xv(0,2) = 0.0;
    xv(1,0) = 1.0;
    xv(1,1) = 0.0;
    xv(1,2) = 0.0;
    xv(2,0) = 0.0;
    xv(2,1) = 1.0;
    xv(2,2) = 0.0;

    // get triangle vertices coordinates separated
    il::StaticArray<double, 3> y1, y2, y3;
    for (il::int_t i = 0; i < 3; i++) {
        y1[i] = xv(0, i);
        y2[i] = xv(1, i);
        y3[i] = xv(2, i);
    }
    // subtractions of coordinates vectors
    il::StaticArray<double, 3> y31, y21, y1x;
    for (il::int_t i = 0; i < 3; i++) {
        y31[i] = y3[i] - y1[i];
        y21[i] = y2[i] - y1[i];
        y1x[i] = y1[i] - x(0,i);
    }

    // local reference system (e1,e2,e3)
    il::StaticArray<double, 3> e1, e2, e3;

    // e1
    e1[0] = y21[0] / il::norm(y21, il::Norm::L2);
    e1[1] = y21[1] / il::norm(y21, il::Norm::L2);
    e1[2] = y21[2] / il::norm(y21, il::Norm::L2);

    // e2
    double sigma = il::dot(y31,y21) / pow(il::norm(y21, il::Norm::L2),2.0);
    il::StaticArray<double, 3> e2vector; // not normalized
    e2vector[0] = y31[0] - sigma * y21[0];
    e2vector[1] = y31[1] - sigma * y21[1];
    e2vector[2] = y31[2] - sigma * y21[2];
    e2[0] = e2vector[0] / il::norm(e2vector, il::Norm::L2);
    e2[1] = e2vector[1] / il::norm(e2vector, il::Norm::L2);
    e2[2] = e2vector[2] / il::norm(e2vector, il::Norm::L2);

    // e3
    e3 = il::cross(e1,e2);

    std::cout << " e1: " << e1[0] << " " << e1[1] << " " << e1[2] <<"\n";
    std::cout << " e2: " << e2[0] << " " << e2[1] << " " << e2[2] <<"\n";
    std::cout << " e3: " << e3[0] << " " << e3[1] << " " << e3[2] <<"\n";

    // geometrical parameters
    double a, b, c;
    a = il::dot(y31,e2);
    b = il::dot(y21,e1);
    c = il::dot(y31,e1);
    double theta1, theta2, theta3;
    theta1 = acos(c / sqrt( pow(c,2.0) + pow(a,2.0) ) );
    theta2 = acos((b - c) / sqrt( pow(b-c,2.0) + pow(a,2.0) ) );
    theta3 = il::pi - (theta1 + theta2);

    std::cout << " a, b, c: " << a << " " << b << " " << c <<"\n";
    std::cout << " theta1, theta2, theta3: " << theta1 << " " << theta2 << " " << theta3 <<"\n";

    // endpoints' coordinates of each triangle edge for the local coordinate systems of each edge

    // vertex 1 coordinates
    double xi1, zeta1, eta;
    xi1 = il::dot(y1x,e1);
    zeta1 = il::dot(y1x,e2);
    eta = il::dot(y1x,e3);

    // auxiliary angles
    double alpha1, alpha2, alpha3;
    alpha1 = 0.;
    alpha2 = il::pi - theta2;
    alpha3 = il::pi + theta1;

    // endpoints' coordinates edge L1
    double p11, p12, q1;
    p11 = xi1;
    p12 = b + xi1;
    q1 = zeta1;

    // endpoints' coordinates edge L2
    double p22, p23, q2;
    p22 = (b + xi1) * cos(alpha2) + zeta1 * sin(alpha2);
    p23 = (c + xi1) * cos(alpha2) + (a + zeta1) * sin(alpha2);
    q2 = -(b + xi1) * sin(alpha2) + zeta1 * cos(alpha2);

    // endpoints' coordinates edge L3
    double p33, p31, q3;
    p33 = (c + xi1) * cos(alpha3) + (a + zeta1) * sin(alpha3);
    p31 = xi1 * cos(alpha3) + zeta1 * sin(alpha3);
    q3 = -xi1 * sin(alpha3) + zeta1 * cos(alpha3);

    // generic recursive functions

    // previous definitions

    // p
    il::StaticArray2D<double, 3, 3> p;
    p(0,0) = p11;
    p(1,1) = p22;
    p(2,2) = p33;
    p(0,1) = p12;
    p(1,2) = p23;
    p(2,0) = p31;

    // q
    il::StaticArray<double, 3> q;
    q[0] = q1;
    q[1] = q2;
    q[2] = q3;

    std::cout << " p11, p12, q1: " << p(0,0) << " " << p(0,1) << " " << q[0] <<"\n";
    std::cout << " p22, p23, q2: " << p(1,1) << " " << p(1,2) << " " << q[1] <<"\n";
    std::cout << " p33, p31, q3: " << p(2,2) << " " << p(2,0) << " " << q[2] <<"\n";

    // rho - distance between source/receiver point 'x' and i-th vertex of triangle
    il::StaticArray<double, 3> rho;
    for (int i = 0; i < 3; i++) {
        rho[i] = sqrt( pow(p(i,i),2.0) + pow(q[i],2.0) + pow(eta,2.0) );
    }

    std::cout << " rho: " << rho[0] << " " << rho[1] << " " << rho[2] <<"\n";

    // definition of cases - where the projected source/receiver point lies
    // also, computation of theta_0

    // rho_plane - distance between projected source/receiver point 'x' and i-th vertex of triangle
    il::StaticArray<double, 3> rho_plane;
    for (int i = 0; i < 3; i++) {
        rho_plane[i] = sqrt( pow(p(i,i),2.0) + pow(q[i],2.0) );
    }

    std::cout << " rho_plane: " << rho_plane[0] << " " << rho_plane[1] << " " << rho_plane[2] <<"\n";

    double eps_tol = 2.221e-016; // 1.0e-015 parameter used for "if conditions" involving inequalities due to numerical precision

    int id;
    double theta_0;
    if( q[0] > eps_tol || q[1] > eps_tol || q[2] > eps_tol ){
        // if projected point lies strictly outside of the triangle
        id = -1;
        theta_0 = 0.0;
    } else if( q[0] < -eps_tol && q[1] < -eps_tol && q[2] < -eps_tol ){
        // if projected point lies strictly inside of the triangle
        id = 0;
        theta_0 = 2*il::pi;
    } else if( rho_plane[0] > eps_tol && rho_plane[1] > eps_tol && rho_plane[2] > eps_tol){
        // if projected point lies on an edge of the triangle but not on a vertex
        id = 4;
        theta_0 = il::pi;
    } else if( rho_plane[0] < eps_tol ){
        // if projected point lies exactly on vertex 1
        id = 1;
        theta_0 = theta1;
    } else if( rho_plane[1] < eps_tol ){
        // if projected point lies exactly on vertex 2
        id = 2;
        theta_0 = theta2;
    } else if( rho_plane[2] < eps_tol ){
        // if projected point lies exactly on vertex 3
        id = 3;
        theta_0 = theta3;
    }

    std::cout << " id: " << id << "\n";
    std::cout << " theta_0 for unit triangle: " << theta_0 << "\n";
    double teta2 = atan(1.0);
    double teta1 = il::pi/2;
    double teta3 = il::pi-teta1-teta2;

    std::cout << " tetas: " << teta1 << " " << teta2 << " " << teta3 <<"\n";

    // check if summation algorithm is working

    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum += rho_plane[i];
    }

    std::cout << " sum: " << sum << "\n";

    // now the generic recursive functions

    int i1; // i+1, will be used in all recursive generic functions

    // d
    il::StaticArray<double, 3> d;
    for (int i = 0; i < 3; i++) {
        d[i] = pow(q[i],2.0) + pow(eta, 2.0);
    }

    // rho tilde
    il::StaticArray<double, 3> rho_t;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        rho_t[i] = rho[i] - rho[i1];
    }

    // phi
    il::StaticArray<double, 3> phi;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        phi[i] = p(i, i) * rho[i] - p(i, i1) * rho[i1];
    }

    // chi
    // Note: in Nintcheu Fata (2009, 2011) there are no comments for the case
    // in which the log arguments are negative or zero and this can happen! Brice
    // regularized this in his mathematica notebook but I don't know yet how (Alexis).
    // I use here Brice's solution
    il::StaticArray<double, 3> chi;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        if ((p(i, i) + rho[i]) > eps_tol && (p(i, i1) + rho[i1]) > eps_tol) {
            // if the log arguments are strictly positive
            chi[i] = log(p(i, i) + rho[i]) - log(p(i, i1) + rho[i1]);
        } else if ((p(i, i) + rho[i]) < -eps_tol && (p(i, i1) + rho[i1]) < -eps_tol) {
            // if the log arguments are strictly negative
            double d = pow(q[i], 2.0) + pow(eta, 2.0);
            double p1 = p(i, i) + rho[i];
            double p2 = p(i, i1) + rho[i1];
            double z1 = (d / p1) / p1;
            double z2 = (d / p2) / p2;
            chi[i] = log(
                    abs(p2 / p1) * ((1.0 - 0.25 * z1 * (1.0 - 0.5 * z1)) / (1.0 - 0.25 * z2 * (1.0 - 0.5 * z2))));
        } else {
            // if any of the log arguments is zero
            chi[i] = 0.0;
        }
    }

    // delta
    il::StaticArray<double, 3> delta;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        delta[i] = p(i, i) / rho[i] - p(i, i1) / rho[i1];
    }

    // L
    il::StaticArray<double, 3> L;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        L[i] = 1.0 / rho[i] - 1.0 / rho[i1];
    }

    // D
    il::StaticArray<double, 3> D;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        D[i] = p(i, i) / pow(rho[i], 3.0) - p(i, i1) / pow(rho[i1], 3.0);
    }

    // Lambda
    il::StaticArray<double, 3> Lambda;
    for (int i = 0; i < 3; i++) {
        i1 = i + 1;
        if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
        Lambda[i] = 1.0 / pow(rho[i], 3.0) - 1.0 / pow(rho[i1], 3.0);
    }

    // gamma

    // Note: in Nintcheu Fata (2009, 2011) there are no comments for the case
    // in which the arctan arguments are not defined (in this case, 0/0) and this can happen! Brice
    // regularized this in his mathematica notebook by taking the limits, I understand the case of
    // the edge, but not the cases for the vertices. I just coded up Brice's solution for now, but
    // I didn't include the case inside the element and the case right at the vertices
    il::StaticArray<double, 3> gamma;

    if (id == 1) {
        // if the projected source/receiver point lies on vertex 1
        gamma[0] = 0.0;
        gamma[2] = 0.0;
        int i = 1;
        i1 = 2;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else if (id == 2) {
        // if the projected source/receiver point lies on vertex 2
        gamma[0] = 0.0;
        gamma[1] = 0.0;
        int i = 2;
        i1 = 0;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else if (id == 3) {
        // if the projected source/receiver point lies on vertex 3
        gamma[1] = 0.0;
        gamma[2] = 0.0;
        int i = 0;
        i1 = 1;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else if (abs(q[0]) < eps_tol) {
        // if the projected source/receiver point lies on the edge 1 or its prolongation (in both directions)
        // the limit is needed only for the case eta = 0. However, for eta =! 0, the value of gamma1 is the same
        // as the limit (=zero), so that's why I don't make the difference
        gamma[0] = 0.0;
        int i = 1;
        i1 = 2;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
        i = 2;
        i1 = 0;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else if (abs(q[1]) < eps_tol) {
        // if the projected source/receiver point lies on the edge 2 or its prolongation (in both directions)
        // the limit is needed only for the case eta = 0. However, for eta =! 0, the value of gamma2 is the same
        // as the limit (=zero), so that's why I don't make the difference
        gamma[1] = 0.0;
        int i = 0;
        i1 = 1;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
        i = 2;
        i1 = 0;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else if (abs(q[2]) < eps_tol) {
        // if the projected source/receiver point lies on the edge 3 or its prolongation (in both directions)
        // the limit is needed only for the case eta = 0. However, for eta =! 0, the value of gamma3 is the same
        // as the limit (=zero), so that's why I don't make the difference
        gamma[2] = 0.0;
        int i = 0;
        i1 = 1;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
        i = 1;
        i1 = 2;
        gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                        (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                   - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                          (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
    } else {
        // any other case
        for (int i = 0; i < 3; i++) {
            i1 = i + 1;
            if (i1 > 2) { i1 = 0; } // if index is equal to 3, replace by 0
            gamma[i] = atan((-2.0 * p(i, i) * q[i] * eta * rho[i]) /
                            (pow(q[i], 2.0) * pow(rho[i], 2.0) - pow(p(i, i), 2.0) * pow(eta, 2.0)))
                       - atan((-2.0 * p(i, i1) * q[i] * eta * rho[i1]) /
                              (pow(q[i], 2.0) * pow(rho[i1], 2.0) - pow(p(i, i1), 2.0) * pow(eta, 2.0)));
        }
    }

    // theta for displacements and tractions at the boundary of the domain
    double theta;
    double thetaSum = 0.0;
    for (int i = 0; i < 3; i++) {
        thetaSum += gamma[i];
    }
    if (eta < eps_tol) {
        // if eta is negative or zero
        theta = 0.5 * thetaSum - theta_0;
    } else {
        // if eta is strictly positive
        theta = 0.5 * thetaSum + theta_0;
    }

    std::cout << " Checking parameters sensitive to cases... " << "\n";

    std::cout << " theta: " << theta << "\n";
    std::cout << " eta: " << eta << "\n";
    std::cout << " gamma 1: " << gamma[0] << " gamma 2: " << gamma[1] << " gamma 3: " << gamma[2] << "\n";
    std::cout << " theta_0: " << theta_0 << "\n";

    // sin(alpha[i]) and cos(alpha[i])

    // alpha
    il::StaticArray<double, 3> alpha;
    alpha[0] = alpha1;
    alpha[1] = alpha2;
    alpha[2] = alpha3;

    il::StaticArray<double, 3> sinAlpha;
    il::StaticArray<double, 3> cosAlpha;
    for (int i = 0; i < 3; i++) {
        sinAlpha[i] = sin(alpha[i]);
        cosAlpha[i] = cos(alpha[i]);
    }

    // check generic integrals for stress influence coefficients

    // check generic integrals for DD1

    double I5_Xi = bie::i5_Xi(delta,d,sinAlpha);
    double I7_Xi_Xi_Xi = bie::i7_Xi_Xi_Xi(q,d,sinAlpha,cosAlpha,D,Lambda,delta);
    double I7_Xi_Zeta_Zeta = bie::i7_Xi_Zeta_Zeta(q,d,sinAlpha,cosAlpha,D,Lambda,delta);
    double I7_Xi = bie::i7_Xi(d,D,delta,sinAlpha);
    double I7_Xi_Xi_Zeta = bie::i7_Xi_Xi_Zeta(q,d,sinAlpha,cosAlpha,D,Lambda,delta);
    double I5_Zeta = bie::i5_Zeta(delta,d,cosAlpha);
    double I7_Xi_Xi_Aux = bie::i7_Xi_Xi_Aux(eta,cosAlpha,Lambda,sinAlpha,q,d,D,delta);
    double I7_Xi_Xi_By_Aux = I7_Xi_Xi_Aux/pow(eta,2.0)+theta/(15.0*pow(eta,3.0)); // to verify when eta =! 0 against mma
    double I5_Zeta_Zeta_Aux = bie::i5_Zeta_Zeta_Aux(L,sinAlpha,q,d,delta,cosAlpha);
    double I5_Zeta_Zeta_By_Aux = I5_Zeta_Zeta_Aux + theta/(3*eta); // to verify when eta =! 0 against mma
    double I7_Xi_Zeta = bie::i7_Xi_Zeta(sinAlpha,Lambda,cosAlpha,q,d,D,delta);
    double I5_Xi_Zeta = bie::i5_Xi_Zeta(L,sinAlpha,q,d,delta,cosAlpha);

    std::cout << " Checking generic integrals for stress ... " << "\n";

    std::cout << " Checking generic integrals for DD1 ... " << "\n";

    std::cout << " I5_Xi: " << I5_Xi << "\n";
    std::cout << " I7_Xi_Xi_Xi: " << I7_Xi_Xi_Xi << "\n";
    std::cout << " I7_Xi_Zeta_Zeta: " << I7_Xi_Zeta_Zeta << "\n";
    std::cout << " I7_Xi: " << I7_Xi << "\n";
    std::cout << " I7_Xi_Xi_Zeta: " << I7_Xi_Xi_Zeta << "\n";
    std::cout << " I5_Zeta: " << I5_Zeta << "\n";
//    std::cout << " I7_Xi_Xi_Aux: " << I7_Xi_Xi_Aux << "\n";
    std::cout << " I7_Xi_Xi_By_Aux: " << I7_Xi_Xi_By_Aux << "\n";
//    std::cout << " I5_Zeta_Zeta_Aux: " << I5_Zeta_Zeta_Aux << "\n";
    std::cout << " I5_Zeta_Zeta_By_Aux: " << I5_Zeta_Zeta_By_Aux << "\n";
    std::cout << " I7_Xi_Zeta: " << I7_Xi_Zeta << "\n";
    std::cout << " I5_Xi_Zeta: " << I5_Xi_Zeta << "\n";

    // check generic integrals for DD2

    double I7_Zeta_Zeta_Zeta = bie::i7_Zeta_Zeta_Zeta(q,d,sinAlpha,cosAlpha,D,Lambda,delta);
    double I7_Zeta = bie::i7_Zeta(d,D,delta,cosAlpha);
    double I7_Zeta_Zeta_Aux = bie::i7_Zeta_Zeta_Aux(eta,cosAlpha,Lambda,sinAlpha,q,d,D,delta);
    double I7_Zeta_Zeta_By_Aux = I7_Zeta_Zeta_Aux/pow(eta,2.0)+theta/(15.0*pow(eta,3.0)); // to verify when eta =! 0 against mma
    double I5_Xi_Xi_Aux = bie::i5_Xi_Xi_Aux(L,sinAlpha,q,d,delta,cosAlpha);
    double I5_Xi_Xi_By_Aux = I5_Xi_Xi_Aux + theta/(3*eta); // to verify when eta =! 0 against mma

    std::cout << " Checking generic integrals for DD2 ... " << "\n";

    std::cout << " I7_Zeta_Zeta_Zeta: " << I7_Zeta_Zeta_Zeta << "\n";
    std::cout << " I7_Zeta: " << I7_Zeta << "\n";
//    std::cout << " I7_Zeta_Zeta_Aux: " << I7_Zeta_Zeta_Aux << "\n";
    std::cout << " I7_Zeta_Zeta_By_Aux: " << I7_Zeta_Zeta_By_Aux << "\n";
//    std::cout << " I5_Xi_Xi_Aux: " << I5_Xi_Xi_Aux << "\n";
    std::cout << " I5_Xi_Xi_By_Aux: " << I5_Xi_Xi_By_Aux << "\n";

    // check generic integrals for DD3

    double I5_Aux = bie::i5_Aux(q,d,delta);
    double I5_By_Aux = I5_Aux/pow(eta,2.0) + theta/(3.0*pow(eta,3.0)); // to verify when eta =! 0 against mma
    double I7_Aux = bie::i7_Aux(eta,q,d,D,delta);
    double I7_By_Aux = I7_Aux/pow(eta,4.0) + theta/(5.0*pow(eta,5.0)); // to verify when eta =! 0 against mma

    std::cout << " Checking generic integrals for DD3 ... " << "\n";

    //    std::cout << " I5_Aux: " << I5_Aux << "\n";
    std::cout << " I5_By_Aux: " << I5_By_Aux << "\n";
//    std::cout << " I7_Aux: " << I7_Aux << "\n";
    std::cout << " I7_By_Aux: " << I7_By_Aux << "\n";

    // Check the stress influence coefficients

    double G = 1.0;
    double nu = 0.3;

//    double prefactor = (G/(4.0*il::pi*(1.0-nu))); // common prefactor of all coefficients
//
//    il::StaticArray2D<double, 3, 6> Stress; // output
//
//    // recall:
//    // DD1 (shear), DD2 (shear), DD3 (normal) -> for rows
//    // Stress components: S11, S22, S33, S12, S13, S23 -> for columns
//
//    // stress components due to the unit displacement discontinuity component DD1 (shear)
//
//    Stress(0, 0) = prefactor * ( -3.0 * eta * (I5_Xi - 5.0 * I7_Xi_Xi_Xi) );
//    // s11 = b111
//    Stress(0, 1) = prefactor * ( 3.0 * eta * ( I5_Xi * (-1.0 + 2.0*nu)
//                                               + 5.0 * I7_Xi_Zeta_Zeta ) );
//    // s22 = b221
//    Stress(0, 2) = prefactor * ( -3.0 * eta * ( I5_Xi - 5.0 * I7_Xi * pow(eta,2.0) ) );
//    // s33 = b331
//    Stress(0, 3) = prefactor * ( 3.0 * eta * ( 5.0 * I7_Xi_Xi_Zeta - I5_Zeta * nu ) );
//    // s12 = b121
//    Stress(0, 4) = prefactor * ( 3.0 * ( 5.0 * I7_Xi_Xi_Aux + I5_Zeta_Zeta_Aux * nu ) );
//    // s13 = b131
//    Stress(0, 5) = prefactor * ( 15.0 * I7_Xi_Zeta * pow(eta,2.0)
//                                 - 3.0 * I5_Xi_Zeta * nu );
//    // s23 = b231
//
//    // stress components due to the unit displacement discontinuity component DD2 (shear)
//
//    Stress(1, 0) = prefactor * ( 3.0 * eta * ( 5.0 * I7_Xi_Xi_Zeta
//                                               + I5_Zeta * (-1.0 + 2.0*nu) ) ); // s11 = b112
//    Stress(1, 1) = prefactor * ( -3.0 * eta * ( I5_Zeta - 5.0 * I7_Zeta_Zeta_Zeta ) );
//    // s22 = b222
//    Stress(1, 2) = prefactor * ( -3.0 * eta * ( I5_Zeta - 5.0 * I7_Zeta * pow(eta,2.0) ) );
//    // s33 = b332
//    Stress(1, 3) = prefactor * ( 3.0 * eta * ( 5.0 * I7_Xi_Zeta_Zeta - I5_Xi * nu ) );
//    // s12 = b122
//    Stress(1, 4) = prefactor * ( 15.0 * I7_Xi_Zeta * pow(eta,2.0) - 3.0 * I5_Xi_Zeta * nu );
//    // s13 = b132
//    Stress(1, 5) = prefactor * ( 3.0 * (5.0 * I7_Zeta_Zeta_Aux + I5_Xi_Xi_Aux * nu ) );
//    // s23 = b232
//
//    // stress components due to the unit displacement discontinuity component DD3 (normal)
//
//    Stress(2, 0) = prefactor * ( 3.0 * ( I5_Zeta_Zeta_Aux + 5.0 * I7_Xi_Xi_Aux
//                                         - 2.0 * I5_Zeta_Zeta_Aux * nu ) ); // s11 = b113 TODO: it can be factorized
//    Stress(2, 1) = prefactor * ( 3.0 * ( I5_Xi_Xi_Aux + 5.0 * I7_Zeta_Zeta_Aux
//                                         - 2.0 * I5_Xi_Xi_Aux * nu ) ); // s22 = b223 TODO: it can be factorized
//    Stress(2, 2) = prefactor * ( -6.0 * I5_Aux + 15.0 * I7_Aux );
//    // s33 = b333
//    Stress(2, 3) = prefactor * ( 15.0 * I7_Xi_Zeta * pow(eta,2.0)
//                                 + I5_Xi_Zeta * (-3.0 + 6.0*nu) ); // s12 = b123
//    Stress(2, 4) = prefactor * ( -3.0 * eta * ( I5_Xi - 5.0 * I7_Xi * pow(eta,2.0) ) );
//    // s13 = b133
//    Stress(2, 5) = prefactor * ( -3.0 * eta * ( I5_Zeta - 5.0 * I7_Zeta * pow(eta,2.0) ) );
//    // s23 = b233
//
//    std::cout << " Checking stress influence coefficients ... " << "\n";
//
//    std::cout << " For DD1 ... " << "\n";
//
//    std::cout << " b111: " << Stress(0, 0) << "\n";
//    std::cout << " b221: " << Stress(0, 1) << "\n";
//    std::cout << " b331: " << Stress(0, 2) << "\n";
//    std::cout << " b121: " << Stress(0, 3) << "\n";
//    std::cout << " b131: " << Stress(0, 4) << "\n";
//    std::cout << " b231: " << Stress(0, 5) << "\n";
//
//    std::cout << " For DD2 ... " << "\n";
//
//    std::cout << " b112: " << Stress(1, 0) << "\n";
//    std::cout << " b222: " << Stress(1, 1) << "\n";
//    std::cout << " b332: " << Stress(1, 2) << "\n";
//    std::cout << " b122: " << Stress(1, 3) << "\n";
//    std::cout << " b132: " << Stress(1, 4) << "\n";
//    std::cout << " b232: " << Stress(1, 5) << "\n";
//
//    std::cout << " For DD3 ... " << "\n";
//
//    std::cout << " b113: " << Stress(2, 0) << "\n";
//    std::cout << " b223: " << Stress(2, 1) << "\n";
//    std::cout << " b333: " << Stress(2, 2) << "\n";
//    std::cout << " b123: " << Stress(2, 3) << "\n";
//    std::cout << " b133: " << Stress(2, 4) << "\n";
//    std::cout << " b233: " << Stress(2, 5) << "\n";
//
    il::StaticArray2D<double, 3, 6> B = bie::StressesKernelT0(x,xv,G,nu);

    std::cout << " Checking stress influence coefficients ... " << "\n";

    std::cout << " For DD1 ... " << "\n";

    std::cout << " b111: " << B(0, 0) << "\n";
    std::cout << " b221: " << B(0, 1) << "\n";
    std::cout << " b331: " << B(0, 2) << "\n";
    std::cout << " b121: " << B(0, 3) << "\n";
    std::cout << " b131: " << B(0, 4) << "\n";
    std::cout << " b231: " << B(0, 5) << "\n";

    std::cout << " For DD2 ... " << "\n";

    std::cout << " b112: " << B(1, 0) << "\n";
    std::cout << " b222: " << B(1, 1) << "\n";
    std::cout << " b332: " << B(1, 2) << "\n";
    std::cout << " b122: " << B(1, 3) << "\n";
    std::cout << " b132: " << B(1, 4) << "\n";
    std::cout << " b232: " << B(1, 5) << "\n";

    std::cout << " For DD3 ... " << "\n";

    std::cout << " b113: " << B(2, 0) << "\n";
    std::cout << " b223: " << B(2, 1) << "\n";
    std::cout << " b333: " << B(2, 2) << "\n";
    std::cout << " b123: " << B(2, 3) << "\n";
    std::cout << " b133: " << B(2, 4) << "\n";
    std::cout << " b233: " << B(2, 5) << "\n";

    // check additional generic integrals for displacement influence coefficients

    double I3_Xi = bie::i3_Xi(chi,sinAlpha);
    double I3_Zeta = bie::i3_Zeta(chi,cosAlpha);

    std::cout << " Checking remaining generic integrals for displacement ... " << "\n";

    std::cout << " I3_Xi: " << I3_Xi << "\n";
    std::cout << " I3_Zeta: " << I3_Zeta << "\n";

    // Check the displacement influence factors

//    double prefactor = (1.0/(8.0*il::pi*(1.0-nu))); // common prefactor of all coefficients
//
//    il::StaticArray2D<double, 3, 3> Displacement; // output
//
//    // Displacement row is dof (DDx,DDy,DDx), columns are Ux,Uy,Uz in the local reference system
//    // TODO: check if this is ok, Carlo's comment above is not what's done
//
//    // displacement components due to the unit displacement discontinuity DD1 (shear)
//    Displacement(0, 0) = prefactor * ( 3.0 * I5_Xi_Xi_Aux * eta - 2.0 * theta * (-1.0 + nu) );
//    // U1 = a11
//    Displacement(1, 0) = prefactor * ( 3.0 * I5_Xi_Zeta * eta );
//    // U2 = a21
//    Displacement(2, 0) = prefactor * ( I3_Xi + 3.0 * I5_Xi * pow(eta,2.0) - 2.0 * I3_Xi * nu );
//    // U3 = a31 TODO: it can be factorized
//
//    // displacement components due to the unit displacement discontinuity DD2 (shear)
//    Displacement(0, 1) = prefactor * ( 3.0 * I5_Xi_Zeta * eta );
//    // U1 = a12
//    Displacement(1, 1) = prefactor * ( 3.0 * I5_Zeta_Zeta_Aux * eta - 2.0 * theta * (-1 + nu) );
//    // U2 = a22
//    Displacement(2, 1) = prefactor * ( I3_Zeta + 3.0 * I5_Zeta * pow(eta,2.0) - 2.0 * I3_Zeta * nu );
//    // U3 = a32 TODO: it can be factorized
//
//    // displacement components due to the unit displacement discontinuity DD3 (normal)
//    Displacement(0, 2) = prefactor * ( 3.0 * I5_Xi * pow(eta,2.0) + I3_Xi * (-1.0 + 2.0 * nu) );
//    // U1 = a13
//    Displacement(1, 2) = prefactor * ( 3.0 * I5_Zeta * pow(eta,2.0) + I3_Zeta * (-1.0 + 2.0 * nu) );
//    // U2 = a23
//    Displacement(2, 2) = prefactor * ( 3.0 * I5_Aux * eta - 2.0 * theta * (-1.0 + nu) );
//    // U3 = a33
//
//    std::cout << " Checking displacement influence coefficients ... " << "\n";
//
//    std::cout << " For DD1 ... " << "\n";
//
//    std::cout << " a11: " << Displacement(0, 0) << "\n";
//    std::cout << " a21: " << Displacement(1, 0) << "\n";
//    std::cout << " a31: " << Displacement(2, 0) << "\n";
//
//    std::cout << " For DD2 ... " << "\n";
//
//    std::cout << " a12: " << Displacement(0, 1) << "\n";
//    std::cout << " a22: " << Displacement(1, 1) << "\n";
//    std::cout << " a32: " << Displacement(2, 1) << "\n";
//
//    std::cout << " For DD3 ... " << "\n";
//
//    std::cout << " a13: " << Displacement(0, 2) << "\n";
//    std::cout << " a23: " << Displacement(1, 2) << "\n";
//    std::cout << " a33: " << Displacement(2, 2) << "\n";

    il::StaticArray2D<double, 3, 3> A = bie::DisplacementKernelT0(x,xv,nu);

    std::cout << " Checking displacement influence coefficients ... " << "\n";

    std::cout << " For DD1 ... " << "\n";

    std::cout << " a11: " << A(0, 0) << "\n";
    std::cout << " a21: " << A(1, 0) << "\n";
    std::cout << " a31: " << A(2, 0) << "\n";

    std::cout << " For DD2 ... " << "\n";

    std::cout << " a12: " << A(0, 1) << "\n";
    std::cout << " a22: " << A(1, 1) << "\n";
    std::cout << " a32: " << A(2, 1) << "\n";

    std::cout << " For DD3 ... " << "\n";

    std::cout << " a13: " << A(0, 2) << "\n";
    std::cout << " a23: " << A(1, 2) << "\n";
    std::cout << " a33: " << A(2, 2) << "\n";


    std::cout << "----------end of test 3DT0  ---------------------\n";

    return 0;
}
/////////////////////////////////////////////////////////////////////////////////


// READ COORD CSV ///////////////////////////////////////////////////////////////////////////////
// For the 3D code
// Functions to read coordinate matrix and connectivity matrix from CSV file

il::Array2D<double> read_coord_CSV(std::string& inputfileCOORD){

    il::Array2D<double> coordinates_table;

    std::string line = "", subline = "";
    char delim = ',';

    std::ifstream nf; // node coordinate file stream
    nf.open(inputfileCOORD.c_str());
    if(nf.is_open()) {
        // counting the array size
        il::int_t n_row = 0, n_col = 0;
        while(!nf.eof()) {
            std::getline(nf, line);
            if (line.length() == 0) {
                std::cout << "Empty row" << std::endl;
            } else {
                std::stringstream linestream(line);
                ++n_row;
                il::int_t n_col_t = 0;
                do {
                    subline = "";
                    std::getline(linestream, subline, delim);
                    if (subline.length() > 0){
                        ++n_col_t;
                    }
                } while(subline.length() > 0);
                if (n_col_t != n_col && n_col > 0) {
                    std::cout << "Row size is not constant" << std::endl;
                }
                else n_col = n_col_t;
            }
        }
        nf.clear();
        nf.seekg(0);
        // resizing the output array (node coordinates)
        coordinates_table.Resize(n_row,n_col);
        // importing the array
        for (il::int_t k = 0; k < coordinates_table.size(0); ++k) {
            std::getline(nf, line);
            std::stringstream linestream(line);
            for (il::int_t j = 0; j < coordinates_table.size(1); ++j) {
                subline = "";
                std::getline(linestream, subline, delim);
                if (subline.length() > 0){
                    double value = -1.0;
                    std::stringstream sublinestream(subline);
                    sublinestream >> value;
                    coordinates_table(k,j) = value;
                }
            }
        }
        nf.close();
        return coordinates_table;
    }
    else {
        std::cout << "Can't open the file" << std::endl;
    }
}
/////////////////////////////////////////////////////////////////////////////////


// READ CONN CSV ///////////////////////////////////////////////////////////////////////////////
il::Array2D<il::int_t> read_conn_CSV(std::string& inputfileCONN){

    il::Array2D<il::int_t> connectivity_table;

    std::string line = "", subline = "";
    char delim = ',';
    std::ifstream cf; // connectivity file stream
    cf.open(inputfileCONN.c_str());
    if(cf.is_open()) {
        // counting the array size
        il::int_t n_row = 0, n_col = 0;
        while(!cf.eof()) {
            std::getline(cf, line);
            if (line.length() == 0) {
                std::cout << "Empty row" << std::endl;
            } else {
                std::stringstream linestream(line);
                ++n_row;
                il::int_t n_col_t = 0;
                do {
                    subline = "";
                    std::getline(linestream, subline, delim);
                    if (subline.length() > 0){
                        ++n_col_t;
                    }
                } while(subline.length() > 0);
                if (n_col_t != n_col && n_col > 0) {
                    std::cout << "Row size is not constant" << std::endl;
                }
                else n_col = n_col_t;
            }
        }
        cf.clear();
        cf.seekg(0);
        // resizing the output array (mesh connectivity)
        connectivity_table.Resize(n_row,n_col);
        // importing the array
        for (il::int_t k = 0; k < connectivity_table.size(0); ++k) {
            std::getline(cf, line);
            std::stringstream linestream(line);
            for (il::int_t j = 0; j < connectivity_table.size(1); ++j) {
                subline = "";
                std::getline(linestream, subline, delim);
                if (subline.length() > 0){
                    il::int_t value = -1;
                    std::stringstream sublinestream(subline);
                    sublinestream >> value;
                    connectivity_table(k, j) = value;
                }
            }
        }

        cf.close();
        return connectivity_table;
    }
    else {
        std::cout << "Can't open the file" << std::endl;
    }

}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DT0 PENNY SHAPED ///////////////////////////////////////////////////////////////////////////////
int test3DT0_PennyShaped(std::string& vertices_file, std::string& connectivity_file){

    // Make sure to use the kernel in local-local

    std::cout << "-------------- test3DT0 Penny-Shaped Crack ---------------------\n";

    il::Array2D<double> nodes = read_coord_CSV(vertices_file);
    il::Array2D<il::int_t> conn = read_conn_CSV(connectivity_file);

    std::cout << nodes.size(0) << " x " << nodes.size(1) << "\n";

    std::cout << nodes(0,0) << " " << nodes(0,1) << " " << nodes(0,2) <<  "\n";
    std::cout << nodes(1,0) << " " << nodes(1,1) << " " << nodes(1,2) <<  "\n";
    std::cout << nodes(2,0) << " " << nodes(2,1) << " " << nodes(2,2) <<  "\n";

    std::cout << conn.size(0) << " x " << conn.size(1) << "\n";

    std::cout << conn(0,0) << " " << conn(0,1) << " " << conn(0,2) <<  "\n";
    std::cout << conn(1,0) << " " << conn(1,1) << " " << conn(1,2) <<  "\n";
    std::cout << conn(2,0) << " " << conn(2,1) << " " << conn(2,2) <<  "\n";

    bie::Mesh3D mesh(nodes, conn, 0);

    double nu = 0.25;
    double G = 1.0;
    double young = 2.0 * G * (1.0+nu);

    // now we use the BIGWHAM

    // convert to std vectors
    std::vector<double> nodes_flat;
    nodes_flat.reserve(3 * nodes.size(0));
    for (int i = 0; i < nodes.size(0); i++){
        for (int j = 0; j < nodes.size(1); j++){
            nodes_flat.push_back(nodes(i,j));
        }
    }
    std::vector<int> conn_flat;
    conn_flat.reserve(3 * conn.size(0));
    for (int i = 0; i < conn.size(0); i++){
        for (int j = 0; j < conn.size(1); j++){
            conn_flat.push_back(conn(i,j));
        }
    }

    std::cout << "Checking conversion of nodes and conn to std vectors ..." << "\n";
    std::cout << "coor =  " << nodes_flat[0] << " " << nodes_flat[1] << " " << nodes_flat[2] << "\n";
    std::cout << "conn =  " << conn_flat[0] << " " << conn_flat[1] << " " << conn_flat[2] << "\n";

    const std::vector<double> properties = {young, nu}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 25;
    const double eta_test = 5.;
    const double eps_aca = 0.0001;

    // create HMAT
    const std::string kernel_name = "3DT0";
    Bigwhamio test;
    test.set(nodes_flat,conn_flat,kernel_name,properties,
                         max_leaf_size, eta_test, eps_aca);

    // test Hdot product

    // first compute analytical dd

    // compute radius at all nodes ( = collocation points for T0)

    il::Array2D<double> nodes_coor = mesh.getNodes(); // be careful they are transposed!

    il::Array<double> radius{nodes_coor.size(0),0.};

    for (int i = 0; i < nodes_coor.size(0); i++){
        double sum = 0.0;
        for (int j = 0; j < 3; j++){
            sum += pow(nodes_coor(i,j),2.0);
        }
        radius[i] = sqrt(sum);
    }

    std::vector<int> perm = test.getPermutation();
    std::cout << "permutation ... " << "\n";
    std::cout << perm[0] << "\n";
    std::cout << perm[1] << "\n";
    std::cout << perm[2] << "\n";
    std::cout << perm[3] << "\n";

    std::cout << "radius ... " << "\n";
    for (int i = 0; i < 9; i++) {
        std::cout << radius[i] << "\n";
    }

    // compute dd at all nodes ( = collocation points for T0)

    // choose the direction to be tested
    int direction = 1;

    double load = 1.0;
    double R = 1.0;
    il::Array<double> dd_analytical{3*nodes_coor.size(0),0.0};

    switch (direction) {
        case 1: {
            // For 1 (shear)
            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
            }
            break;
        }
        case 2: {
            // For 2 (shear)
            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i+1] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
            }
            break;
        }
        case 3: {
            // For 3 (normal)
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i+2] = ( (8.0*load)/( il::pi * (young/(1.0 - pow(nu,2.0))) ) )
                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
            }
        }
    }

    // repeating
//    direction = 2;
//    load = 0.5;
//    switch (direction) {
//        case 1: {
//            // For 1 (shear)
//            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
//                                     * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//            }
//            break;
//        }
//        case 2: {
//            // For 2 (shear)
//            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i+1] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
//                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//            }
//            break;
//        }
//        case 3: {
//            // For 3 (normal)
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i+2] = ( (8.0*load)/( il::pi * (young/(1.0 - pow(nu,2.0))) ) )
//                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//            }
//        }
//    }

    std::cout << "analytical dd ... " << "\n";
    for (int i = 0; i < 9; i++){
        std::cout << dd_analytical[i] << "\n";
    }

    // convert dd to std vector
    std::vector<double> dd_analytical_std;
    dd_analytical_std.reserve(3 * nodes_coor.size(0));
    for (int i = 0; i < 3 * nodes_coor.size(0); i++){
        dd_analytical_std.push_back(dd_analytical[i]);
    }

    std::vector<double> traction_numerical;
    std::cout << "Doing hmat dot product \n" ;

    traction_numerical = test.hdotProduct(dd_analytical_std);

    // output tractions

    std::cout << "numerical traction ... " << "\n";
    for (int i = 0; i < nodes_coor.size(0); i++){
        std::cout << radius[i] << " (" << traction_numerical[3*i] << ", " << traction_numerical[3*i+1]
        << ", " << traction_numerical[3*i+2] << ")" << "\n";
    }

    // average traction - only for component 3
    double sum = 0;
    int count_nan = 0;

    for (int i = 0; i < nodes_coor.size(0); i++){
        if (std::isnan(traction_numerical[3*i+(direction-1)]) == 0){
            sum += traction_numerical[3*i+(direction-1)];
        } else {
            count_nan++;
        }
    }
    double average_traction = sum / (nodes_coor.size(0)-count_nan);
    std::cout << "average_traction =  " << average_traction << "\n";
    std::cout << "numer of nan's =  " << count_nan << "\n";

    std::cout << "Testing compute stress 3D function" << "\n";

    std::vector<double> obsPts{0.0,0.0,1.0}; //+2.22045e-16

    std::vector<double> stress_tensor = test.computeStresses(dd_analytical_std, obsPts, 1, properties, nodes_flat,
            conn_flat, true);

    std::cout << "s11 = " << stress_tensor[0] << "\n";
    std::cout << "s22 = " << stress_tensor[1] << "\n";
    std::cout << "s33 = " << stress_tensor[2] << "\n";
    std::cout << "s12 = " << stress_tensor[3] << "\n";
    std::cout << "s13 = " << stress_tensor[4] << "\n";
    std::cout << "s23 = " << stress_tensor[5] << "\n";

    std::cout << "-------------- End of 3DT0 - Penny-shaped crack test ------------- " << "\n";

    return  0;

}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DT6 PENNY SHAPED ///////////////////////////////////////////////////////////////////////////////
int test3DT6_PennyShaped(std::string& vertices_file, std::string& connectivity_file){

    std::cout << "-------------- test3DT6 Penny-Shaped Crack ---------------------\n";

    il::Array2D<double> nodes = read_coord_CSV(vertices_file);
    il::Array2D<il::int_t> conn = read_conn_CSV(connectivity_file);

    std::cout << nodes.size(0) << " x " << nodes.size(1) << "\n";

    std::cout << nodes(0,0) << " " << nodes(0,1) << " " << nodes(0,2) <<  "\n";
    std::cout << nodes(1,0) << " " << nodes(1,1) << " " << nodes(1,2) <<  "\n";
    std::cout << nodes(2,0) << " " << nodes(2,1) << " " << nodes(2,2) <<  "\n";

    std::cout << conn.size(0) << " x " << conn.size(1) << "\n";

    std::cout << conn(0,0) << " " << conn(0,1) << " " << conn(0,2) <<  "\n";
    std::cout << conn(1,0) << " " << conn(1,1) << " " << conn(1,2) <<  "\n";
    std::cout << conn(2,0) << " " << conn(2,1) << " " << conn(2,2) <<  "\n";

    bie::Mesh3D mesh(nodes, conn, 0);

    il::Array2D<double> coll_points = mesh.getCollocationPoints();

//    il::int_t leaf_size = 32; // 10
//    il::Timer tt;
//    tt.Start();
//    const il::Cluster cluster = il::cluster(leaf_size, il::io, coll_points);
//
//    double eta=10;
//    const il::Tree<il::SubHMatrix, 4> hmatrix_tree =
//            il::hmatrixTreeIxI(coll_points, cluster.partition, eta);
//    tt.Stop();
//
//    std::cout << "Time for cluster construction " << tt.time() <<"\n";
//    tt.Reset();
//    std::cout << cluster.partition.depth() <<"\n";

    double nu = 0.2;
    double G = 1.0;
    double young = 2.0 * G * (1.0+nu);

//    bie::ElasticProperties elas(young,nu);
//
//    il::HMatrix<double> h_ ;
//    const bie::ElasticHMatrix3DT0<double> M{coll_points,cluster.permutation,mesh,elas,1,1,1};
//
//    std::cout << " create h mat " << M.size(0) <<"\n";
//    tt.Start();
//    double epsilon_aca = 0.001;
//    h_= il::toHMatrix(M, hmatrix_tree, epsilon_aca);
//    tt.Stop();
//    std::cout << " create h mat ended " << h_.isBuilt() << "time" << tt.time() << "\n";
//    tt.Reset();
//    std::cout << " compression ratio " << il::compressionRatio(h_) << "\n";
//
//    // Analytical solution
//
//    // compute radius at all nodes ( = collocation points for T0)
//
//    il::Array2D<double> nodes_coor = mesh.getNodes();
//    il::Array<double> radius{nodes_coor.size(0)};
//
//    for (int i = 0; i < nodes_coor.size(0); i++){
//        double sum = 0.0;
//        for (int j = 0; j < 3; j++){
//            sum += pow(nodes_coor(i,j),2.0);
//        }
//        radius[i] = sqrt(sum);
//    }
//
//    std::cout << "radius ... " << "\n";
//    for (int i = 0; i < 9; i++) {
//        std::cout << radius[i] << "\n";
//    }
//
//    // compute dd at all nodes ( = collocation points for T0)
//
//    // choose the direction to be tested
//    int direction = 2;
//
//    double load = 1.0;
//    double R = 1.0;
//    il::Array<double> dd_analytical{3*nodes_coor.size(0)};
//
//    switch (direction) {
//        case 1: {
//            // For 1 (shear)
//            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i] = ( ((8.0*lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
//                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//                dd_analytical[3*i+1] = 0.0;
//                dd_analytical[3*i+2] = 0.0;
//            }
//            break;
//        }
//        case 2: {
//            // For 2 (shear)
//            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i] = 0.0;
//                dd_analytical[3*i+1] = ( ((8.0*lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
//                * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//                dd_analytical[3*i+2] = 0.0;
//            }
//            break;
//        }
//        case 3: {
//            // For 3 (normal)
//            for (int i = 0; i < nodes_coor.size(0); i++){
//                dd_analytical[3*i] = 0.0;
//                dd_analytical[3*i+1] = 0.0;
//                dd_analytical[3*i+2] = ( (8.0*load)/( il::pi * (young/(1.0 - pow(nu,2.0))) ) )
//                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
//            }
//        }
//    }
//
//    std::cout << "analytical dd ... " << "\n";
//    for (int i = 0; i < 9; i++){
//        std::cout << dd_analytical[i] << "\n";
//    }
//
//    // perform h-dot
//
//    // first see permutation
//    il::Array<il::int_t> perm = cluster.permutation;
////    std::cout << "permuation vector, size =  " << perm.size() << "\n";
////
////    std::cout << "the vector ... " << "\n";
////    for (int i = 0; i < perm.size(); i++){
////        std::cout << perm[i] << "\n";
////    }
//
//    // permute dd vector
//    il::Array<double> dd_analytical_perm{3*nodes_coor.size(0)};
//    for (int i = 0; i < nodes_coor.size(0); i++){
//        dd_analytical_perm[3*i] = dd_analytical[3*perm[i]];
//        dd_analytical_perm[3*i+1] = dd_analytical[3*perm[i]+1];
//        dd_analytical_perm[3*i+2] = dd_analytical[3*perm[i]+2];
//    }
//
//    il::Array<double> traction_numerical = il::dot(h_,dd_analytical_perm);
//
//    // output tractions
//
//    std::cout << "numerical traction ... " << "\n";
//    for (int i = 0; i < nodes_coor.size(0); i++){
//        std::cout << radius[i] << " (" << traction_numerical[3*i] << ", " << traction_numerical[3*i+1]
//        << ", " << traction_numerical[3*i+2] << ")" << "\n";
//    }
//
//    // average traction - only for component 3
//    double sum = 0;
//    int count_nan = 0;
//
//    for (int i = 0; i < nodes_coor.size(0); i++){
//        if (std::isnan(traction_numerical[3*i+2]) == 0){
//            sum += traction_numerical[3*i+2];
//        } else {
//            count_nan++;
//        }
//    }
//    double average_traction = sum / (nodes_coor.size(0)-count_nan);
//    std::cout << "average_traction =  " << average_traction << "\n";

    // now we use the BIGWHAM

    // convert to std vectors
    std::vector<double> nodes_flat;
    nodes_flat.reserve(3 * nodes.size(0));
    for (int i = 0; i < nodes.size(0); i++){
        for (int j = 0; j < nodes.size(1); j++){
            nodes_flat.push_back(nodes(i,j));
        }
    }
    std::vector<int> conn_flat;
    conn_flat.reserve(3 * conn.size(0));
    for (int i = 0; i < conn.size(0); i++){
        for (int j = 0; j < conn.size(1); j++){
            conn_flat.push_back(conn(i,j));
        }
    }

    std::cout << "Checking conversion of nodes and conn to std vectors ..." << "\n";
    std::cout << "coor =  " << nodes_flat[0] << " " << nodes_flat[1] << " " << nodes_flat[2] << "\n";
    std::cout << "conn =  " << conn_flat[0] << " " << conn_flat[1] << " " << conn_flat[2] << "\n";

    const std::vector<double> properties = {young, nu}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 25;
    const double eta_test = 5.;
    const double eps_aca = 0.0001;

    // create HMAT
    const std::string kernel_name = "3DT6";
    Bigwhamio test;
    test.set(nodes_flat,conn_flat,kernel_name,properties,
             max_leaf_size, eta_test, eps_aca);

    // test Hdot product

    // first compute analytical dd

    // compute radius at all nodes ( = collocation points for T0)

    il::Array2D<double> nodes_coor = mesh.getNodes();
    il::Array<double> radius{nodes_coor.size(0)};

    for (int i = 0; i < nodes_coor.size(0); i++){
        double sum = 0.0;
        for (int j = 0; j < 3; j++){
            sum += pow(nodes_coor(i,j),2.0);
        }
        radius[i] = sqrt(sum);
    }

    std::cout << "radius ... " << "\n";
    for (int i = 0; i < 9; i++) {
        std::cout << radius[i] << "\n";
    }

    // compute dd at all nodes ( = collocation points for T0)

    // choose the direction to be tested
    int direction = 1;

    double load = 1.0;
    double R = 1.0;
    il::Array<double> dd_analytical{3*nodes_coor.size(0)};

    switch (direction) {
        case 1: {
            // For 1 (shear)
            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
                                     * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
                dd_analytical[3*i+1] = 0.0;
                dd_analytical[3*i+2] = 0.0;
            }
            break;
        }
        case 2: {
            // For 2 (shear)
            double lame2 = (young*nu)/((1.0+nu)*(1.0-2.0*nu));
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i] = 0.0;
                dd_analytical[3*i+1] = ( (8.0*(lame2+2.0*G)*load)/(il::pi*G*(3.0*lame2+4.0*G)) )
                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
                dd_analytical[3*i+2] = 0.0;
            }
            break;
        }
        case 3: {
            // For 3 (normal)
            for (int i = 0; i < nodes_coor.size(0); i++){
                dd_analytical[3*i] = 0.0;
                dd_analytical[3*i+1] = 0.0;
                dd_analytical[3*i+2] = ( (8.0*load)/( il::pi * (young/(1.0 - pow(nu,2.0))) ) )
                                       * sqrt(abs( pow(R,2.0) - pow(radius[i],2.0) ));
            }
        }
    }

    // permute dd
    std::vector<int> perm = test.getPermutation();

    // permute dd vector
    il::Array<double> dd_analytical_perm{3*nodes_coor.size(0)};
    for (int i = 0; i < nodes_coor.size(0); i++){
        dd_analytical_perm[3*i] = dd_analytical[3*perm[i]];
        dd_analytical_perm[3*i+1] = dd_analytical[3*perm[i]+1];
        dd_analytical_perm[3*i+2] = dd_analytical[3*perm[i]+2];
    }

    std::cout << "analytical dd ... " << "\n";
    for (int i = 0; i < 9; i++){
        std::cout << dd_analytical[i] << "\n";
    }

    // convert dd to std vector
    std::vector<double> dd_analytical_std;
    dd_analytical_std.reserve(3 * nodes_coor.size(0));
    for (int i = 0; i < 3 * nodes_coor.size(0); i++){
        dd_analytical_std.push_back(dd_analytical[i]);
    }

    std::vector<double> traction_numerical;
    std::cout << "Doing hmat dot product \n" ;

    traction_numerical = test.hdotProduct(dd_analytical_std);

    // output tractions

    std::cout << "numerical traction ... " << "\n";
    for (int i = 0; i < nodes_coor.size(0); i++){
        std::cout << radius[i] << " (" << traction_numerical[3*i] << ", " << traction_numerical[3*i+1]
                  << ", " << traction_numerical[3*i+2] << ")" << "\n";
    }

    // average traction - only for component 3
    double sum = 0;
    int count_nan = 0;

    for (int i = 0; i < nodes_coor.size(0); i++){
        if (std::isnan(traction_numerical[3*i+2]) == 0){
            sum += traction_numerical[3*i+2];
        } else {
            count_nan++;
        }
    }
    double average_traction = sum / (nodes_coor.size(0)-count_nan);
    std::cout << "average_traction =  " << average_traction << "\n";
    std::cout << "numer of nan's =  " << count_nan << "\n";


//
//
//    for(int i=0; i<x.size(); ++i) std::cout << x[i] << " ";
//    std::cout << "\n" ;
//    std::cout << "Displacement HMAT dot product \n" ;
//    x = displacementHMAT.hdotProduct(xx);
//    for(int i=0; i<x.size(); ++i) std::cout << x[i] << ' ';
//
//    // compute stresses at a set of observation points
//    std::vector<double> coorobsp={-10.,-10.,0.,
//                                  20.,-20.,0.};
//    std::vector<double> mysol = {1.,1.,1.,1.,1.,1.};
//    std::vector<double> bb = tractionHMAT.computeStresses(mysol, coorobsp, 2, properties, coor, conn, true);
//
//    for(int i=0; i<bb.size()/6; ++i) {
//        std::cout << "\n stress at point #" << i << "\n ";
//        for(int j=0; j<6; ++j){
//            std::cout << bb[i*6+j] << " ";
//        }
//    }



    std::cout << "-------------- End of 3DT6 - Penny-shaped crack test ------------- " << "\n";

    return  0;

}
/////////////////////////////////////////////////////////////////////////////////


// CHECK 3DR0 ///////////////////////////////////////////////////////////////////////////////
int check3DR0() {

    std::cout << "-------------- test3DR0 ---------------------\n";

    std::string vertices_file ="/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/vertices.csv";
    std::string connectivity_file = "/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/conn.csv";

    il::Array2D<double> nodes = read_coord_CSV(vertices_file);
    il::Array2D<il::int_t> conn = read_conn_CSV(connectivity_file);

    std::cout << nodes.size(0) << " x " << nodes.size(1) << "\n";

    std::cout << nodes(0,0) << " " << nodes(0,1) << " " << nodes(0,2) <<  "\n";
    std::cout << nodes(1,0) << " " << nodes(1,1) << " " << nodes(1,2) <<  "\n";
    std::cout << nodes(2,0) << " " << nodes(2,1) << " " << nodes(2,2) <<  "\n";

    std::cout << conn.size(0) << " x " << conn.size(1) << "\n";

    std::cout << conn(0,0) << " " << conn(0,1) << " " << conn(0,2) <<  "\n";
    std::cout << conn(1,0) << " " << conn(1,1) << " " << conn(1,2) <<  "\n";
    std::cout << conn(2,0) << " " << conn(2,1) << " " << conn(2,2) <<  "\n";

    bie::Mesh3D mesh(nodes, conn, 0);



    // now we use the BIGWHAM

    // convert to std vectors
    std::vector<double> nodes_flat;
    nodes_flat.reserve(3 * nodes.size(0));
    for (int i = 0; i < nodes.size(0); i++){
        for (int j = 0; j < nodes.size(1); j++){
            nodes_flat.push_back(nodes(i,j));
        }
    }
    std::vector<int> conn_flat;
    conn_flat.reserve(3 * conn.size(0));
    for (int i = 0; i < conn.size(0); i++){
        for (int j = 0; j < conn.size(1); j++){
            conn_flat.push_back(conn(i,j));
        }
    }

    std::cout << "Checking conversion of nodes and conn to std vectors ..." << "\n";
    std::cout << "coor =  " << nodes_flat[0] << " " << nodes_flat[1] << " " << nodes_flat[2] << "\n";
    std::cout << "conn =  " << conn_flat[0] << " " << conn_flat[1] << " " << conn_flat[2] << "\n";

    il::int_t nnodes_elts=4, p=0, dimension_=3;
    il::int_t nelts = conn_flat.size() / nnodes_elts;
    il::int_t nvertex = nodes_flat.size() / dimension_;

    il::Array2D<double> Coor{nvertex, dimension_, 0.};  // columm major order
    il::Array2D<il::int_t> Conn{nelts, nnodes_elts, 0};

    // populate mesh (loops could be optimized - passage row-major to
    // col-major)
    int index = 0;
    for (il::int_t i = 0; i < Coor.size(0); i++) {
        for (il::int_t j = 0; j < Coor.size(1); j++) {
            Coor(i, j) = nodes_flat[index];
            index++;
        }
    }

    index = 0;
    for (il::int_t i = 0; i < Conn.size(0); i++) {
        for (il::int_t j = 0; j < Conn.size(1); j++) {
            Conn(i, j) = conn_flat[index];
            index++;
        }
    }

    bie::Mesh3D mesh3d(Coor, Conn, p);

    // Create and open a text file    --- TODO this is machine dependent - needs to be fixed
    std::string normal_out_file = "/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/normalout.csv";
    std::string s1_out_file = "/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/s1out.csv";
    std::string s2_out_file = "/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/s2out.csv";
    std::string cp_out_file = "/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/cpout.csv";
    std::ofstream normalFile(normal_out_file);
    std::ofstream s1File(s1_out_file);
    std::ofstream s2File(s2_out_file);
    std::ofstream cpFile(cp_out_file);

    // print the normal to file
    for (il::int_t i = 0; i < nelts; i++) {
        bie::FaceData myel=mesh3d.getElementData(i);
        il::Array<double> normal = myel.getNormal();
        normalFile << normal[0] << "," << normal[1] << "," << normal[2];
        normalFile << "\n";
        il::Array2D<double> cp = myel.getCollocationPoints();
        cpFile << cp(0,0) << "," <<  cp(0,1)<< "," <<  cp(0,2);
        cpFile << "\n";
    }

    // print the s1 and s2 to file
    for (il::int_t i = 0; i < nelts; i++) {
        bie::FaceData myel=mesh3d.getElementData(i);
        il::Array<double> s1 = myel.getS1();
        s1File << s1[0] << "," << s1[1] << "," << s1[2];
        s1File << "\n";
        il::Array<double> s2 = myel.getS2();
        s2File << s2[0] << "," << s2[1] << "," << s2[2];
        s2File << "\n";
    }

    // Close the file
    normalFile.close();\
    cpFile.close();
    std::cout << "DONE! " << "\n";

    double nu = 0.48;
    //double G = 1.0;
    //double young = 2.0 * G * (1.0+nu);
    double young = 97000;
    const std::vector<double> properties = {young, nu}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 100;
    const double eta_test = 0.;
    const double eps_aca = 0.0001;

    // create HMAT
    const std::string kernel_name = "3DR0";
    Bigwhamio test;
    test.set(nodes_flat,conn_flat,kernel_name,properties,
             max_leaf_size, eta_test, eps_aca);

//    std::cout << test.matrixSize(0);
//    std::cout << "  \n";
//    std::cout << test.matrixSize(1);
//    // use the Hdot product
//    std::cout<<"\n NoE: " << conn_flat.size()/4;
//    std::vector<double>  xx(3*conn_flat.size()/4);
//    for(int i=0; i<xx.size(); ++i) xx[i]=1.;
//    std::vector<double> res = test.hdotProduct(xx);
//
//    std::cout << "Traction HMAT dot product \n" ;
//    for(int i=0; i<xx.size(); ++i) std::cout << res[i] << " ";
//    std::cout << "\n" ;

    return 0;

}
/////////////////////////////////////////////////////////////////////////////////


// TEST NEW HMAT /////////////////////////////////////////////////////////////////////////////
int testNewHmat() {
  std::cout << "--------------- test new Hmat implemntation Hdot ---------------------\n";
#ifndef NUMBEROFTHREADS
#define NUMBEROFTHREADS 4
#endif
#if _OPENMP
#pragma omp parallel default(none)  num_threads(NUMBEROFTHREADS)
  {
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
#endif

  // star cracks mesh - crack length unity
  il::int_t nfracs=10;
  il::int_t ne_per_frac=100;
  il::Array<double> rad{ne_per_frac+1,0.};
  il::Array<double> angle{nfracs,0.};
//
  for (il::int_t i=0;i<=ne_per_frac;i++){
    rad[i]=1.*i/ne_per_frac;
  }
  for (il::int_t i=0;i<nfracs;i++){
    angle[i]=2*(il::pi)*i/nfracs;
  }
//
  il::int_t ne=ne_per_frac*nfracs;
  il::Array2D<double> nodes{ne+1,2,0.};
  il::Array2D<il::int_t> conn{ne,2};
  il::int_t index=0;
  nodes(index,0)=0.;
  nodes(index,1)=0.;
  for (il::int_t i=0;i<nfracs;i++){
    for (il::int_t j=1;j<ne_per_frac+1;j++){
      index++;
      nodes(index,0)=rad[j]*cos(angle[i]);
      nodes(index,1)=rad[j]*sin(angle[i]);
    }
  }
  for (il::int_t i=0;i<ne;i++){
    conn(i,0)=i;
    conn(i,1)=i+1;
  }
  for (il::int_t j=0;j<nfracs;j++){
    // first node of each fract is the first nod of the mesh.
    conn(j*ne_per_frac,0)=0;
  }

  std::cout << " N unknowns: " << 4*ne <<"\n";
  bie::Mesh mesh(nodes,conn,1);

  bie::ElasticProperties elas(1.0,0.2);
  il::Array2D<double> coll_points = mesh.getCollocationPoints();

  // hmat parameters
  il::int_t leaf_size=32;
  double eta =3.0;
  double epsilon_aca=1.e-4;

  il::Timer tt;

// Construction of the binary cluster Tree
  tt.Start();
  const bie::Cluster cluster = bie::cluster(leaf_size, il::io, coll_points);
  tt.Stop();
  std::cout << "Time for  cluster tree construction " << tt.time() <<"\n";
  std::cout << " cluster depth ..." <<  cluster.partition.depth() <<"\n";
  std::cout << " cluster - part " << cluster.permutation.size() << "\n";
  std::cout << " N nodes at level 2:: "<<  cluster.partition.nnodesAtLevel(2) <<"\n";
  tt.Reset();
  il::Array<il::Range> listnodes=cluster.partition.getNodesAtLevel(2);
  std::cout << " nodes 0 at level 2 " <<  listnodes[0].begin << " / " << listnodes[0].end <<"\n";

//  construction of the block-cluster tree -> the partition of the h-mat
  tt.Start();
  const il::Tree<bie::SubHMatrix, 4> block_tree =
      bie::hmatrixTreeIxI(coll_points, cluster.partition, eta);
  tt.Stop();
  std::cout << "Time for binary cluster tree construction " << tt.time() <<"\n";
  std::cout << " binary cluster depth ..." << block_tree.depth() << "\n";
  std::cout << " root - " << block_tree.root().index << "\n";
  tt.Reset();

  tt.Start();
  const il::Tree<bie::SubHMatrix, 4> block_tree2 =
      bie::hmatrixTreeIxJ(coll_points, cluster.partition, coll_points, cluster.partition,eta);
  tt.Stop();
  std::cout << "Time for binary cluster tree construction 2 " << tt.time() <<"\n";
  std::cout << " binary cluster depth ..." << block_tree2.depth() << "\n";
  std::cout << " root - " << block_tree2.root().index << "\n";
  tt.Reset();

  // the matrix generator
  bie::HMatrix<double> h_ ;
  const bie::ElasticHMatrix2DP1<double> M{coll_points, cluster.permutation,
                                          mesh, elas};
  // creation of the Hmatrix the old way.....
  std::cout << " Create h mat - the old way ...  size" << M.size(0) << " * " << M.size(1) <<"\n";
  tt.Start();
  h_= bie::toHMatrix(M, block_tree, epsilon_aca);
  tt.Stop();
  std::cout << " create h mat  old way in " << tt.time() <<"\n";
  tt.Reset();
  std::cout << " create h mat ended " << h_.isBuilt() <<  " in " << tt.time() <<  "\n";
  std::cout << " compression ratio " << bie::compressionRatio(h_)<<"\n";
  std::cout << " Compressed memory " << (bie::compressionRatio(h_)*8*(4*ne*4*ne)) <<"\n";
  std::cout << " dense case memory " << 8*(4*ne*4*ne) << "\n";
  std::cout <<  "number of blocks " << bie::numberofBlocks(h_) << "\n";
  std::cout << "number of full blocks " << bie::numberofFullBlocks(h_) <<"\n";
  tt.Reset();
  tt.Start();
  il::Array2D<il::int_t> pattern=output_hmatPattern(h_);
  tt.Stop();
  std::cout << " time for getting pattern (old way)"  << tt.time() <<  " number of blocks "  << pattern.size(1) << "\n";
  tt.Reset();

  //
  il::int_t nb=bie::nbBlocks(block_tree);
  std::cout << " Number of sub-matrix blocks: "  << nb <<  " \n";
  il::int_t n_fullb=bie::nbFullBlocks(block_tree);
  std::cout << " Number of sub-matrix full blocks: "  << n_fullb <<  " \n";

// creation of the hmatrix the new way....
  tt.Start();
  bie::HPattern my_patt=bie::createPattern(block_tree);
  std::cout << "Time for pattern construction " << tt.time() <<"\n";
  std::cout << " Number of sub-matrix full blocks: "  << my_patt.n_FRB <<  " \n";
  std::cout  << " n fb " <<  my_patt.FRB_pattern.size(1) <<"\n";
  std::cout  << " n lrb " <<  my_patt.LRB_pattern.size(1) <<"\n";
  my_patt.nr=M.size(0);
  my_patt.nc=M.size(1);
  tt.Reset();
  tt.Start();
  bie::HPattern my_patt2=bie::createPattern(block_tree2);
  std::cout << "Time for pattern construction " << tt.time() <<"\n";
  std::cout << " Number of sub-matrix full blocks: "  << my_patt2.n_FRB <<  " \n";
  std::cout  << " n fb " <<  my_patt2.FRB_pattern.size(1) <<"\n";
  std::cout  << " n lrb " <<  my_patt2.LRB_pattern.size(1) <<"\n";
  tt.Reset();

  //
  bie::Hmat<double> hmt_(my_patt2);
  tt.Start();
  hmt_.build(M,epsilon_aca);
  tt.Stop();
  std::cout << "Creation hmat new way new in "  << tt.time() <<"\n";
  tt.Reset();
  std::cout << "Compression ratio new way " << hmt_.compressionRatio() <<"\n";

  // dot-product of linear system and checks
  double Ep=1.0/(1.0-0.2*0.2);
  double sig = 1.0;
  double a=1.0;
  double coef = 4.0 * sig / (Ep);
  // at collocation points
  il::Array<double> wsol_coll{mesh.numberCollocationPoints(), 0.};
  for (int i = 0; i < coll_points.size(0); ++i) {
    if (std::abs(coll_points(i,0)) < a) {
      wsol_coll[i] = coef * sqrt(pow(a, 2) - pow(coll_points(i,0), 2));
    }
  }
  //at corresponding nodes - works here due to simple mesh....
  il::Array<double> wsol_nodes{coll_points.size(0), 0.};
  double aux_x=0.; int j=0;
  for (int i = 0; i < conn.size(0); ++i) {
    aux_x=nodes(conn(i,0),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
    aux_x=nodes(conn(i,1),0);
    if (std::abs(aux_x) < a) {
      wsol_nodes[j] = coef * sqrt(pow(a, 2) - pow(aux_x, 2));
    }
    j++;
  }

  il::Array<double> xx{coll_points.size(0)*2};
  for (il::int_t i=0;i<xx.size()/2;i++){
    xx[2*i]=0.0;
    xx[2*i+1]=wsol_nodes[i];
  }

  int np = 10;

  tt.Start();
  il::Array< double> y1=il::dot(h_,xx);
  for (int i=0;i<np;i++){
    y1=il::dot(h_,xx);
  }
  tt.Stop();
  std::cout << " time for recursive hdot " << tt.time()/(np+1) <<"\n";
  tt.Reset();
  tt.Start();
  il::Array< double> y2= il::dotwithpattern_serial(h_, pattern, xx);
  for (int i=0;i<np;i++){
    y2= il::dotwithpattern_serial(h_, pattern, xx);
  }
  tt.Stop();
  std::cout << " time for Non-recursive hdot serial " << tt.time()/(np+1) <<"\n";

  il::Array2D<il::int_t> lr_patt{pattern.size(0),0};
  il::Array2D<il::int_t> fr_patt{pattern.size(0),0};
  il::Array2D<il::int_t> fr_patt2{pattern.size(0)+2,0};

  lr_patt.Reserve(3,pattern.size(1));
  fr_patt.Reserve(3,pattern.size(1));
  fr_patt2.Reserve(5,pattern.size(1));

  il::int_t nfb=0;il::int_t nlb=0;
  for (il::int_t i=0;i<pattern.size(1);i++){
    il::spot_t s(pattern(0,i));
    if (h_.isFullRank(s)){

      fr_patt.Resize(3,nfb+1);
      fr_patt(0,nfb)=pattern(0,i);
      fr_patt(1,nfb)=pattern(1,i);
      fr_patt(2,nfb)=pattern(2,i);

      fr_patt2.Resize(5,nfb+1);
      fr_patt2(0,nfb)=pattern(0,i);
      fr_patt2(1,nfb)=pattern(1,i);
      fr_patt2(2,nfb)=pattern(2,i);
      const bie::SubHMatrix info = block_tree.value(s);
      fr_patt2(3,nfb)=info.range0.end;
      fr_patt2(4,nfb)=info.range1.end;

      nfb++;
    } else if (h_.isLowRank(s)){

      lr_patt.Resize(3,nlb+1);
      lr_patt(0,nlb)=pattern(0,i);
      lr_patt(1,nlb)=pattern(1,i);
      lr_patt(2,nlb)=pattern(2,i);
      nlb++;
    } else {
      std::cout <<"error in pattern !\n" ;
      il::abort();
    }
  }
  tt.Reset();

  tt.Start();
  il::Array< double> y3= il::dotwithpattern(h_, fr_patt, lr_patt, xx);
  for (int i=0;i<np;i++){
    y3= il::dotwithpattern(h_, fr_patt, lr_patt, xx);
  }
  tt.Stop();
  std::cout << " time for Non-recursive hdot (parallel_invoke) " << tt.time()/(np+1) <<"\n";

  tt.Reset();

  tt.Start();
  il::Array<double> y4= hmt_.matvec(xx);
  for (int i=0;i<np;i++){
    y4= hmt_.matvec(xx);
  }
  tt.Stop();
  std::cout << " time for Non-recursive hdot .. new way " << tt.time()/(np+1)  <<"\n";
  std::cout << " norm y1 " << il::norm(y1,il::Norm::L2) << " y2 " << il::norm(y2,il::Norm::L2)
            <<" y3 " << il::norm(y3,il::Norm::L2)
          <<" y4 " << il::norm(y4,il::Norm::L2) << "\n";
  std::cout << "----------end of test hdot  ---------------------\n";

  return 0;

}
////////////////////////////////////////////////////////////////////////////////


// TEST PL3D///////////////////////////////////////////////////////////////////////////////
int testPl3D(){
// testing the square DD rectangle kernel used in Pyfrac.
  il::int_t nx=100;
  il::int_t ny=100;
  il::int_t ne=nx*ny;
  double Lx=100.;
  double Ly=100.;
  double hx=2.*Lx/(nx-1);
  double hy=2.*Ly/(ny-1);


  il::Array2D<double> coor{(nx+1)*(ny+1),3,0.};
  il::Array2D<il::int_t> conn{nx*ny,4,0};
  il::int_t k=0;
  for (il::int_t j=0;j<ny+1;j++){
    for (il::int_t i=0;i<nx+1;i++){
      coor(k,0)= i*hx -Lx-hx/2. ; //x
      coor(k,1)= j*hy -Ly-hy/2.; // y
      coor(k,2)=0.;
      k++;
    }
  }
  il::int_t e=0;
  for (il::int_t j=0;j<ny;j++) {
    for (il::int_t i = 0; i < nx; i++) {
      conn(e,0)=j*(nx+1)+i;
      conn(e,1)=j*(nx+1)+i+1;
      conn(e,2)=(j+1)*(nx+1)+i+1;
      conn(e,3)=(j+1)*(nx+1)+i;
      e++;
    }
  }

  bie::Mesh3D mesh(coor,conn,0);
  il::Array2D<double> xv{4,3,0.};
  for (il::int_t j=0;j<3;j++){
    for (il::int_t i=0;i<4;i++) {
      xv(i,j)=coor(conn(5,i),j);
      xv(i,j)=coor(conn(5,i),j);
    }
  }


  bie::FaceData fd(xv,0);
  bie::ElasticProperties elas(1.0,0.2);
  il::Array2D<double> coll_points = mesh.getCollocationPoints();

  // hmat parameters
  il::int_t max_leaf_size=100;
  double eta =3.0;
  double epsilon_aca=1.e-4;

  il::Timer tt;
  tt.Start();
  const bie::Cluster cluster = bie::cluster(max_leaf_size, il::io, coll_points);
  tt.Stop();
  std::cout << "Time for  cluster tree construction " << tt.time() <<"\n";
  std::cout << " cluster depth ..." <<  cluster.partition.depth() <<"\n";
  std::cout << " cluster - part " << cluster.permutation.size() << "\n";
  std::cout << " N nodes at level 2:: "<<  cluster.partition.nnodesAtLevel(2) <<"\n";
  tt.Reset();

  tt.Start();
  const il::Tree<bie::SubHMatrix, 4> block_tree =
      bie::hmatrixTreeIxI(coll_points, cluster.partition,eta);
  tt.Stop();
  std::cout << "Time for binary cluster tree construction 2 " << tt.time() <<"\n";
  std::cout << " binary cluster depth ..." << block_tree.depth() << "\n";
  std::cout << " root - " << block_tree.root().index << "\n";
  tt.Reset();

  // the matrix generator
  bie::HMatrix<double> h_ ;
  const bie::ElasticHMatrix3DR0_mode1Cartesian<double> M{coll_points, cluster.permutation,
                                          mesh, elas};

  // creation of the hmatrix the new way....
  tt.Start();
  bie::HPattern my_patt=bie::createPattern(block_tree);
  std::cout << "Time for pattern construction " << tt.time() <<"\n";
  std::cout << " Number of sub-matrix full blocks: "  << my_patt.n_FRB <<  " \n";
  std::cout  << " n fb " <<  my_patt.FRB_pattern.size(1) <<"\n";
  std::cout  << " n lrb " <<  my_patt.LRB_pattern.size(1) <<"\n";
  my_patt.nr=M.size(0);
  my_patt.nc=M.size(1);
  tt.Reset();

  //
  const il::int_t pp=M.blockSize();
  bie::Hmat<double> hmt_(my_patt);
  tt.Start();
  hmt_.build(M,epsilon_aca);
  tt.Stop();
  std::cout << "Creation hmat new way new in "  << tt.time() <<"\n";
  tt.Reset();
  std::cout << "Compression ratio new way " << hmt_.compressionRatio() <<"\n";

  // some hdot speed
  il::Array<double> xx{mesh.numberOfElts(),0.};
  il::Array<double> y{mesh.numberOfElts(),1.};
    xx[0]=1.;
  tt.Start();
  y=hmt_.matvec(xx);
  tt.Stop();
  std::cout << "Hdot new way " << tt.time() << il::norm(y,il::Norm::L2) <<"\n";

  // using bigwhamio2
  const std::vector<double> properties = {elas.getE(), elas.getNu()}; // Young Modulus , Poisson's ratio

  const std::string kernel_name = "3DR0opening";
  bie:
  Bigwhamio testb{};
    // convert to std vectors
    std::vector<double> nodes_flat;
    nodes_flat.reserve(3 * coor.size(0));
    for (int i = 0; i < coor.size(0); i++){
      for (int j = 0; j < coor.size(1); j++){
        nodes_flat.push_back(coor(i,j));
      }
    }
    std::vector<int> conn_flat;
    conn_flat.reserve(3 * conn.size(0));
    for (int i = 0; i < conn.size(0); i++){
      for (int j = 0; j < conn.size(1); j++){
        conn_flat.push_back(conn(i,j));
      }
    }
    testb.set(nodes_flat,conn_flat,kernel_name,properties,
             static_cast<int>(max_leaf_size), eta, epsilon_aca);
    std::vector<double> xs;
    xs.reserve(xx.size());
    for (int i=0;i<xx.size();i++){
      xs.push_back(xx[i]);
    }
    tt.Start();
    std::vector<double> y4=testb.hdotProduct(xs);
    tt.Stop();
    for (int i=0;i<xx.size();i++){
      y[i]=y4[i];
    }
    std::cout << "Hdot new bigwhamio " << tt.time() << " E.x norm " << il::norm(y,il::Norm::L2) <<"\n";
    std::cout << " val "<< y4[0]<<  "\n";
    std::vector<double> val;
    std::vector<int> pos;
    testb.getFullBlocks(val,pos);
    for (il::int_t i=0;i<10;i++){
        std::cout << " val "<< val[i] << "  pos size"<< pos[2*i] << "-"<< pos[2*i+1] << "\n";
    }
    std::cout << " val "<< val[val.size()-1] << "  pos size"<< pos[2*(val.size()-1)] << "-"<< pos[2*(val.size()-1)+1] << "\n";
  return 0;
}
/////////////////////////////////////////////////////////////////////////////////


// TEST 3DT0 MATRIX BUILD ///////////////////////////////////////////////////////////////////////////////
int test3DT0_matrix_build(){

    std::cout << "-------------- test3DR0 Matrix-Build ---------------------\n";

    il::int_t nx, ny, k=0 ;
    nx = 301;
    ny = 301;
    double Lx = 0.3, Ly = 0.3, hx = 0., hy = 0.;

    hx = 2. * Lx / (nx - 1);
    hy = 2. * Ly / (ny - 1);

    il::Array2D<il::int_t> conn{nx * ny, 4};

    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            conn(k, 0) = (i + j * (nx + 1));
            conn(k, 1) = (i + 1) + j * (nx + 1);
            conn(k, 2) = i + 1 + (j + 1) * (nx + 1);
            conn(k, 3) = i + (j + 1) * (nx + 1);
            k = k + 1;
        }
    }

    il::Array2D<double> nodes{(nx + 1)*(ny + 1),3};
    il::Array<double> x{(nx + 1)}, y{(ny + 1)};

    for (int i = 0; i < (nx + 1); i++){
        x[i] = - Lx - hx/2. + i * hx/2.;
    }
    for (int i = 0; i < (ny + 1); i++){
        y[i] = - Ly - hy/2. + i * hy/2.;
    }

    int index = 0;
    for (int j = 0; j < (ny + 1); j++){
        for (int i = 0; i < (nx + 1); i++){
            nodes(index, 0) = x[i];
            nodes(index, 1) = y[j];
            nodes(index, 2) = 0.;
            index = index + 1;
        }
    }


    bie::Mesh3D mesh(nodes, conn, 0);

    double nu = 0.25;
    double G = 1.0;
    double young = 2.0 * G * (1.0+nu);

    // now we use the BIGWHAM

    // convert to std vectors
    std::vector<double> nodes_flat;
    nodes_flat.reserve(3 * nodes.size(0));
    for (int i = 0; i < nodes.size(0); i++){
        for (int j = 0; j < nodes.size(1); j++){
            nodes_flat.push_back(nodes(i,j));
        }
    }
    std::vector<int> conn_flat;
    conn_flat.reserve(3 * conn.size(0));
    for (int i = 0; i < conn.size(0); i++){
        for (int j = 0; j < conn.size(1); j++){
            conn_flat.push_back(conn(i,j));
        }
    }

    //std::cout << "Memory requirements " << "\n";
    //std::cout << "GiB =  " << 0.0383212 * nx * ny * nx * ny * 8. / 1024. / 1024. /1024.  << " "  << "\n";

    const std::vector<double> properties = {young, nu}; // Young Modulus , Poisson's ratio
    const int max_leaf_size = 450;
    const double eta_test = 3.;
    const double eps_aca = 0.0001;

    // create HMAT
    const std::string kernel_name = "3DR0opening";
    Bigwhamio test;
    test.set(nodes_flat,conn_flat,kernel_name,properties,
             max_leaf_size, eta_test, eps_aca);

    std::cout << "-------------- End of test3DT0 Matrix-Build ------------- " << "\n";
    return  0;
}
/////////////////////////////////////////////////////////////////////////////////


// TEST AXISYM 3DP0 Penny-Shaped Crack ///////////////////////////////////////////////////////////////////////////////

void testAxi3DP0() {

    std::cout << "-------------- test Axi3DP0 ---------------------\n";

    //// test influence factors

//double aux1 = elliptic_em(0.9);
//double aux2 = elliptic_em(0.5);
//double aux3 = elliptic_em(0.1);
//
//std::cout << " val = "<< aux1 << "\n";
//std::cout << " val = "<< aux2 << "\n";
//std::cout << " val = "<< aux3 << "\n";
//
//    aux1 = elliptic_fm(0.9);
//    aux2 = elliptic_fm(0.5);
//    aux3 = elliptic_fm(0.1);
//
//    std::cout << " val = "<< aux1 << "\n";
//    std::cout << " val = "<< aux2 << "\n";
//    std::cout << " val = "<< aux3 << "\n";

//    double IF1 = bie::stress_disk_dislocation(1.0,0.9);
//    double IF2 = bie::stress_disk_dislocation(6.0,6.5);
//    double IF3 = bie::stress_disk_dislocation(0.0001,0.0005);
//
//    std::cout << " val = "<< IF1 << "\n";
//    std::cout << " val = "<< IF2 << "\n";
//    std::cout << " val = "<< IF3 << "\n";

//    il::StaticArray2D<double, 2, 2> Xs;
//    Xs(0,0)=1.0;
//    Xs(0,1)=0.0;
//    Xs(1,0)=2.0;
//    Xs(1,1)=0.0;
//    bie::SegmentData src_el(Xs,0);
//
//    il::StaticArray2D<double, 2, 2> Xr;
//    Xr(0,0)=0.0;
//    Xr(0,1)=0.0;
//    Xr(1,0)=1.0;
//    Xr(1,1)=0.0;
//    bie::SegmentData rec_el(Xr,0);
//
//    bie::ElasticProperties elas(1.0,0.0);
//
//    il::StaticArray2D<double, 2, 2> traction = traction_influence_AxiSym3DP0(src_el, rec_el, elas);
//
//    std::cout << " e11 = "<< traction(0,0) << "\n";
//    std::cout << " e12 = "<< traction(0,1) << "\n";
//    std::cout << " e21 = "<< traction(1,0) << "\n";
//    std::cout << " e22 = "<< traction(1,1) << "\n";

//// test Hmat and penny-shaped crack

// simple 1D mesh
    il::int_t ne=140;

    il::Array2D<double> nodes{ne+1,2,0.};
    il::Array2D<il::int_t> conn{ne,2};

    for (il::int_t i=0;i<ne+1;i++){
        nodes(i,0)=1.0*i/ne;
        nodes(i,1)=0.;
    }

    for (il::int_t i=0;i<ne;i++){
        conn(i,0)=i;
        conn(i,1)=i+1;
    }
    bie::Mesh Mesh0(nodes,conn,0);

    il::Array2D<double> coll_points = Mesh0.getCollocationPoints();

    il::int_t leaf_size=12;
    il::Timer tt;
    tt.Start();
    const bie::Cluster cluster = bie::cluster(leaf_size, il::io, coll_points);

    const il::Tree<bie::SubHMatrix, 4> hmatrix_tree =
            bie::hmatrixTreeIxI(coll_points, cluster.partition, 10.0);
    tt.Stop();
    std::cout << "Time for cluster construction " << tt.time() <<"\n";
    tt.Reset();
    std::cout << cluster.partition.depth() <<"\n";

    double young = 1.0;
    double nu = 0.0;
    bie::ElasticProperties elas_aux(young,nu);

    bie::HMatrix<double> h_ ;
    const bie::ElasticHMatrixAxi3DP0<double> M{coll_points, cluster.permutation,
                                               Mesh0, elas_aux};

    std::cout << " create h mat " << M.size(0) <<"\n";
    h_= bie::toHMatrix(M, hmatrix_tree, 0.001);
    std::cout << " create h mat ended " << h_.isBuilt() <<"\n";
    std::cout << " compression ratio " << bie::compressionRatio(h_)<<"\n";

    double P = 1.0; // load
    double R = 1.0; // crack radius

    double coef = 8.0 * P / ( il::pi * young );

    // compute analytical solution at collocation points
    il::Array<double> wsol_coll{coll_points.size(0), 0.};

    for (int i = 0; i < coll_points.size(0); ++i) {
        if (std::abs(coll_points(i,0)) < R) {
            wsol_coll[i] = coef * sqrt(pow(R, 2) - pow(coll_points(i,0), 2));
        }
    }
    //at corresponding nodes - works here due to simple mesh....

    il::Array<double> xx{coll_points.size(0)*2};

    for (il::int_t i=0;i<xx.size()/2;i++){
        xx[2*i] = wsol_coll[i];
        xx[2*i+1] = wsol_coll[i];
    }

    il::Array< double> y=il::dot(h_,xx);

    for (il::int_t i=0;i<xx.size();i++){
        std::cout << y[i] <<"\n";
    }

    std::cout << " Now creating similar BigWham instance \n";
    std::string kernelname = "Axi3DP0";

    // now testing the Bigwhamio class...
    Bigwhamio testbie;
    //Bigwhamio *test = new Bigwhamio();


    std::vector<double> f_coor;
    f_coor.assign(2*Mesh0.numberOfNodes(),0.);
    for (il::int_t i=0;i<Mesh0.numberOfNodes();i++){
        f_coor[2*i]=Mesh0.coordinates(i,0);
        f_coor[2*i+1]=Mesh0.coordinates(i,1);
    }

    std::vector<int> f_conn;
    f_conn.assign(2*Mesh0.numberOfElts(),0);
    for (il::int_t i=0;i<Mesh0.numberOfElts();i++){
        f_conn[2*i]=Mesh0.connectivity(i,0);
        f_conn[2*i+1]=Mesh0.connectivity(i,1);
    }

    std::vector<double> f_prop;
    f_prop.assign(2,0);
    f_prop[0]=elas_aux.getE();
    f_prop[1]=elas_aux.getNu();

    std::cout << " now setting things in bigwhamio obj \n";

    testbie.set(f_coor,f_conn,kernelname,f_prop,12,10,0.001);

    std::cout        <<  " C R :"<< testbie.getCompressionRatio() <<"\n";

    std::vector<double> x;
    x.assign(xx.size(),0.);
    for (il::int_t i=0;i<xx.size();i++){
        x[i]=xx[i];
    }

    std::vector<double> y2=testbie.hdotProduct(x);
    std::cout << " elastic dot product solution:: \n";
    for (il::int_t i=0;i<coll_points.size(0);i++){
        std::cout <<" y " << i  <<" " << y[2*i] << " - with obj - " << y2[2*i]  <<"\n";
        std::cout <<" y " << i  <<" " << y[2*i+1] << " - with obj - " << y2[2*i+1] <<"\n";
    }

    std::cout << "----------end of Axi3DP0 test ---------------------\n";

}


/////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////

int main() {

  std::cout << "++++++++++++++++++++\n";

  //int a = check3DR0();   // this gives an error - Can't open the file - make sure that test in the main are short and can be run when pushed.

  //test3DR0();

  test3DR0modes2and3();

  test3DT0modes2and3();

  //perf3DR0();

  //test2DP1();

  //testS3DP0();

  testAxi3DP0();

//std::cout << "Mahcine epsilon " << std::numeric_limits<double>::epsilon() <<"\n";

  //testFullMat();

  //testNewHmat();

  //test3DT0_matrix_build();

  //testPl3D();

  //// tests for 3DT6 not updated since the change of interface
  //test3DT6Mesh();

   //std::string vertices_file = "/home/alexis/bigwham/vertices5000.csv";
   //std::string connectivity_file = "/home/alexis/bigwham/connectivity5000.csv";
   //test3DT6PennyShaped(vertices_file,connectivity_file);
   //test3DT6_PennyShaped(vertices_file,connectivity_file);

   //test3DT0();

   //// note that here you need to pass the csv files for the mesh (vertices and connectivity)
   //std::string vertices_file = "/Users/alexis/BigWhamLink/vertices.csv";
   //std::string connectivity_file = "/Users/alexis/BigWhamLink/conn.csv";
   //test3DT0_PennyShaped(vertices_file,connectivity_file);

  std::cout << "\n End of BigWham - exe " << "\n";

}

/////////////////////////////////////////////////////////////////////////////////
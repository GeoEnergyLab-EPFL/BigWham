//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 24.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland, Geo-Energy Laboratory, 2016-2023.  All rights reserved.
// See the LICENSE.TXT file for more details.
//

#pragma once
#include <gtest/gtest.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/math.h>

#include <src/core/BoundaryElement.h>
#include <src/core/BEMesh.h>
#include <src/core/SquareMatrixGenerator.h>
#include <src/hmat/cluster/cluster.h>
#include <src/elasticity/2d/BIE_elastostatic_segment_0_impls.h>
#include <src/elasticity/2d/BIE_elastostatic_segment_1_impls.h>
#include <src/elasticity/3d/BIE_elastostatic_triangle_0_impls.h>

#include <src/elasticity/2d/ElasticS3DP0_element.h>
#include <src/hmat/hmatrix/Hmat.h>

TEST(SquareMatGen,segment_0_1){
     /// single element
    il::Array2D<double> xy{2,2,0.};
    xy(1,0)=1.0;
    il::Array2D<il::int_t> conn{1,2,0};
    conn(0,1)=1;
    bie::BEMesh<bie::Segment<0>> my_mesh(xy, conn);
    bie::Segment<0> source;
//    source.setElement(xy);
    bie::ElasticProperties elas(1,0.3);
    bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>  test(elas,xy.size(1));
    il::Array<il::int_t> permutation{1,0};
    bie::SquareMatrixGenerator<double,bie::Segment<0>,bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                                          test, permutation);
    ASSERT_TRUE(M.size(1)==2 && M.size(0)==2 );
}

TEST(SquareMatGen,segment_0_2){
    /// single element
    il::Array2D<double> xy{2,2,0.};
    xy(1,0)=1.0;
    il::Array2D<il::int_t> conn{1,2,0};
    conn(0,1)=1;
    bie::BEMesh<bie::Segment<0>> my_mesh(xy, conn);
    bie::Segment<0> source;
//    source.setElement(xy);
    bie::ElasticProperties elas(1,0.3);
    bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>  test(elas,xy.size(1));
    il::Array<il::int_t> permutation{1,0};
    bie::SquareMatrixGenerator<double,bie::Segment<0>,bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                                          test, permutation);
    ASSERT_TRUE(M.blockSize()==2 && M.sizeAsBlocks(0)==my_mesh.numberOfElts() );
}


TEST(SquareMatGen,segment_0_3){
    /// single element
    il::Array2D<double> xy{2,2,0.};
    xy(1,0)=1.0;
    il::Array2D<il::int_t> conn{1,2,0};
    conn(0,1)=1;
    bie::BEMesh<bie::Segment<0>> my_mesh(xy, conn);
    bie::Segment<0> source;
//    source.setElement(xy);
    bie::ElasticProperties elas(1,0.3);
    bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>  test(elas,xy.size(1));
    il::Array<double> prop{1,1000.};
    test.setKernelProperties(prop);
    il::Array<il::int_t> permutation{1,0};
    bie::SquareMatrixGenerator<double,bie::Segment<0>,bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                                          test, permutation);
    il::Array2D<double> A{M.size(0),M.size(1),0.0};
    il::Array2DEdit<double> v=A.Edit();
    M.set(0,0,il::io,v);
    // check with known values of entry for that case
// 0.34979115367667662
//0.34979125861394933
//    std::cout << v(0,0) << "-" << v(0,1) <<"\n";
 //   std::cout << v(1,0) << "-" << v(1,1) <<"\n";
    double eps=1e-6;
    ASSERT_TRUE( (abs(v(0,0)-0.34979115367667662)<eps) && (abs(v(0,1)-0.0)<eps) && (abs(v(1,0)-0.0)<eps) && (abs(v(1,1)-0.34979125861394933)<eps) );
}

//
TEST(SquareMatGen,segment_0_Hmat_1){
    ///  simple mesh
    int n_elts=88;
    il::Array2D<double> coor{n_elts+1,2,0.};

    il::Array2D<il::int_t> conn{n_elts,2};
    double h=0.123;
    for (int i=0;i<n_elts+1;i++) {
        coor(i,0)=i*h;
    }
    for (int i=0;i<n_elts;i++){
        conn(i,0)= i;
        conn(i,1)=i+1;
    }
    bie::BEMesh<bie::Segment<0>> my_mesh(coor, conn);
    il::Array2D<double> xcol=my_mesh.getCollocationPoints();
    bie::ElasticProperties elas(1,0.3);
    bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>  ker(elas,coor.size(1));
    il::Array<double> prop{1,1000.};
    ker.setKernelProperties(prop);

    il::int_t max_leaf_size=32;
    bie::HRepresentation hr=bie::h_representation_square_matrix(my_mesh,max_leaf_size,1.0);
    bie::SquareMatrixGenerator<double,bie::Segment<0>,bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                                          ker,hr.permutation_0_);
    bie::Hmat<double>  h_(M,hr,1.e-3);
    ASSERT_TRUE( h_.isBuilt() );//h_.isBuilt()
}


TEST(SquareMatGen,segment_0_Hmat_2){
    ///  simple mesh
    int n_elts=1200;
    il::Array2D<double> coor{n_elts+1,2,0.};

    il::Array2D<il::int_t> conn{n_elts,2};
    double L=1.;double h=2.*L/n_elts;
    for (int i=0;i<n_elts+1;i++) {
        coor(i,0)=i*h-L;
    }
    for (int i=0;i<n_elts;i++){
        conn(i,0)= i;
        conn(i,1)=i+1;
    }
    bie::BEMesh<bie::Segment<0>> my_mesh(coor, conn);
    il::Array2D<double> xcol=my_mesh.getCollocationPoints();
    bie::ElasticProperties elas(1,0.0);
    bie::BIE_elastostatic<bie::Segment<0>,bie::Segment<0>,bie::ElasticKernelType::H>  ker(elas,coor.size(1));
    il::Array<double> prop{1,1000.};
    ker.setKernelProperties(prop);
    il::int_t max_leaf_size=32;double eta=2.0;
    bie::HRepresentation hr=bie::h_representation_square_matrix(my_mesh,max_leaf_size,eta);
    using el_type = bie::Segment<0>;
    bie::SquareMatrixGenerator<double,el_type,bie::BIE_elastostatic<el_type,el_type,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                  ker,hr.permutation_0_);
    double eps_aca=1.e-3;
    bie::Hmat<double>  h_(M,hr,eps_aca);
    //simple opening mode...
    il::Array<double> x{M.size(1),0.0},y{M.size(1),0.0};
    for(il::int_t i=0;i<M.sizeAsBlocks(0);i++){
        x[2*hr.permutation_0_[i]+1]=4.0*sqrt(L*L-xcol(i,0)*xcol(i,0) );
    }
    y=h_.matvec(x);
    il::Array<double> rel_err{M.sizeAsBlocks(0),0.};
    for (il::int_t i=0;i<M.sizeAsBlocks(0);i++){
        rel_err[i]=sqrt((y[2*i+1]-1.)*(y[2*i+1]-1.));
       //std::cout << "rel x: " << rel_err[i] << "\n";
    }
   // std::cout << "Linf rel error " << il::norm(rel_err,il::Norm::Linf) <<"\n";
    //std::cout << "L2 rel error " << il::norm(rel_err,il::Norm::L2) <<"\n";
    std::cout << "Mean rel error " << il::mean(rel_err) <<"\n";
    ASSERT_TRUE( il::mean(rel_err)<0.05 );//h_.isBuilt()
}


/// test for Segment P1
TEST(SquareMatGen,segment_1_Hmat_1_segs_45_a1) {
    // two segments at 90 degree from one another
    // oriented 45 degree from the axis e_x
    //
    //        /\        //

    il::Array2D<double> xy{3, 2, 0.};
    xy(0, 0) = -1.;
    xy(0, 1) = 0.;
    xy(1, 0) = 0.;
    xy(1, 1) = 1.;
    xy(2, 0) = 1.;
    xy(2, 1) = 0.;

    il::Array2D<il::int_t> ien{2, 2, 0};
    ien(0, 0) = 0;
    ien(0, 1) = 1;
    ien(1, 0) = 1;
    ien(1, 1) = 2;

    bie::BEMesh<bie::Segment<1>> mesh(xy, ien);
    bie::ElasticProperties elas(1., 0.);
    bie::BIE_elastostatic<bie::Segment<1>,bie::Segment<1>,bie::ElasticKernelType::H>  ker(elas,xy.size(1));
    il::int_t max_leaf_size=320;double eta=0.0;
    bie::HRepresentation hr=bie::h_representation_square_matrix(mesh,max_leaf_size,eta);
    using el_type = bie::Segment<1>;
    bie::SquareMatrixGenerator<double,el_type,bie::BIE_elastostatic<el_type,el_type,bie::ElasticKernelType::H>> M(mesh,
                                                                                                                  ker,hr.permutation_0_);
    double eps_aca=1.e-3;
    bie::Hmat<double>  h_(M,hr,eps_aca);
/// analytical results from mathematica integration for that particular case
    // we compare the results of the assembly w.t the mma code
    il::Array2D<double> Kmma{8, 8, 0.};
    ;
    Kmma(0, 0) = 0.483423;
    Kmma(0, 1) = -1.20657e-16;
    Kmma(0, 2) = -0.0332652;
    Kmma(0, 3) = 2.07014e-17;
    Kmma(0, 4) = 0.00824514;
    Kmma(0, 5) = -0.0381383;
    Kmma(0, 6) = -0.0133572;
    Kmma(0, 7) = -0.0321492;

    Kmma(1, 0) = -2.27998e-16;
    Kmma(1, 1) = 0.483423;
    Kmma(1, 2) = 2.80878e-17;
    Kmma(1, 3) = -0.0332652;
    Kmma(1, 4) = -0.00355156;
    Kmma(1, 5) = -0.00824514;
    Kmma(1, 6) = 0.00954068;
    Kmma(1, 7) = 0.0133572;

    Kmma(2, 0) = -0.0332652;
    Kmma(2, 1) = 1.4713e-33;
    Kmma(2, 2) = 0.483423;
    Kmma(2, 3) = 0.;
    Kmma(2, 4) = -0.0536082;
    Kmma(2, 5) = -0.376167;
    Kmma(2, 6) = 0.000833234;
    Kmma(2, 7) = -0.0157962;

    Kmma(3, 0) = 7.38637e-18;
    Kmma(3, 1) = -0.0332652;
    Kmma(3, 2) = -1.07342e-16;
    Kmma(3, 3) = 0.483423;
    Kmma(3, 4) = 0.23189;
    Kmma(3, 5) = 0.0536082;
    Kmma(3, 6) = 0.128481;
    Kmma(3, 7) = -0.000833234;

    Kmma(4, 0) = 0.000833234;
    Kmma(4, 1) = 0.0157962;
    Kmma(4, 2) = -0.0536082;
    Kmma(4, 3) = 0.376167;
    Kmma(4, 4) = 0.483423;
    Kmma(4, 5) = 0.;
    Kmma(4, 6) = -0.0332652;
    Kmma(4, 7) = 0.;

    Kmma(5, 0) = -0.128481;
    Kmma(5, 1) = -0.000833234;
    Kmma(5, 2) = -0.23189;
    Kmma(5, 3) = 0.0536082;
    Kmma(5, 4) = 1.07342e-16;
    Kmma(5, 5) = 0.483423;
    Kmma(5, 6) = -7.38637e-18;
    Kmma(5, 7) = -0.0332652;

    Kmma(6, 0) = -0.0133572;
    Kmma(6, 1) = 0.0321492;
    Kmma(6, 2) = 0.00824514;
    Kmma(6, 3) = 0.0381383;
    Kmma(6, 4) = -0.0332652;
    Kmma(6, 5) = -2.07014e-17;
    Kmma(6, 6) = 0.483423;
    Kmma(6, 7) = 1.20657e-16;

    Kmma(7, 0) = -0.00954068;
    Kmma(7, 1) = 0.0133572;
    Kmma(7, 2) = 0.00355156;
    Kmma(7, 3) = -0.00824514;
    Kmma(7, 4) = -2.80878e-17;
    Kmma(7, 5) = -0.0332652;
    Kmma(7, 6) = 2.27998e-16;
    Kmma(7, 7) = 0.483423;

    const il::Array<il::int_t> permutation=hr.permutation_0_;
    il::Array<double> val_list;il::Array<int> pos_list;
    h_.fullBlocksOriginal(permutation,il::io,val_list,pos_list);

    double my_sum = 0.;
    int k=0;
    for (il::int_t j = 0.; j < Kmma.size(1); j++) {
        for (il::int_t i = 0; i < Kmma.size(0); i++) {
            my_sum += abs(val_list[k] - Kmma(i, j));
            k++;
        }
    }

    std::cout << "SUMMA: " <<my_sum << "\n";

    ASSERT_NEAR(my_sum, 0., 1.e-5);

 //   ASSERT_TRUE(M.sizeAsBlocks(0)==mesh.numberCollocationPoints() && M.size(0)==8 && M.size(1)==8);

}


TEST(SquareMatGen,segment_1_Hmat_1_two_adjacent_segs) {
// two adjacents straight segments
    // just one DD is mobilised
    //        ._._.        //

    il::Array2D<double> xy{3, 2, 0.};
    xy(0, 0) = -1.;
    xy(0, 1) = 0.;
    xy(1, 0) = 0.;
    xy(1, 1) = 0.;
    xy(2, 0) = 1.;
    xy(2, 1) = 0.;

    il::Array2D<il::int_t> ien{2, 2, 0};
    ien(0, 0) = 0;
    ien(0, 1) = 1;
    ien(1, 0) = 1;
    ien(1, 1) = 2;


    bie::BEMesh<bie::Segment<1>> mesh(xy, ien);
    bie::ElasticProperties elas(1., 0.);
    bie::BIE_elastostatic<bie::Segment<1>,bie::Segment<1>,bie::ElasticKernelType::H>  ker(elas,xy.size(1));
    il::int_t max_leaf_size=320;double eta=0.0;
    bie::HRepresentation hr=bie::h_representation_square_matrix(mesh,max_leaf_size,eta);
    using el_type = bie::Segment<1>;
    bie::SquareMatrixGenerator<double,el_type,bie::BIE_elastostatic<el_type,el_type,bie::ElasticKernelType::H>> M(mesh,
                                                                                                                  ker,hr.permutation_0_);
    double eps_aca=1.e-3;
    bie::Hmat<double>  h_(M,hr,eps_aca);
/// analytical results from mathematica integration for that particular case
    // we compare the results of the assembly w.t the mma code
    il::Array2D<double> Kmma{8, 8, 0.};
    ;
    Kmma(0, 0) = -0.683664;
    Kmma(0, 1) = 0.;
    Kmma(0, 2) = 0.0470442;
    Kmma(0, 3) = 0.;
    Kmma(0, 4) = 0.0943392;
    Kmma(0, 5) = 0.;
    Kmma(0, 6) = 0.0187761;
    Kmma(0, 7) = 0.;

    Kmma(1, 0) = 0.;
    Kmma(1, 1) = -0.683664;
    Kmma(1, 2) = 0.;
    Kmma(1, 3) = 0.0470442;
    Kmma(1, 4) = 0.;
    Kmma(1, 5) = 0.0943392;
    Kmma(1, 6) = 0.;
    Kmma(1, 7) = 0.0187761;

    Kmma(2, 0) = 0.0470442;
    Kmma(2, 1) = 0.;
    Kmma(2, 2) = -0.683664;
    Kmma(2, 3) = 0.;
    Kmma(2, 4) = 0.379637;
    Kmma(2, 5) = 0.;
    Kmma(2, 6) = 0.0315223;
    Kmma(2, 7) = 0.;

    Kmma(3, 0) = 0.;
    Kmma(3, 1) = 0.0470442;
    Kmma(3, 2) = 0.;
    Kmma(3, 3) = -0.683664;
    Kmma(3, 4) = 0.;
    Kmma(3, 5) = 0.379637;
    Kmma(3, 6) = 0.;
    Kmma(3, 7) = 0.0315223;

    Kmma(4, 0) = 0.0315223;
    Kmma(4, 1) = 0.;
    Kmma(4, 2) = 0.379637;
    Kmma(4, 3) = 0.;
    Kmma(4, 4) = -0.683664;
    Kmma(4, 5) = 0.;
    Kmma(4, 6) = 0.0470442;
    Kmma(4, 7) = 0.;

    Kmma(5, 0) = 0.;
    Kmma(5, 1) = 0.0315223;
    Kmma(5, 2) = 0.;
    Kmma(5, 3) = 0.379637;
    Kmma(5, 4) = 0.;
    Kmma(5, 5) = -0.683664;
    Kmma(5, 6) = 0.;
    Kmma(5, 7) = 0.0470442;

    Kmma(6, 0) = 0.0187761;
    Kmma(6, 1) = 0.;
    Kmma(6, 2) = 0.0943392;
    Kmma(6, 3) = 0.;
    Kmma(6, 4) = 0.0470442;
    Kmma(6, 5) = 0.;
    Kmma(6, 6) = -0.683664;
    Kmma(6, 7) = 0.;

    Kmma(7, 0) = 0.;
    Kmma(7, 1) = 0.0187761;
    Kmma(7, 2) = 0.;
    Kmma(7, 3) = 0.0943392;
    Kmma(7, 4) = 0.;
    Kmma(7, 5) = 0.0470442;
    Kmma(7, 6) = 0.;
    Kmma(7, 7) = -0.683664;

    const il::Array<il::int_t> permutation=hr.permutation_0_;
    il::Array<double> val_list;il::Array<int> pos_list;
    h_.fullBlocksOriginal(permutation,il::io,val_list,pos_list);

    double my_sum = 0.;
    int k=0;
    for (il::int_t j = 0.; j < Kmma.size(1); j++) {
        for (il::int_t i = 0; i < Kmma.size(0); i++) {
            my_sum +=  val_list[k] + Kmma(i, j) ;   // note the change here
            k++;
        }
    }

    std::cout << "SUMMA: " <<my_sum << "\n";

    ASSERT_NEAR(my_sum, 0., 1.e-5);
}

TEST(SquareMatGen,Triangle_0_1) {
    // 2 triangles on the same plane making a square of size 1 by 1
    //
    il::Array2D<double> coor{4,3,0.};

    coor(1,0)=1.;
    coor(2,1)=1.;
    coor(3,0)=1.;coor(3,1)=1.;

    il::Array2D<il::int_t> conn{2,3,0};
    conn(0,1)=1; conn(0,2)=2;
    conn(1,0)=2; conn(1,1)=1; conn(1,2)=3;

    using El_type = bie::Triangle<0>;
    bie::BEMesh<El_type> my_mesh(coor, conn);
    il::Array2D<double> xcol=my_mesh.getCollocationPoints();
    bie::ElasticProperties elas(1,0.0);
    bie::BIE_elastostatic<El_type,El_type,bie::ElasticKernelType::H>  ker(elas,coor.size(1));
    il::Array<double> prop{1,1000.};
    ker.setKernelProperties(prop);
    il::int_t max_leaf_size=32;double eta=2.0;
    bie::HRepresentation hr=bie::h_representation_square_matrix(my_mesh,max_leaf_size,eta);

    bie::SquareMatrixGenerator<double,El_type,bie::BIE_elastostatic<El_type,El_type,bie::ElasticKernelType::H>> M(my_mesh,
                                                                                                                  ker,hr.permutation_0_);
    double eps_aca=1.e-3;
    bie::Hmat<double>  h_(M,hr,eps_aca);
    const il::Array<il::int_t> permutation=hr.permutation_0_;
    il::Array<double> val_list;il::Array<int> pos_list;
    h_.fullBlocksOriginal(permutation,il::io,val_list,pos_list);
    for (int i=0;i<9;i++){
        std::cout << val_list[i] <<"\n";
    }
    ASSERT_TRUE(h_.isBuilt());
    //simple opening mode...
//    il::Array<double> x{M.size(1),0.0},y{M.size(1),0.0};
//    for(il::int_t i=0;i<M.sizeAsBlocks(0);i++){
//        x[2*hr.permutation_0_[i]+1]=4.0*sqrt(L*L-xcol(i,0)*xcol(i,0) );
//    }
//    y=h_.matvec(x);
//    il::Array<double> rel_err{M.sizeAsBlocks(0),0.};
//    for (il::int_t i=0;i<M.sizeAsBlocks(0);i++){
//        rel_err[i]=sqrt((y[2*i+1]-1.)*(y[2*i+1]-1.));
//        //std::cout << "rel x: " << rel_err[i] << "\n";
//    }


}
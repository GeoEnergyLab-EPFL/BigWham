//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 22.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2020.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#include <iostream>
#include <string>

#include "io/bigwham_io_gen.h"

/////////////////////////////////////////////////////////////////////////////////
int main() {
    // create a simple mesh for a griffith crack -
    // use the bigwhamio interface.
    ///  simple mesh
    // check diag

    // int n_elts=1200;
    // std::vector<double> coor(2*(n_elts+1),0.);
    // double L=1.;double h=2.*L/n_elts;
    // int k=0;
    // for (int i=0;i<n_elts+1;i++) {
    //     coor[k]=i*h-L;
    //     k=k+2;
    // }
    // std::vector<int> conn(n_elts*2,0.);
    // k=0;
    // for (int i=0;i<n_elts;i++){
    //     conn[k]= i;conn[k+1]=i+1;
    //     k=k+2;
    // }

    // Bigwhamio my_io;
    // std::vector<double> properties{1.,0.,100};
    // my_io.set(coor,conn,"S3DP0",properties,32,2,1.e-3);

    // std::vector<double> x(my_io.matrixSize(1),0.);
    // for(il::int_t i=0;i<n_elts ;i++){
    //     x[2*i+1]=4.0*sqrt(L*L-coor[2*i]*coor[2*i]);
    // }
    // auto y= my_io.matvect(x);
    // std::vector<double> the_diag(n_elts*2,0.);
    // my_io.getDiagonal(the_diag);

    // il::Array<double> rel_err{n_elts,0.};
    // for (il::int_t i=0;i<n_elts;i++){
    //     rel_err[i]=sqrt((the_diag[2*i+1]-190.985)*(the_diag[2*i+1]-190.985));
    // }
    // std::cout << "Mean rel error (using biwghamio) " << il::mean(rel_err) <<"\n";

    // std::cout << "\n End of BigWham - exe "   << "\n";
}

/////////////////////////////////////////////////////////////////////////////////

// This file is part of BigWham.
//
// Created by Brice Lecampion on 15.12.19.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#ifndef BIE_SEGMENTDATA_H
#define BIE_SEGMENTDATA_H

#include <cmath>
#include <iostream>

#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/Array2D.h>
#include <il/linearAlgebra.h>


namespace bie {

//   Rotation Matrix
il::StaticArray2D<double, 2, 2> rotationMatrix2D(double theta);

inline il::StaticArray2D<double, 2, 2> rotationMatrix2D(double theta) {
  il::StaticArray2D<double, 2, 2> R;

  R(0, 0) = cos(1. * theta);
  R(0, 1) = -1. * sin(1. * theta);
  R(1, 0) = sin(theta);
  R(1, 1) = cos(theta);

  return R;
}


class SegmentData {

 private:
  double size_;
  double theta_;  // angle w.r. to e_1 (Ox)
  // unit normal to segment in global system of coordinates
  il::StaticArray<double, 2> n_;
  // unit tangent to segment in global system of coordinates
  il::StaticArray<double, 2> s_;
  // segment mid points coordinates.
  il::StaticArray<double, 2> Xmid_;
  // collocation points in global system of coordinates
  il::Array2D<double> CollocationPoints_;

 public:

  //////////////////////////////////////////////////////////////////////////////
  // constructor from segment end point coordinates matrix.
  SegmentData(il::StaticArray2D<double, 2, 2>  Xs, il::int_t p) {

    // compute element size
    il::StaticArray<double, 2> xdiff, s, n, xmean, xaux;
    il::Array2D<double> Xcol{p + 1, 2, 0};
    il::StaticArray2D<double, 2, 2> R;

    xdiff[0] = Xs(1, 0) - Xs(0, 0);
    xdiff[1] = Xs(1, 1) - Xs(0, 1);

//    // Order the segment data such that the seg tangent vector is pointing 'to the right'
//    // (from -90+\epsilon to +90)
//    if (xdiff[0] <= 0.) {
//      if (xdiff[0] < 0.) {
//      xdiff[0] = -xdiff[0]; xdiff[1] = -xdiff[1];
//      } else {
//        if (xdiff[1] < 0){
//          xdiff[1] = -xdiff[1];
//        }
//      }
//    };

    theta_ = std::atan2(xdiff[1], xdiff[0]);

    size_ = sqrt(xdiff[0] * xdiff[0] + xdiff[1] * xdiff[1]);

    // s & n provide a direct orthonormal frame with s oriented from node 1 to node 2

    // tangent vector s
    s[0] = xdiff[0] / size_;
    s[1] = xdiff[1] / size_;
    // normal vector n
    n[0] = -1. * s[1];
    n[1] = s[0];

    s_ = s;
    n_ = n;

    // mid point of the element
    xmean[0] = (Xs(1, 0) + Xs(0, 0)) / 2.;
    xmean[1] = (Xs(1, 1) + Xs(0, 1)) / 2.;
    Xmid_ = xmean;

    switch (p) {
      case 1: {  // linear DD
        Xcol(0, 0) = -1. / sqrt(2.);
        Xcol(0, 1) = 0.;
        Xcol(1, 0) = 1. / sqrt(2.);
        Xcol(1, 1) = 0.;
      };
        break;

      case 0: {
        Xcol(0, 0) = 0.;
        Xcol(0, 1) = 0.;
      };
        break;
      default:std::cout << "error\n";  //  error
        break;
    };

// Returning the collocation point in the global frame

    R = bie::rotationMatrix2D(theta_);

    for (int i = 0; i < p + 1; ++i) {

      xaux[0] = (size_) * Xcol(i, 0) / 2.;
      xaux[1] = (size_) * Xcol(i, 1) / 2.;

      xaux = il::dot(R, xaux);

      Xcol(i, 0) = xaux[0] + xmean[0];
      Xcol(i, 1) = xaux[1] + xmean[1];

    }

    CollocationPoints_ = Xcol;

  }
  //////////////////////////////////////////////////////////////////////////////
  //
  // get functions
  double size() const { return size_;};
  double theta() const { return theta_;}
  il::StaticArray<double,2> n() const {return n_;};
  il::StaticArray<double,2> s() const {return s_;};
  il::StaticArray<double, 2> Xmid() const { return Xmid_;};
  il::Array2D<double> CollocationPoints() const { return CollocationPoints_;};

  double CollocationPoints(il::int_t i, il::int_t j) const { return CollocationPoints_(i,j);};
  double Xmid(il::int_t i) const { return Xmid_[i];};

};


}
#endif

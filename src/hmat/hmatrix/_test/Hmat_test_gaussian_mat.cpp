//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 26.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2023.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
#include <gtest/gtest.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/math.h>
#include <memory>
//
#include "hmat/cluster/cluster.h"
#include "hmat/hierarchical_representation.h"
#include "hmat/hmatrix/Hmat.h"
#include "hmat/hmatrix/h_pattern.h"

#include "hmat/arrayFunctor/GaussianMatrix.h"

TEST(hmat, gaussian_1) {
  il::int_t n = 100;
  double beta = 1.0;
  il::Array2D<double> points{n, 1, 0.};
  for (il::int_t i = 0; i < n; i++) {
    points(i, 0) = 1.0 * i;
  }
  bie::Cluster cluster = bie::cluster(32, il::io, points);
  const il::Tree<bie::SubHMatrix, 4> block_tree =
      bie::hmatrixTreeIxI(points, cluster.partition, 3.);
  auto hr = std::make_shared<bie::HRepresentation>();
  hr->pattern_ = bie::createPattern(block_tree);
  hr->permutation_0_ = cluster.permutation;
  bie::GaussianMatrix<double> M{n, beta, hr};
  bie::Hmat<double> h_(M, 1.e-3);
  std::cout << " n FB " << hr->pattern_.n_FRB << "\n";
  std::cout << " Size " << M.size(0) << " -" << M.size(1) << "\n";
  std::cout << " Compression " << h_.compressionRatio() << "\n";
  ASSERT_TRUE(h_.isBuilt());
}

TEST(hmat, gaussian_hmat_diag) {
  // checking the function to get the diagonal of Gaussian Matrix
  il::int_t n = 150;
  double alpha = 10.0;
  il::Array2D<double> points{n, 1, 0.};
  for (il::int_t i = 0; i < n; i++) {
    points(i, 0) = 1.0 * i;
  }
  bie::Cluster cluster = bie::cluster(32, il::io, points);
  const il::Tree<bie::SubHMatrix, 4> block_tree =
      bie::hmatrixTreeIxI(points, cluster.partition, 3.);
  auto hr = std::make_shared<bie::HRepresentation>();
  hr->pattern_ = bie::createPattern(block_tree);
  hr->permutation_0_ = cluster.permutation;
  bie::GaussianMatrix<double> M{n, alpha, hr};
  bie::Hmat<double> h_(M, 1.e-3);
  std::cout << " n FB " << hr->pattern_.n_FRB << "\n";
  std::cout << " Size " << M.size(0) << " -" << M.size(1) << "\n";
  std::cout << " Compression " << h_.compressionRatio() << "\n";
  std::vector<double> diag = h_.diagonal();
  double val = 0.;
  for (il::int_t i = 0; i < M.size(0); i++) {
    val = val + (diag[i]);
    //      //  std::cout << " d " << (diag[i]==1.) <<"\n";
  }
  ASSERT_TRUE(val == (1.0 * n));
}

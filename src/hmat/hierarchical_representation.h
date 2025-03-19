//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 26.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne), Switzerland,
// Geo-Energy Laboratory, 2016-2025.  All rights reserved. See the LICENSE.TXT
// file for more details.
//

#pragma once

#ifndef BIGWHAM_HIERARCHICAL_REPRESENTATION_H
#define BIGWHAM_HIERARCHICAL_REPRESENTATION_H

#include <memory>

#include <il/Array.h>
#include <il/Timer.h>

#include "hmat/cluster/cluster.h"
#include "hmat/hmatrix/h_pattern.h"
#include "core/mesh.h"

namespace bigwham {

struct HRepresentation {
  HPattern pattern_;
  il::Array<il::int_t> permutation_0_; // for rows
  il::Array<il::int_t> permutation_1_; // for columns
  bool is_square_ =true; // if true - square matrix, and only permutation_0_ is stored -> not used ?
  
  HRepresentation() = default;
  ~HRepresentation() = default;
};

inline std::shared_ptr<HRepresentation>
HRepresentationSquareMatrix(const std::shared_ptr<Mesh> &mesh,
                            const il::int_t max_leaf_size, const double eta, const bool homegeneous_size = false) {
  // std::cout << "Pattern construction started .... \n";
  auto hr = std::make_shared<HRepresentation>();
  hr->is_square_ = true;
  // creation of the cluster
  // first get all collocation points in the mesh
  il::Array2D<double> Xcol = mesh->collocation_points();
  // std::cout << " Got col points construction ...."<< "\n";
  il::Timer tt;
  tt.Start();
  Cluster cluster = bigwham::cluster(max_leaf_size, il::io, Xcol, homegeneous_size);
  std::cout << "Cluster tree creation time :  " << tt.time() << "\n";
  tt.Stop();
  tt.Reset();
  hr->permutation_0_ = cluster.permutation; // todo: make a separate routine for cluster creation ... to be interfaced to python (for domain decomposition for MPI rect Hmat)
  hr->permutation_1_ = cluster.permutation;

  tt.Start();
  const il::Tree<SubHMatrix, 4> block_tree =
      hmatrixTreeIxI(Xcol, cluster.partition, eta, homegeneous_size);
  tt.Stop();
  std::cout << "Time for binary cluster tree construction  " << tt.time() << "\n";
  std::cout << "Binary cluster tree depth = " << block_tree.depth() << "\n";
  hr->pattern_ = createPattern(block_tree);

  std::cout << "Number of blocks = " << hr->pattern_.n_B << "\n";
  std::cout << "Number of full blocks = " << hr->pattern_.n_FRB << "\n";
  std::cout << "Number of low rank blocks = " << hr->pattern_.n_LRB << "\n";
  std::cout << "Pattern Created \n";

  return hr;
}

inline std::shared_ptr<HRepresentation>
HRepresentationRectangularMatrix(const std::shared_ptr<Mesh> &source_mesh,
                                 const std::shared_ptr<Mesh> &receiver_mesh,
                                 const il::int_t max_leaf_size,
                                 const double eta,
                                  bool verbose,
                                const bool homegeneous_size = false) {
  auto hr = std::make_shared<HRepresentation>();
  hr->is_square_ = false;
  // creation of the cluster
  // first get all collocation points in the mesh
  il::Timer tt;
  il::Array2D<double> Xcol_source = source_mesh->collocation_points();
  il::Array2D<double> Xcol_receiver = receiver_mesh->collocation_points();

  tt.Start();
  Cluster cluster_s = cluster(max_leaf_size, il::io, Xcol_source, homegeneous_size);
  if (verbose)
    std::cout << "Cluster tree creation time for the source mesh :  " << tt.time() << "\n";
  tt.Stop();
  tt.Reset();
  hr->permutation_1_ = cluster_s.permutation; // sources permutation

  tt.Start();
  Cluster cluster_r = cluster(max_leaf_size, il::io, Xcol_receiver, homegeneous_size);
  if (verbose)
    std::cout << "Cluster tree creation time for the receiver mesh :  " << tt.time() << "\n";
  tt.Stop();
  tt.Reset();
  hr->permutation_0_ = cluster_r.permutation; // receivers permutation

  tt.Start();
  const il::Tree<SubHMatrix, 4> block_tree =
      hmatrixTreeIxJ(Xcol_receiver, cluster_r.partition, Xcol_source,
                     cluster_s.partition, eta, homegeneous_size);
  tt.Stop();
  if (verbose){
    std::cout << "Time for binary cluster tree construction  " << tt.time() << "\n";
    std::cout << " binary cluster tree depth =" << block_tree.depth() << "\n";
  }
  
  hr->pattern_ = createPattern(block_tree);
  //        hr.pattern_.nr = receiver_mesh.numberCollocationPoints();
  //        hr.pattern_.nc = source_mesh.numberCollocationPoints();

  if (verbose){
    std::cout << " Number of blocks =" << hr->pattern_.n_B << "\n";
    std::cout << " Number of full blocks =" << hr->pattern_.n_FRB << "\n";
    std::cout << " Number of low rank blocks =" << hr->pattern_.n_LRB << "\n";
    std::cout << "Pattern Created \n";
  }
  

    return hr;
}

} // namespace bigwham

#endif // BIGWHAM_HIERARCHICAL_REPRESENTATION_H

//
// This file is part of BigWham.
//
// Created by Francois Fayard - 2018
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
#pragma once

#include <il/Timer.h>
#include <il/linearAlgebra/dense/blas/blas_static.h>
#include <il/math.h>

#include <hmat/compression/routines.h>
#include <hmat/hmatrix/LowRank.h>

namespace bie {

template <il::int_t p, typename T>
std::unique_ptr<LowRank<T>> adaptiveCrossApproximation(const bie::MatrixGenerator<T>& M,
                                                       il::Range range0, il::Range range1,
                                                       double epsilon) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;

  auto lrb = std::make_unique<LowRank<T>>();

  auto & A = lrb->A;
  auto & B = lrb->B;
  A.Resize(n0 * p, 0);
  B.Resize(0, n1 * p);

  A.Reserve(n0*p, 20);
  B.Reserve(20, n1 * p);

  il::Array<il::int_t> i0_used{};
  il::Array<il::int_t> i1_used{};

  il::int_t rank = 0;
  il::int_t i0_search = range0.begin;
  double frobenius_low_rank = 0.0;
  double frobenius_norm_difference = -1.0;

  il::Array2D<T> row{p, n1 * p};
  il::Array2D<T> column{n0 * p, p};
  while (true) {
    bie::residual_row<p>(M, A, B, range0, range1, i0_search, rank, il::io,
                        row.Edit());
    const il::int_t i1_search =
        bie::find_largest_singular_value<p>(row, range1, i1_used);
    if (i1_search == -1) {
      break;
    }
    i0_used.Append(i0_search);
    i1_used.Append(i1_search);
    // Now, we compute the inverse for the pxp-matrix which has the largest
    // smallest singular value. The matrix we have has to be nonsingular.
    // Otherwise, we would have gotten i1_search == -1
    il::StaticArray2D<T, p, p> pivot_matrix{};
    for (il::int_t j1 = 0; j1 < p; ++j1) {
      for (il::int_t j0 = 0; j0 < p; ++j0) {
        pivot_matrix(j0, j1) = row(j0, (i1_search - range1.begin) * p + j1);
      }
    }
    il::Status status{};
    il::LU<il::StaticArray2D<T, p, p>> lu{pivot_matrix, il::io, status};
    status.AbortOnError();
    il::StaticArray2D<T, p, p> gamma = lu.inverse();

    // Check if we have a finite matrix
    // Be careful, as the error can raise only at the last minute with
    // denormalized number
    bool is_finite = true;
    for (il::int_t i0 = 0; i0 < p; ++i0) {
      for (il::int_t i1 = 0; i1 < p; ++i1) {
        if (!il::isFinite(gamma(i0, i1))) {
          is_finite = false;
        }
      }
    }
    if (!is_finite) {
      break;
    }
    // Just to check
    il::StaticArray2D<T, p, p> check_identity = il::dot(gamma, pivot_matrix);

    // Update the Matrices A and B to take into account the new ranks
    A.Resize(n0 * p, (rank + 1) * p);
    bie::residual_column<p>(M, A, B, range0, range1, i1_search, rank, il::io,
                           column.Edit());
    for (il::int_t i0 = range0.begin; i0 < range0.end; ++i0) {
      for (il::int_t j1 = 0; j1 < p; ++j1) {
        for (il::int_t j0 = 0; j0 < p; ++j0) {
          A((i0 - range0.begin) * p + j0, rank * p + j1) =
              column((i0 - range0.begin) * p + j0, j1);
        }
      }
    }
    B.Resize((rank + 1) * p, n1 * p);
    bie::residual_row<p>(M, A, B, range0, range1, i0_search, rank, il::io,
                        row.Edit());
    for (il::int_t i1 = range1.begin; i1 < range1.end; ++i1) {
      il::StaticArray2D<T, p, p> matrix{};
      for (il::int_t j1 = 0; j1 < p; ++j1) {
        for (il::int_t j0 = 0; j0 < p; ++j0) {
          matrix(j0, j1) = row(j0, (i1 - range1.begin) * p + j1);
        }
      }
      matrix = il::dot(gamma, matrix);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          IL_EXPECT_MEDIUM(il::isFinite(matrix(b0, b1)));
          B(rank * p + b0, (i1 - range1.begin) * p + b1) = matrix(b0, b1);
        }
      }
    }

    // New value for the norm of A
    il::StaticArray2D<T, p, p> frobenius_A{0.0};
    il::blas(1.0,
             A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
             il::Dot::Star,
             A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
             0.0, il::io, frobenius_A.Edit());
    // New value for the norm of B
    il::StaticArray2D<T, p, p> frobenius_B{0.0};
    il::blas(1.0,
             B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
             B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
             il::Dot::Star, 0.0, il::io, frobenius_B.Edit());
    // compute ||A_k B_k||^2
    T frobenius_norm_ab = 0.0;
    for (il::int_t b1 = 0; b1 < p; ++b1) {
      for (il::int_t b0 = 0; b0 < p; ++b0) {
        frobenius_norm_ab +=
            frobenius_A(b0, b1) * il::conjugate(frobenius_B(b0, b1));
      }
    }
    // Compute the T scalar product
    T scalar_product = 0.0;
    for (il::int_t r = 0; r < rank; ++r) {
      // Compute Ar*.Arank
      il::StaticArray2D<T, p, p> ars_arank{0.0};
      il::Array2DEdit<T> ref_ars_arank = ars_arank.Edit();
      il::blas(
          1.0, A.view(il::Range{0, n0 * p}, il::Range{r * p, (r + 1) * p}),
          il::Dot::Star,
          A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
          0.0, il::io, ref_ars_arank);
      // Compute (Ar*.Arank).Brank
      il::Array2D<T> ars_arank_brank{p, n1 * p, 0.0};
      il::Array2DEdit<T> ref_ars_arank_brank = ars_arank_brank.Edit();
      il::blas(
          1.0, ars_arank.view(),
          B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
          0.0, il::io, ref_ars_arank_brank);

      for (il::int_t j1 = 0; j1 < n1 * p; ++j1) {
        for (il::int_t b = 0; b < p; ++b) {
          scalar_product +=
              il::conjugate(B(r * p + b, j1)) * ars_arank_brank(b, j1);
        }
      }
    }
    frobenius_low_rank +=
        2 * il::real(scalar_product) + il::real(frobenius_norm_ab);

    i0_search = bie::searchI0<p>(A, range0, range1, i0_used, i1_search, rank);
    ++rank;

    //    il::Array2D<T> low_rank =
    //        il::lowRankApproximation(M, range0, range1, A, B, rank);
    //    const T forbenius_norm_low_rank = il::frobeniusNorm(low_rank);

    //     Just to check
    //    il::Array2D<T> difference_matrix =
    //        il::fullDifference(M, range0, range1, A, B, rank);
    //    frobenius_norm_difference =
    //        il::frobeniusNorm(difference_matrix);

    if (i0_search == -1 || il::abs(frobenius_norm_ab) <=  il::ipow<2>(epsilon) * il::abs(frobenius_low_rank) ||
        rank == il::min(n0, n1)) {
      break;
    }
  }

  //  il::Array2D<T> difference_matrix =
  //      il::fullDifference(M, range0, range1, A, B, rank);
  //  frobenius_norm_difference = il::frobeniusNorm(difference_matrix);
  //  const T frobenius_norm_matrix =  frobeniusNorm(il::fullMatrix(M,
  //  range0, range1));
  //  IL_EXPECT_MEDIUM(std::isfinite(frobenius_norm_difference));
  //  IL_EXPECT_MEDIUM(std::isfinite(frobenius_norm_matrix));
  //  std::cout << "Relative Error: " << frobenius_norm_difference /
  //  frobenius_norm_matrix << std::endl;

  return std::move(lrb);
}

}  // namespace il

#pragma once

#include <Eigen/Dense>
#include <lapacke.h>

#include "polar_algorithms.h"
#include "polar_enums.h"

namespace SVD_Project {

/**
 * @brief Reusable symmetric eigensolver / diagonalizer suite for polar factor H.
 *
 * Computes H = V * diag(singularValues) * V^T with singularValues sorted in
 * non-increasing order, designed to be shared across all polar decomposition
 * algorithms.
 */
template <typename _MatrixType>
struct SymmetricDiagonalizer {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  /// Dispatch diagonalization according to the selected DiagonalizationAlgorithm.
  static void diagonalize(DiagonalizationAlgorithm method,
                          const MatrixDynamic& H, VectorDynamic& singularValues,
                          MatrixDynamic& V);

  /// Method 1: Classical two-sided cyclic Jacobi on H.
  static void two_sided_jacobi(const MatrixDynamic& H,
                               VectorDynamic& singularValues, MatrixDynamic& V);

  /// Method 2: Complete pivoted Cholesky + Drmac-Veselic one-sided Jacobi on H.
  static void cholesky_onesided_evd(const MatrixDynamic& H,
                                    VectorDynamic& singularValues,
                                    MatrixDynamic& V);

  /// Method 3: Spectral Divide & Conquer via QDWH matrix sign (QDWH-EIG).
  static void qdwh_eig_evd(const MatrixDynamic& H,
                           VectorDynamic& singularValues, MatrixDynamic& V);

  /// Method 4: Tridiagonalization + Gu-Eisenstat D&C (LAPACK dsyevd).
  static void tridiag_dc_evd(const MatrixDynamic& H,
                             VectorDynamic& singularValues, MatrixDynamic& V);

 private:
  static void qdwh_eig_step(const MatrixDynamic& H_sub, VectorDynamic& lambdas,
                            MatrixDynamic& V_sub);

  static void sort_eigpairs_desc(VectorDynamic& singularValues,
                                 MatrixDynamic& V);
};

}  // namespace SVD_Project

#include "diagonalizers.cpp"

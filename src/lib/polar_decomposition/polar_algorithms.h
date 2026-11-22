#pragma once

#include <Eigen/Dense>
#include <ostream>

#include "polar_enums.h"

namespace SVD_Project {

/**
 * @brief Dynamically weighted Halley (QDWH) polar decomposition: A = Q * H.
 */
template <typename _MatrixType>
struct HalleyPolar {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  /**
   * @brief Compute polar decomposition A = Q * H via Halley's method (QDWH).
   *
   * @param A Input m-by-n matrix with m >= n.
   * @param Q Output m-by-n unitary/isometric polar factor.
   * @param H Output n-by-n Hermitian/symmetric positive semi-definite factor.
   * @param os Optional diagnostic stream for per-iteration convergence logs.
   */
  static void compute(const MatrixDynamic& A, MatrixDynamic& Q,
                      MatrixDynamic& H, std::ostream* os = nullptr);

 private:
  static RealScalar estimate_inv_r1_norm(const MatrixDynamic& R);
};

/**
 * @brief Dispatch helper selecting a polar decomposition algorithm at runtime.
 */
template <typename _MatrixType>
struct PolarDecomposition {
  using MatrixDynamic = typename HalleyPolar<_MatrixType>::MatrixDynamic;

  static void compute(PolarAlgorithm method, const MatrixDynamic& A,
                      MatrixDynamic& Q, MatrixDynamic& H,
                      std::ostream* os = nullptr);
};

}  // namespace SVD_Project

#include "polar_algorithms.cpp"

#pragma once

#include <Eigen/Dense>
#include <ostream>

#include "halley.h"
#include "newton.h"
#include "newton_acceleration.h"
#include "polar_enums.h"

namespace SVD_Project {

namespace detail {

// Reduces tall m > n matrices via thin Householder QR (A = Q0 * R) before
// applying a square n x n polar solver to R, then forms Hermitian H.
template <typename MatrixDynamic, typename SquarePolarFn>
inline void compute_polar_via_square(const MatrixDynamic& A, MatrixDynamic& Q,
                                     MatrixDynamic& H,
                                     SquarePolarFn&& square_polar) {
  using RealScalar =
      typename Eigen::NumTraits<typename MatrixDynamic::Scalar>::Real;
  const Eigen::Index m = A.rows();
  const Eigen::Index n = A.cols();

  if (n == 0 || A.norm() == RealScalar(0)) {
    Q = MatrixDynamic::Identity(m, n);
    H = MatrixDynamic::Zero(n, n);
    return;
  }

  if (m > n) {
    Eigen::HouseholderQR<MatrixDynamic> qr(A);
    MatrixDynamic Q0 = qr.householderQ() * MatrixDynamic::Identity(m, n);
    MatrixDynamic R =
        qr.matrixQR().topRows(n).template triangularView<Eigen::Upper>();
    MatrixDynamic U_R = square_polar(R);
    Q = Q0 * U_R;
  } else {
    Q = square_polar(A);
  }

  H = RealScalar(0.5) * (Q.adjoint() * A + A.adjoint() * Q);
  H = RealScalar(0.5) * (H + H.adjoint());
}

}  // namespace detail

/**
 * @brief Umbrella dispatcher selecting a polar decomposition algorithm at runtime.
 */
template <typename _MatrixType>
struct PolarDecomposition {
  using MatrixDynamic = typename HalleyPolar<_MatrixType>::MatrixDynamic;

  static void compute(PolarAlgorithm method, const MatrixDynamic& A,
                      MatrixDynamic& Q, MatrixDynamic& H,
                      std::ostream* os = nullptr) {
    switch (method) {
      case PolarAlgorithm::Newton:
        detail::compute_polar_via_square(
            A, Q, H, [](const MatrixDynamic& sqA) -> MatrixDynamic {
              Newton_Polar<MatrixDynamic> np;
              return np.compute(sqA);
            });
        break;
      case PolarAlgorithm::NewtonAccelerated:
        detail::compute_polar_via_square(
            A, Q, H, [](const MatrixDynamic& sqA) -> MatrixDynamic {
              Newton_Polar_Acceleration<MatrixDynamic> np;
              return np.compute(sqA);
            });
        break;
      case PolarAlgorithm::Halley:
      default:
        HalleyPolar<_MatrixType>::compute(A, Q, H, os);
        break;
    }
  }
};

}  // namespace SVD_Project

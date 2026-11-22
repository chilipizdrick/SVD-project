#pragma once

#include "newton_acceleration.h"

namespace SVD_Project {
template <typename MatrixType>
typename Newton_Polar_Acceleration<MatrixType>::RealScalar
Newton_Polar_Acceleration<MatrixType>::norm1(const MatrixType& M) {
  return M.cwiseAbs().colwise().sum().maxCoeff();
}

template <typename MatrixType>
typename Newton_Polar_Acceleration<MatrixType>::RealScalar
Newton_Polar_Acceleration<MatrixType>::normInf(const MatrixType& M) {
  return M.cwiseAbs().rowwise().sum().maxCoeff();
}

template <typename MatrixType>
MatrixType Newton_Polar_Acceleration<MatrixType>::compute(
    const MatrixType& A, Index max_iter, RealScalar tol) {
  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton polar decomposition requires square matrix");

  m_steps        = 0;
  m_scaled_steps = 0;

  MatrixType X = A;
  if (norm1(X) == RealScalar(0)) return X;

  const MatrixType I = MatrixType::Identity(m, n);

  bool use_scaling = true;

  for (Index k = 0; k < max_iter; ++k) {
    const RealScalar nX1   = norm1(X);
    const RealScalar nXinf = use_scaling ? normInf(X) : RealScalar(0);

    MatrixType Z;
    {
      Eigen::PartialPivLU<MatrixType> lu(X);
      Z = lu.solve(I);
    }
    if (!Z.allFinite()) {
      Eigen::FullPivLU<MatrixType> lu(X);
      if (!lu.isInvertible()) break;
      Z = lu.solve(I);
      if (!Z.allFinite()) break;
    }

    Z.adjointInPlace();
    const MatrixType& Y = Z;

    RealScalar gamma = RealScalar(1);
    if (use_scaling) {
      MatrixType  R   = X - Y;
      RealScalar  rho = std::sqrt(norm1(R) * normInf(R));

      if (rho <= m_tau) {
        use_scaling = false;
      } else {
        const RealScalar nY1   = norm1(Y);
        const RealScalar nYinf = normInf(Y);
        gamma = std::sqrt(std::sqrt((nY1 * nYinf) / (nX1 * nXinf)));
        if (!std::isfinite(gamma) || gamma <= RealScalar(0))
          gamma = RealScalar(1);
        ++m_scaled_steps;
      }
    }

    MatrixType X_new = RealScalar(0.5) *
                       (gamma * X + (RealScalar(1) / gamma) * Y);
    if (!X_new.allFinite()) break;

    X_new -= X;
    const RealScalar nDelta = norm1(X_new);
    X += X_new;

    ++m_steps;

    if (nDelta <= tol * nX1) break;
  }

  return X;
}

}  // namespace SVD_Project

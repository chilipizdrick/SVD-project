#pragma once

#include <Eigen/LU>

#include "newton.h"

namespace SVD_Project {
template <typename MatrixType>
MatrixType Newton_Polar<MatrixType>::compute(const MatrixType& A,
                                             Index max_iter,
                                             RealScalar tol) {
  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton polar decomposition requires square matrix");

  MatrixType X = A;
  if (norm1(X) == RealScalar(0)) return X;

  const MatrixType I = MatrixType::Identity(m, n);

  for (Index k = 0; k < max_iter; ++k) {
    const RealScalar nX = norm1(X);

    MatrixType Y = X.adjoint().partialPivLu().solve(I);
    if (!Y.allFinite()) return X;

    MatrixType delta = RealScalar(0.5) * (Y - X);
    X += delta;

    if (norm1(delta) <= tol * nX) break;
  }

  return X;
}

}  // namespace SVD_Project

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

#include "halley.h"

namespace SVD_Project {

template <typename _MatrixType>
typename HalleyPolar<_MatrixType>::RealScalar
HalleyPolar<_MatrixType>::estimate_inv_r1_norm(const MatrixDynamic& R) {
  Index n = R.cols();
  if (n == 0) return RealScalar(0);

  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
  RealScalar min_d = R.diagonal().cwiseAbs().minCoeff();
  RealScalar max_d = R.diagonal().cwiseAbs().maxCoeff();

  if (min_d <= eps * max_d || min_d == RealScalar(0)) {
    return RealScalar(1) / (eps * eps);
  }

  MatrixDynamic I = MatrixDynamic::Identity(n, n);
  MatrixDynamic invR = R.template triangularView<Eigen::Upper>().solve(I);
  RealScalar gamma = invR.colwise().template lpNorm<1>().maxCoeff();

  if (std::isnan(gamma) || std::isinf(gamma) || gamma <= RealScalar(0)) {
    return RealScalar(1) / (eps * eps);
  }

  return gamma;
}

template <typename _MatrixType>
void HalleyPolar<_MatrixType>::compute(const MatrixDynamic& A, MatrixDynamic& Q,
                                       MatrixDynamic& H, std::ostream* os) {
  Index m = A.rows();
  Index n = A.cols();

  RealScalar alpha = A.norm();
  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();

  if (alpha == RealScalar(0) || n == 0) {
    Q = MatrixDynamic::Identity(m, n);
    H = MatrixDynamic::Zero(n, n);
    return;
  }

  MatrixDynamic X = A / alpha;

  Eigen::HouseholderQR<MatrixDynamic> qrA(A);
  MatrixDynamic RA =
      qrA.matrixQR().topRows(n).template triangularView<Eigen::Upper>();
  RealScalar gamma = estimate_inv_r1_norm(RA);
  RealScalar beta =
      RealScalar(1) / (gamma * std::sqrt(static_cast<RealScalar>(n)));
  RealScalar l = beta / alpha;

  if (l < eps) l = eps;
  if (l > RealScalar(1)) l = RealScalar(1);

  RealScalar tau = eps;
  RealScalar conv_tol = std::cbrt(RealScalar(4) * tau);
  const int max_iters = 30;

  for (int iter = 0; iter < max_iters; ++iter) {
    RealScalar a, b, c;
    if (std::abs(RealScalar(1) - l) <= tau) {
      a = RealScalar(3);
      b = RealScalar(1);
      c = RealScalar(3);
    } else {
      RealScalar l2 = l * l;
      RealScalar l4 = l2 * l2;
      RealScalar delta = std::cbrt(RealScalar(4) * (RealScalar(1) - l2) / l4);
      RealScalar sqrt_1_plus_delta = std::sqrt(RealScalar(1) + delta);
      RealScalar radicand =
          RealScalar(8) - RealScalar(4) * delta +
          (RealScalar(8) * (RealScalar(2) - l2)) / (l2 * sqrt_1_plus_delta);
      a = sqrt_1_plus_delta +
          RealScalar(0.5) * std::sqrt(std::max(RealScalar(0), radicand));
      b = (a - RealScalar(1)) * (a - RealScalar(1)) / RealScalar(4);
      c = a + b - RealScalar(1);
    }

    MatrixDynamic W(m + n, n);
    W.topRows(m) = std::sqrt(c) * X;
    W.bottomRows(n) = MatrixDynamic::Identity(n, n);

    Eigen::HouseholderQR<MatrixDynamic> qrW(W);
    MatrixDynamic Y = qrW.householderQ() * MatrixDynamic::Identity(m + n, n);
    MatrixDynamic Y1 = Y.topRows(m);
    MatrixDynamic Y2 = Y.bottomRows(n);

    MatrixDynamic X_next = (b / c) * X + (RealScalar(1) / std::sqrt(c)) *
                                             (a - (b / c)) *
                                             (Y1 * Y2.adjoint());
    RealScalar l_next = l * (a + b * l * l) / (RealScalar(1) + c * l * l);
    if (l_next > RealScalar(1)) l_next = RealScalar(1);

    RealScalar diff = (X_next - X).norm();
    if (os) {
      *os << "QDWH iter " << iter << ": diff = " << diff
          << ", l_next = " << l_next << "\n";
    }

    X = X_next;
    l = l_next;

    if (diff <= conv_tol && (RealScalar(1) - l) <= tau) {
      break;
    }
  }

  Q = X;
  H = RealScalar(0.5) * (Q.adjoint() * A + A.adjoint() * Q);
  H = RealScalar(0.5) * (H + H.adjoint());
}

}  // namespace SVD_Project

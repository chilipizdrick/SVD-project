#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "halley_svd.h"

namespace SVD_Project {

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
Halley_SVD<_MatrixType, _DiagMethod>::Halley_SVD() {}

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
Halley_SVD<_MatrixType, _DiagMethod>::Halley_SVD(
    const MatrixType& A, unsigned int computationOptions) {
  compute(A, computationOptions);
}

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
Halley_SVD<_MatrixType, _DiagMethod>::Halley_SVD(
    const MatrixType& A, std::ostream* os, unsigned int computationOptions)
    : m_divOstream(os) {
  compute(A, computationOptions);
}

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
typename Halley_SVD<_MatrixType, _DiagMethod>::RealScalar
Halley_SVD<_MatrixType, _DiagMethod>::estimate_inv_r1_norm(
    const MatrixDynamic& R) {
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

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_qdwh_polar(
    const MatrixDynamic& A, MatrixDynamic& Q, MatrixDynamic& H) {
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
    if (m_divOstream) {
      *m_divOstream << "QDWH iter " << iter << ": diff = " << diff
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

// ============================================================================
// Method 1: Two-Sided Cyclic Jacobi on H
// ============================================================================
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_two_sided_jacobi(
    const MatrixDynamic& H, VectorDynamic& singularValues, MatrixDynamic& V) {
  Index n = H.rows();
  V = MatrixDynamic::Identity(n, n);
  MatrixDynamic A_work = H;

  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
  RealScalar h_norm = H.norm();
  if (h_norm == RealScalar(0)) {
    singularValues = VectorDynamic::Zero(n);
    return;
  }

  const int max_sweeps = 100;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    RealScalar off_norm_sq = RealScalar(0);
    for (Index p = 0; p < n; ++p) {
      for (Index q = p + 1; q < n; ++q) {
        RealScalar val = std::abs(A_work(p, q));
        off_norm_sq += val * val;
      }
    }
    RealScalar off_norm = std::sqrt(RealScalar(2) * off_norm_sq);
    if (off_norm <= eps * h_norm) {
      break;
    }

    RealScalar threshold =
        (sweep < 3)
            ? (RealScalar(0.2) * off_norm / static_cast<RealScalar>(n * n))
            : RealScalar(0);

    for (Index p = 0; p < n; ++p) {
      for (Index q = p + 1; q < n; ++q) {
        RealScalar apq = std::real(A_work(p, q));
        if (std::abs(apq) <= threshold && sweep < 3) continue;

        RealScalar app = std::real(A_work(p, p));
        RealScalar aqq = std::real(A_work(q, q));

        if (std::abs(apq) <= eps * (std::abs(app) + std::abs(aqq))) {
          A_work(p, q) = RealScalar(0);
          A_work(q, p) = RealScalar(0);
          continue;
        }

        RealScalar tau = (aqq - app) / (RealScalar(2) * apq);
        RealScalar t;
        if (tau >= RealScalar(0)) {
          t = RealScalar(1) / (tau + std::sqrt(RealScalar(1) + tau * tau));
        } else {
          t = -RealScalar(1) / (-tau + std::sqrt(RealScalar(1) + tau * tau));
        }

        RealScalar c = RealScalar(1) / std::sqrt(RealScalar(1) + t * t);
        RealScalar s = t * c;

        A_work(p, p) = app - t * apq;
        A_work(q, q) = aqq + t * apq;
        A_work(p, q) = RealScalar(0);
        A_work(q, p) = RealScalar(0);

        for (Index i = 0; i < n; ++i) {
          if (i != p && i != q) {
            Scalar a_ip = A_work(i, p);
            Scalar a_iq = A_work(i, q);
            A_work(i, p) = c * a_ip - s * a_iq;
            A_work(p, i) = A_work(i, p);
            A_work(i, q) = s * a_ip + c * a_iq;
            A_work(q, i) = A_work(i, q);
          }
        }

        for (Index i = 0; i < n; ++i) {
          Scalar v_ip = V(i, p);
          Scalar v_iq = V(i, q);
          V(i, p) = c * v_ip - s * v_iq;
          V(i, q) = s * v_ip + c * v_iq;
        }
      }
    }
  }

  singularValues.resize(n);
  for (Index i = 0; i < n; ++i) {
    RealScalar val = std::real(A_work(i, i));
    if (val < RealScalar(0)) {
      val = -val;
      V.col(i) = -V.col(i);
    }
    singularValues(i) = val;
  }

  std::vector<Index> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](Index i, Index j) {
    return singularValues(i) > singularValues(j);
  });

  VectorDynamic sorted_S(n);
  MatrixDynamic sorted_V(n, n);
  for (Index i = 0; i < n; ++i) {
    sorted_S(i) = singularValues(order[i]);
    sorted_V.col(i) = V.col(order[i]);
  }
  singularValues = sorted_S;
  V = sorted_V;
}

// ============================================================================
// Method 2: Complete Pivoting Cholesky + Drmac-Veselic One-Sided Jacobi on H
// ============================================================================
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_cholesky_onesided_evd(
    const MatrixDynamic& H, VectorDynamic& singularValues, MatrixDynamic& V) {
  Index n = H.rows();
  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();

  if (n == 0) {
    singularValues.resize(0);
    V.resize(0, 0);
    return;
  }

  // Step 1: Complete pivoted Cholesky factorization of H: P^T H P = R^T R
  MatrixDynamic A_work = H;
  MatrixDynamic P = MatrixDynamic::Identity(n, n);
  MatrixDynamic R = MatrixDynamic::Zero(n, n);

  for (Index k = 0; k < n; ++k) {
    Index max_idx = k;
    RealScalar max_val = std::real(A_work(k, k));
    for (Index j = k + 1; j < n; ++j) {
      RealScalar val = std::real(A_work(j, j));
      if (val > max_val) {
        max_val = val;
        max_idx = j;
      }
    }
    if (max_idx != k) {
      A_work.row(k).swap(A_work.row(max_idx));
      A_work.col(k).swap(A_work.col(max_idx));
      P.col(k).swap(P.col(max_idx));
      R.block(0, k, k, 1).swap(R.block(0, max_idx, k, 1));
    }

    RealScalar d = std::real(A_work(k, k));
    if (d <= eps * eps) {
      break;
    }

    RealScalar rkk = std::sqrt(d);
    R(k, k) = rkk;
    Index rem = n - 1 - k;
    if (rem > 0) {
      R.row(k).tail(rem) = A_work.row(k).tail(rem) / rkk;
      A_work.bottomRightCorner(rem, rem) -=
          R.row(k).tail(rem).transpose() * R.row(k).tail(rem);
    }
  }

  // Step 2: One-sided Jacobi on R: R * V_R = W, where columns of W are orthogonal
  MatrixDynamic C = R;
  MatrixDynamic V_R = MatrixDynamic::Identity(n, n);
  const int max_sweeps = 60;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    bool any_rot = false;
    for (Index p = 0; p < n; ++p) {
      for (Index q = p + 1; q < n; ++q) {
        RealScalar a = C.col(p).squaredNorm();
        RealScalar b = C.col(q).squaredNorm();
        RealScalar d = C.col(p).dot(C.col(q));

        if (std::abs(d) <= eps * std::sqrt(a * b) || a == RealScalar(0) || b == RealScalar(0)) {
          continue;
        }

        any_rot = true;
        RealScalar tau = (a - b) / (RealScalar(2) * d);
        RealScalar t = (tau >= RealScalar(0))
                           ? (RealScalar(1) / (tau + std::sqrt(RealScalar(1) + tau * tau)))
                           : (-RealScalar(1) / (-tau + std::sqrt(RealScalar(1) + tau * tau)));
        RealScalar c = RealScalar(1) / std::sqrt(RealScalar(1) + t * t);
        RealScalar s = t * c;

        VectorDynamic cp = C.col(p);
        VectorDynamic cq = C.col(q);
        C.col(p) = c * cp + s * cq;
        C.col(q) = -s * cp + c * cq;

        VectorDynamic vp = V_R.col(p);
        VectorDynamic vq = V_R.col(q);
        V_R.col(p) = c * vp + s * vq;
        V_R.col(q) = -s * vp + c * vq;
      }
    }
    if (!any_rot) break;
  }

  // Step 3: Eigenvalues of H are the squares of singular values of R!
  // lambda_i(H) = sigma_i(R)^2
  singularValues.resize(n);
  for (Index i = 0; i < n; ++i) {
    RealScalar s_i = C.col(i).norm();
    singularValues(i) = s_i * s_i;
  }

  // V = P * V_R
  MatrixDynamic V_unperm = P * V_R;

  std::vector<Index> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](Index i, Index j) {
    return singularValues(i) > singularValues(j);
  });

  VectorDynamic sorted_S(n);
  MatrixDynamic sorted_V(n, n);
  for (Index i = 0; i < n; ++i) {
    sorted_S(i) = singularValues(order[i]);
    sorted_V.col(i) = V_unperm.col(order[i]);
  }
  singularValues = sorted_S;
  V = sorted_V;
}

// ============================================================================
// Method 3: Spectral Divide & Conquer via QDWH Matrix Sign (QDWH-EIG)
// ============================================================================
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_qdwh_eig_step(
    const MatrixDynamic& H_sub, VectorDynamic& lambdas, MatrixDynamic& V_sub) {
  Index n = H_sub.rows();
  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();

  if (n == 1) {
    lambdas.resize(1);
    lambdas(0) = std::real(H_sub(0, 0));
    V_sub = MatrixDynamic::Identity(1, 1);
    return;
  }

  if (n == 2) {
    RealScalar h00 = std::real(H_sub(0, 0));
    RealScalar h11 = std::real(H_sub(1, 1));
    RealScalar h01 = std::real(H_sub(0, 1));

    if (std::abs(h01) <= eps * (std::abs(h00) + std::abs(h11))) {
      lambdas.resize(2);
      lambdas(0) = h00;
      lambdas(1) = h11;
      V_sub = MatrixDynamic::Identity(2, 2);
      return;
    }

    RealScalar tau = (h11 - h00) / (RealScalar(2) * h01);
    RealScalar t = (tau >= RealScalar(0))
                       ? (RealScalar(1) / (tau + std::sqrt(RealScalar(1) + tau * tau)))
                       : (-RealScalar(1) / (-tau + std::sqrt(RealScalar(1) + tau * tau)));
    RealScalar c = RealScalar(1) / std::sqrt(RealScalar(1) + t * t);
    RealScalar s = t * c;

    lambdas.resize(2);
    lambdas(0) = h00 - t * h01;
    lambdas(1) = h11 + t * h01;

    V_sub.resize(2, 2);
    V_sub(0, 0) = c;
    V_sub(1, 0) = -s;
    V_sub(0, 1) = s;
    V_sub(1, 1) = c;
    return;
  }

  RealScalar diag_norm_sq = H_sub.diagonal().squaredNorm();
  RealScalar total_norm_sq = H_sub.squaredNorm();
  RealScalar off_norm = std::sqrt(std::max(RealScalar(0), total_norm_sq - diag_norm_sq));
  if (off_norm <= eps * std::sqrt(total_norm_sq)) {
    lambdas = H_sub.diagonal();
    V_sub = MatrixDynamic::Identity(n, n);
    return;
  }

  std::vector<RealScalar> diag_vals(n);
  for (Index i = 0; i < n; ++i) diag_vals[i] = std::real(H_sub(i, i));
  std::sort(diag_vals.begin(), diag_vals.end());
  RealScalar sigma = RealScalar(0.5) * (diag_vals[n / 2 - 1] + diag_vals[n / 2]);

  if (diag_vals.back() - diag_vals.front() <= eps * std::abs(diag_vals.back())) {
    lambdas = H_sub.diagonal();
    V_sub = MatrixDynamic::Identity(n, n);
    return;
  }

  MatrixDynamic M = H_sub - sigma * MatrixDynamic::Identity(n, n);
  MatrixDynamic U_M, H_dummy;
  compute_qdwh_polar(M, U_M, H_dummy);

  MatrixDynamic P = RealScalar(0.5) * (MatrixDynamic::Identity(n, n) + U_M);
  Eigen::ColPivHouseholderQR<MatrixDynamic> qrP(P);
  Index k = qrP.rank();

  if (k <= 0 || k >= n) {
    compute_two_sided_jacobi(H_sub, lambdas, V_sub);
    return;
  }

  MatrixDynamic Q_P = qrP.householderQ() * MatrixDynamic::Identity(n, n);
  MatrixDynamic V1 = Q_P.leftCols(k);
  MatrixDynamic V2 = Q_P.rightCols(n - k);

  MatrixDynamic H1 = V1.adjoint() * H_sub * V1;
  MatrixDynamic H2 = V2.adjoint() * H_sub * V2;
  H1 = RealScalar(0.5) * (H1 + H1.adjoint());
  H2 = RealScalar(0.5) * (H2 + H2.adjoint());

  VectorDynamic lam1, lam2;
  MatrixDynamic W1, W2;
  compute_qdwh_eig_step(H1, lam1, W1);
  compute_qdwh_eig_step(H2, lam2, W2);

  lambdas.resize(n);
  lambdas.head(k) = lam1;
  lambdas.tail(n - k) = lam2;

  V_sub.resize(n, n);
  V_sub.leftCols(k) = V1 * W1;
  V_sub.rightCols(n - k) = V2 * W2;
}

template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_qdwh_eig_evd(
    const MatrixDynamic& H, VectorDynamic& singularValues, MatrixDynamic& V) {
  Index n = H.rows();
  if (n == 0) {
    singularValues.resize(0);
    V.resize(0, 0);
    return;
  }

  VectorDynamic lambdas;
  compute_qdwh_eig_step(H, lambdas, V);

  singularValues.resize(n);
  for (Index i = 0; i < n; ++i) {
    RealScalar val = lambdas(i);
    if (val < RealScalar(0)) {
      val = -val;
      V.col(i) = -V.col(i);
    }
    singularValues(i) = val;
  }

  std::vector<Index> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](Index i, Index j) {
    return singularValues(i) > singularValues(j);
  });

  VectorDynamic sorted_S(n);
  MatrixDynamic sorted_V(n, n);
  for (Index i = 0; i < n; ++i) {
    sorted_S(i) = singularValues(order[i]);
    sorted_V.col(i) = V.col(order[i]);
  }
  singularValues = sorted_S;
  V = sorted_V;
}

// ============================================================================
// Method 4: Tridiagonalization + Divide & Conquer (LAPACK dsyevd)
// ============================================================================
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
void Halley_SVD<_MatrixType, _DiagMethod>::compute_tridiag_dc_evd(
    const MatrixDynamic& H, VectorDynamic& singularValues, MatrixDynamic& V) {
  int n = static_cast<int>(H.rows());
  if (n == 0) {
    singularValues.resize(0);
    V.resize(0, 0);
    return;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> H_work =
      H.template cast<double>();
  Eigen::Matrix<double, Eigen::Dynamic, 1> S(n);

  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, H_work.data(), n, S.data());
  if (info != 0) {
    compute_two_sided_jacobi(H, singularValues, V);
    return;
  }

  singularValues.resize(n);
  V.resize(n, n);

  for (int i = 0; i < n; ++i) {
    double val = S(n - 1 - i);
    singularValues(i) = static_cast<Scalar>(std::max(0.0, val));
    V.col(i) = H_work.col(n - 1 - i).template cast<Scalar>();
  }
}

// ============================================================================
// Main Compute Dispatcher
// ============================================================================
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
Halley_SVD<_MatrixType, _DiagMethod>&
Halley_SVD<_MatrixType, _DiagMethod>::compute(
    const MatrixType& A, unsigned int computationOptions) {
  Index m = A.rows();
  Index n = A.cols();

  if (m < n) {
    Halley_SVD<MatrixType, _DiagMethod> svd_adj;
    svd_adj.setDiagonalizer(m_diagonalizer);
    unsigned int adjOptions = 0;
    if (computationOptions & Eigen::ComputeFullU) adjOptions |= Eigen::ComputeFullV;
    if (computationOptions & Eigen::ComputeThinU) adjOptions |= Eigen::ComputeThinV;
    if (computationOptions & Eigen::ComputeFullV) adjOptions |= Eigen::ComputeFullU;
    if (computationOptions & Eigen::ComputeThinV) adjOptions |= Eigen::ComputeThinU;

    svd_adj.setThreshold(m_threshold);
    svd_adj.compute(A.adjoint(), adjOptions);
    this->m_singularValues = svd_adj.singularValues();

    if (computationOptions & Eigen::ComputeFullU) {
      this->m_matrixU = svd_adj.matrixV();
      this->m_computeFullU = true;
    } else if (computationOptions & Eigen::ComputeThinU) {
      this->m_matrixU = svd_adj.matrixV();
      this->m_computeThinU = true;
    }

    if (computationOptions & Eigen::ComputeFullV) {
      this->m_matrixV = svd_adj.matrixU();
      this->m_computeFullV = true;
    } else if (computationOptions & Eigen::ComputeThinV) {
      this->m_matrixV = svd_adj.matrixU();
      this->m_computeThinV = true;
    }
    this->m_isInitialized = true;
    return *this;
  }

  compute_qdwh_polar(A, m_matrixQ, m_matrixH);

  VectorDynamic S;
  MatrixDynamic V;

  switch (m_diagonalizer) {
    case HalleyDiagonalizer::CholeskyOneSided:
      compute_cholesky_onesided_evd(m_matrixH, S, V);
      break;
    case HalleyDiagonalizer::QDWH_EIG:
      compute_qdwh_eig_evd(m_matrixH, S, V);
      break;
    case HalleyDiagonalizer::TridiagDC:
      compute_tridiag_dc_evd(m_matrixH, S, V);
      break;
    case HalleyDiagonalizer::TwoSidedJacobi:
    default:
      compute_two_sided_jacobi(m_matrixH, S, V);
      break;
  }

  if (m_threshold > RealScalar(0)) {
    RealScalar cutoff = m_threshold * A.norm();
    for (Index i = 0; i < S.size(); ++i) {
      if (S(i) <= cutoff) {
        S(i) = RealScalar(0);
      }
    }
  }

  this->m_singularValues = S;

  MatrixDynamic thinU = m_matrixQ * V;

  bool computeU =
      (computationOptions & (Eigen::ComputeThinU | Eigen::ComputeFullU)) != 0;
  bool computeV =
      (computationOptions & (Eigen::ComputeThinV | Eigen::ComputeFullV)) != 0;
  bool fullU = (computationOptions & Eigen::ComputeFullU) != 0;

  if (computeU) {
    Eigen::HouseholderQR<MatrixDynamic> qrU(thinU);
    if (fullU) {
      MatrixDynamic full_U = qrU.householderQ() * MatrixDynamic::Identity(m, m);
      for (Index j = 0; j < n; ++j) {
        if (full_U.col(j).dot(thinU.col(j)) < RealScalar(0)) {
          full_U.col(j) = -full_U.col(j);
        }
      }
      this->m_matrixU = full_U;
      this->m_computeFullU = true;
    } else {
      MatrixDynamic ortho_thinU =
          qrU.householderQ() * MatrixDynamic::Identity(m, n);
      for (Index j = 0; j < n; ++j) {
        if (ortho_thinU.col(j).dot(thinU.col(j)) < RealScalar(0)) {
          ortho_thinU.col(j) = -ortho_thinU.col(j);
        }
      }
      this->m_matrixU = ortho_thinU;
      this->m_computeThinU = true;
    }
  }

  if (computeV) {
    this->m_matrixV = V;
    if (computationOptions & Eigen::ComputeFullV) {
      this->m_computeFullV = true;
    } else {
      this->m_computeThinV = true;
    }
  }

  this->m_isInitialized = true;
  return *this;
}

}  // namespace SVD_Project

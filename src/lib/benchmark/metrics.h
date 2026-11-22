#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "types.h"

namespace SVD_Project::Benchmark {

inline uint64_t fnv1a_update(uint64_t hash, const void* data, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline std::string hash_matrix_instance(const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& S_true,
                                        bool has_ground_truth) {
  uint64_t h = 14695981039346656037ULL;
  int64_t rows = A.rows();
  int64_t cols = A.cols();
  h = fnv1a_update(h, &rows, sizeof(rows));
  h = fnv1a_update(h, &cols, sizeof(cols));
  if (A.size() > 0) {
    h = fnv1a_update(h, A.data(), static_cast<size_t>(A.size()) * sizeof(double));
  }
  uint8_t gt_flag = has_ground_truth ? 1 : 0;
  h = fnv1a_update(h, &gt_flag, sizeof(gt_flag));
  if (has_ground_truth && S_true.size() > 0) {
    h = fnv1a_update(h, S_true.data(),
                     static_cast<size_t>(S_true.size()) * sizeof(double));
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

inline void evaluate_svd_metrics(const MatrixInstance& inst,
                                 const SVDOutput& out, double time_ms,
                                 BenchmarkRecord& rec) {
  const Eigen::MatrixXd& A = inst.A;
  const Eigen::MatrixXd& U = out.U;
  const Eigen::VectorXd& S_calc = out.S;
  const Eigen::MatrixXd& V = out.V;

  int m_rows = static_cast<int>(A.rows());
  int n_cols = static_cast<int>(A.cols());
  int p = std::min(m_rows, n_cols);

  rec.metrics.time_ms = time_ms;
  rec.has_ground_truth = inst.has_ground_truth && (inst.S_true.size() == p);

  bool finite = true;
  if (S_calc.size() < p || U.rows() != m_rows || U.cols() < p ||
      V.rows() != n_cols || V.cols() < p) {
    rec.status = RunStatus::Failed;
    rec.error_msg = "Dimension mismatch in SVD output";
    return;
  }

  for (int i = 0; i < S_calc.size() && finite; ++i) {
    if (!std::isfinite(S_calc(i))) finite = false;
  }
  for (int j = 0; j < U.cols() && finite; ++j) {
    for (int i = 0; i < U.rows(); ++i) {
      if (!std::isfinite(U(i, j))) {
        finite = false;
        break;
      }
    }
  }
  for (int j = 0; j < V.cols() && finite; ++j) {
    for (int i = 0; i < V.rows(); ++i) {
      if (!std::isfinite(V(i, j))) {
        finite = false;
        break;
      }
    }
  }

  if (!finite) {
    rec.status = RunStatus::NonFinite;
    rec.error_msg = "Non-finite value detected in U, S, or V";
    return;
  }

  rec.status = RunStatus::Ok;
  rec.metrics.sigma_max = (p > 0) ? S_calc.head(p).maxCoeff() : 0.0;
  rec.metrics.sigma_min = (p > 0) ? S_calc.head(p).minCoeff() : 0.0;
  rec.metrics.kappa =
      (rec.metrics.sigma_min > 0.0)
          ? (rec.metrics.sigma_max / rec.metrics.sigma_min)
          : std::numeric_limits<double>::quiet_NaN();

  if (m_rows == n_cols) {
    double log_sum = 0.0;
    bool all_pos = true;
    for (int i = 0; i < p; ++i) {
      if (S_calc(i) > 0.0) {
        log_sum += std::log(S_calc(i));
      } else {
        all_pos = false;
        break;
      }
    }
    rec.metrics.log_det =
        all_pos ? log_sum : std::numeric_limits<double>::quiet_NaN();
  }

  // 1. Orthonormality of U (thin U columns)
  Eigen::MatrixXd UtU = U.leftCols(p).transpose() * U.leftCols(p);
  double max_wU = 0.0;
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      double diff = std::abs(UtU(i, j) - (i == j ? 1.0 : 0.0));
      if (diff > max_wU) max_wU = diff;
    }
  }
  rec.metrics.omega_U = max_wU;

  // 2. Orthonormality of V (thin V columns)
  Eigen::MatrixXd VtV = V.leftCols(p).transpose() * V.leftCols(p);
  double max_wV = 0.0;
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      double diff = std::abs(VtV(i, j) - (i == j ? 1.0 : 0.0));
      if (diff > max_wV) max_wV = diff;
    }
  }
  rec.metrics.omega_V = max_wV;

  // 3. Residuals: f_j = A * v_j - s_j * u_j, g_j = A^T * u_j - s_j * v_j
  double M = A.cwiseAbs().maxCoeff();
  if (M == 0.0) M = 1.0;
  double sum_sq = 0.0;
  double max_rel_res = 0.0;
  double sig_cutoff = std::max(1e-15 * rec.metrics.sigma_max, 1e-100);

  for (int j = 0; j < p; ++j) {
    Eigen::VectorXd fj = A * V.col(j) - S_calc(j) * U.col(j);
    Eigen::VectorXd gj = A.transpose() * U.col(j) - S_calc(j) * V.col(j);
    double norm_sq = fj.squaredNorm() + gj.squaredNorm();
    sum_sq += norm_sq;
    if (S_calc(j) > sig_cutoff) {
      double rel = std::sqrt(norm_sq) / S_calc(j);
      if (rel > max_rel_res) max_rel_res = rel;
    }
  }
  rec.metrics.r = std::sqrt(sum_sq) / M;
  rec.metrics.eta = max_rel_res;

  // 4. Backward reconstruction error rho = ||A - U_p * S * V_p^T||_F / ||A||_F
  Eigen::MatrixXd A_rec =
      U.leftCols(p) * S_calc.head(p).asDiagonal() * V.leftCols(p).transpose();
  double norm_A = A.norm();
  rec.metrics.rho = (norm_A > 0.0) ? (A - A_rec).norm() / norm_A : 0.0;

  // 5. Forward singular value errors against ground truth (if available)
  if (rec.has_ground_truth) {
    Eigen::VectorXd s_sorted = S_calc.head(p);
    std::sort(s_sorted.data(), s_sorted.data() + p, std::greater<double>());
    double max_rel = 0.0;
    double max_abs = 0.0;
    double s0 = inst.S_true(0);
    for (int j = 0; j < p; ++j) {
      double s_c = s_sorted(j);
      double s_t = inst.S_true(j);
      double abs_diff = std::abs(s_c - s_t);
      if (abs_diff > max_abs) max_abs = abs_diff;
      if (s_t >= 1e-15 * s0 && s_t > 0.0) {
        double rel_diff = abs_diff / s_t;
        if (rel_diff > max_rel) max_rel = rel_diff;
      }
    }
    rec.metrics.delta = max_rel;
    rec.metrics.Delta = max_abs;
  }
}

}  // namespace SVD_Project::Benchmark

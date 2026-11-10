#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>

#include "lib/SVD_project.h"

namespace {

struct BenchmarkMetrics {
  std::string algo_name;
  double time_ms{0.0};
  double omega_U{0.0};
  double omega_V{0.0};
  double r{0.0};
  double eta{0.0};
  double rho{0.0};
  double cond_num{0.0};
  double max_sigma_rel_err{0.0};
  double max_sigma_abs_err{0.0};
  double log_det_val{0.0};
  double sigma_min{0.0};
  double sigma_max{0.0};
  bool has_ground_truth{false};
  bool success{true};
  bool is_finite{true};
};

struct MatrixInstance {
  Eigen::MatrixXd A;
  Eigen::VectorXd S_true;
};

void clamp_min_entries(Eigen::MatrixXd& A) {
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      if (std::abs(A(i, j)) < 1e-16) {
        A(i, j) = (A(i, j) >= 0.0) ? 1e-16 : -1e-16;
      }
    }
  }
}

// Generates a random Haar-distributed orthogonal matrix Q in O(n^3) via Householder QR
Eigen::MatrixXd generate_orthogonal_matrix(int n, std::mt19937_64& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      M(i, j) = dist(rng);
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
  // Ensure positive diagonal elements in R to make Q uniquely Haar-distributed
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
  for (int i = 0; i < n; ++i) {
    if (R(i, i) < 0.0) {
      Q.col(i) = -Q.col(i);
    }
  }
  return Q;
}

// Compute standard SVD error metrics
BenchmarkMetrics evaluate_svd(const std::string& name, const Eigen::MatrixXd& A,
                             const Eigen::VectorXd& S_true,
                             const Eigen::MatrixXd& U, const Eigen::VectorXd& S_calc,
                             const Eigen::MatrixXd& V, double time_ms) {
  int m_rows = static_cast<int>(A.rows());
  int n_cols = static_cast<int>(A.cols());
  int p = std::min(m_rows, n_cols);

  BenchmarkMetrics res;
  res.algo_name = name;
  res.time_ms = time_ms;

  for (int i = 0; i < S_calc.size(); ++i) {
    if (!std::isfinite(S_calc(i))) res.is_finite = false;
  }
  for (int i = 0; i < U.rows() && res.is_finite; ++i) {
    for (int j = 0; j < U.cols(); ++j) {
      if (!std::isfinite(U(i, j))) res.is_finite = false;
    }
  }
  for (int i = 0; i < V.rows() && res.is_finite; ++i) {
    for (int j = 0; j < V.cols(); ++j) {
      if (!std::isfinite(V(i, j))) res.is_finite = false;
    }
  }

  if (!res.is_finite) {
    res.success = false;
    res.omega_U = res.omega_V = res.r = res.eta = res.rho = res.max_sigma_rel_err = res.max_sigma_abs_err =
        std::numeric_limits<double>::infinity();
    return res;
  }

  res.sigma_max = (S_calc.size() > 0) ? S_calc.maxCoeff() : 0.0;
  res.sigma_min = (S_calc.size() > 0) ? S_calc.minCoeff() : 0.0;
  res.cond_num = (res.sigma_min > 0.0) ? (res.sigma_max / res.sigma_min) : std::numeric_limits<double>::infinity();

  double log_sum = 0.0;
  for (int i = 0; i < S_calc.size(); ++i) {
    if (S_calc(i) > 0.0) {
      log_sum += std::log(S_calc(i));
    } else {
      log_sum = -std::numeric_limits<double>::infinity();
      break;
    }
  }
  res.log_det_val = log_sum;

  // 1. Orthonormality of U (thin U columns)
  int u_cols = static_cast<int>(U.cols());
  int check_u_cols = std::min(u_cols, p);
  Eigen::MatrixXd UtU = U.leftCols(check_u_cols).transpose() * U.leftCols(check_u_cols);
  double max_wU = 0.0;
  for (int i = 0; i < check_u_cols; ++i) {
    for (int j = 0; j < check_u_cols; ++j) {
      double diff = std::abs(UtU(i, j) - (i == j ? 1.0 : 0.0));
      if (diff > max_wU) max_wU = diff;
    }
  }
  res.omega_U = max_wU;

  // 2. Orthonormality of V (thin V columns)
  int v_cols = static_cast<int>(V.cols());
  int check_v_cols = std::min(v_cols, p);
  Eigen::MatrixXd VtV = V.leftCols(check_v_cols).transpose() * V.leftCols(check_v_cols);
  double max_wV = 0.0;
  for (int i = 0; i < check_v_cols; ++i) {
    for (int j = 0; j < check_v_cols; ++j) {
      double diff = std::abs(VtV(i, j) - (i == j ? 1.0 : 0.0));
      if (diff > max_wV) max_wV = diff;
    }
  }
  res.omega_V = max_wV;

  // 3. Residuals: f_j = A * v_j - s_j * u_j, g_j = A^T * u_j - s_j * v_j
  double M = A.cwiseAbs().maxCoeff();
  if (M == 0.0) M = 1.0;
  double sum_sq = 0.0;
  double max_rel_res = 0.0;
  double sig_cutoff = std::max(1e-15 * res.sigma_max, 1e-100);

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
  res.r = std::sqrt(sum_sq) / M;
  res.eta = max_rel_res;

  // 4. Reconstruction error rho = ||A - U_p * S * V_p^T||_F / ||A||_F
  Eigen::MatrixXd A_rec = U.leftCols(p) * S_calc.head(p).asDiagonal() * V.leftCols(p).transpose();
  double norm_A = A.norm();
  res.rho = (norm_A > 0.0) ? (A - A_rec).norm() / norm_A : 0.0;

  // 5. Comparison against ground truth if available
  if (S_true.size() == p) {
    res.has_ground_truth = true;
    double max_rel = 0.0;
    double max_abs = 0.0;
    for (int j = 0; j < p; ++j) {
      double s_c = S_calc(j);
      double s_t = S_true(j);
      double abs_diff = std::abs(s_c - s_t);
      if (abs_diff > max_abs) max_abs = abs_diff;
      if (s_t >= 1e-15 * S_true(0) && s_t > 0.0) {
        double rel_diff = abs_diff / s_t;
        if (rel_diff > max_rel) max_rel = rel_diff;
      }
    }
    res.max_sigma_rel_err = max_rel;
    res.max_sigma_abs_err = max_abs;
  }

  return res;
}

// Runners
BenchmarkMetrics run_halley_tridiag_dc(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  SVD_Project::Halley_SVD<Eigen::MatrixXd, SVD_Project::HalleyDiagonalizer::TridiagDC> svd;
  svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  return evaluate_svd("Halley (TridiagDC)", A, S_true, svd.matrixU(), svd.singularValues(),
                      svd.matrixV(), ms);
}

BenchmarkMetrics run_halley_two_sided(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  SVD_Project::Halley_SVD<Eigen::MatrixXd, SVD_Project::HalleyDiagonalizer::TwoSidedJacobi> svd;
  svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  return evaluate_svd("Halley (TwoSidedJacobi)", A, S_true, svd.matrixU(), svd.singularValues(),
                      svd.matrixV(), ms);
}

BenchmarkMetrics run_dgesdd(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  DGESDD_SVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  return evaluate_svd("DGESDD_SVD (D&C)", A, S_true, svd.matrixU(), svd.singularValues(),
                      svd.matrixV(), ms);
}

BenchmarkMetrics run_dgesvd(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  DGESVD_SVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  return evaluate_svd("DGESVD_SVD (QR)", A, S_true, svd.matrixU(), svd.singularValues(),
                      svd.matrixV(), ms);
}

BenchmarkMetrics run_dbdsvdx(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  try {
    DBDSVDX_SVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return evaluate_svd("DBDSVDX_SVD (Bisect)", A, S_true, svd.matrixU(), svd.singularValues(),
                        svd.matrixV(), ms);
  } catch (...) {
    BenchmarkMetrics res;
    res.algo_name = "DBDSVDX_SVD (Bisect)";
    res.success = false;
    res.is_finite = false;
    res.rho = std::numeric_limits<double>::infinity();
    return res;
  }
}

BenchmarkMetrics run_jacobi(const Eigen::MatrixXd& A, const Eigen::VectorXd& S_true) {
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  return evaluate_svd("Eigen JacobiSVD", A, S_true, svd.matrixU(), svd.singularValues(),
                      svd.matrixV(), ms);
}

void print_table(const std::string& title, const std::vector<BenchmarkMetrics>& results,
                 bool show_log_det = false) {
  std::cout << "\n========================================================================================================================\n";
  std::cout << "  " << title << "\n";
  std::cout << "========================================================================================================================\n";
  std::cout << std::left << std::setw(26) << "Algorithm"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(11) << "omega_U"
            << std::setw(11) << "omega_V"
            << std::setw(11) << "Resid. r"
            << std::setw(11) << "Rel. eta"
            << std::setw(11) << "Recon. rho"
            << std::setw(11) << "cond(A)"
            << std::setw(13) << (show_log_det ? "sum(ln s_i)" : "Rel. Err(S)")
            << std::setw(13) << (show_log_det ? "s_min" : "Abs. Err(S)")
            << "\n";
  std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

  for (const auto& m : results) {
    std::cout << std::left << std::setw(26) << m.algo_name << std::right << std::fixed
              << std::setprecision(2) << std::setw(10) << m.time_ms << "  "
              << std::scientific << std::setprecision(2) << std::setw(9) << m.omega_U
              << "  " << std::setw(9) << m.omega_V << "  " << std::setw(9) << m.r
              << "  " << std::setw(9) << m.eta << "  " << std::setw(9) << m.rho
              << "  " << std::setw(9) << m.cond_num;

    if (show_log_det) {
      std::cout << "  " << std::fixed << std::setprecision(2) << std::setw(11) << m.log_det_val
                << "  " << std::scientific << std::setprecision(2) << std::setw(11) << m.sigma_min << "\n";
    } else {
      if (m.has_ground_truth) {
        std::cout << "  " << std::scientific << std::setprecision(2) << std::setw(11) << m.max_sigma_rel_err
                  << "  " << std::setw(11) << m.max_sigma_abs_err << "\n";
      } else {
        std::cout << "  " << std::setw(11) << "N/A"
                  << "  " << std::setw(11) << "N/A" << "\n";
      }
    }
  }
  std::cout << "========================================================================================================================\n";
}

void run_suite_on_instances(const std::string& family_name,
                            const std::vector<MatrixInstance>& instances,
                            bool show_log_det = false) {
  int num_instances = static_cast<int>(instances.size());
  std::cout << "\n>>> Testing " << family_name << " on " << num_instances << " distinct matrices...\n" << std::flush;

  const int NUM_ALGOS = 6;
  std::vector<std::vector<BenchmarkMetrics>> algo_metrics(NUM_ALGOS);

  for (int inst = 0; inst < num_instances; ++inst) {
    const auto& A = instances[inst].A;
    const auto& S_true = instances[inst].S_true;
    algo_metrics[0].push_back(run_halley_tridiag_dc(A, S_true));
    algo_metrics[1].push_back(run_halley_two_sided(A, S_true));
    algo_metrics[2].push_back(run_dgesdd(A, S_true));
    algo_metrics[3].push_back(run_dgesvd(A, S_true));
    algo_metrics[4].push_back(run_dbdsvdx(A, S_true));
    algo_metrics[5].push_back(run_jacobi(A, S_true));
  }

  // Aggregate results (compute mean and max worst-case across instances)
  std::vector<BenchmarkMetrics> mean_res(NUM_ALGOS);
  std::vector<BenchmarkMetrics> max_res(NUM_ALGOS);

  for (int a = 0; a < NUM_ALGOS; ++a) {
    mean_res[a].algo_name = algo_metrics[a][0].algo_name;
    mean_res[a].has_ground_truth = algo_metrics[a][0].has_ground_truth;
    max_res[a].algo_name = algo_metrics[a][0].algo_name;
    max_res[a].has_ground_truth = algo_metrics[a][0].has_ground_truth;

    double sum_time = 0.0, sum_wU = 0.0, sum_wV = 0.0, sum_r = 0.0, sum_eta = 0.0, sum_rho = 0.0;
    double sum_cond = 0.0, sum_rel = 0.0, sum_abs = 0.0, sum_log_det = 0.0, sum_smin = 0.0;

    double max_time = 0.0, max_wU = 0.0, max_wV = 0.0, max_r = 0.0, max_eta = 0.0, max_rho = 0.0;
    double max_cond = 0.0, max_rel = 0.0, max_abs = 0.0;

    int valid_count = 0;
    for (int inst = 0; inst < num_instances; ++inst) {
      const auto& m = algo_metrics[a][inst];
      if (m.is_finite) {
        sum_time += m.time_ms;
        sum_wU += m.omega_U;
        sum_wV += m.omega_V;
        sum_r += m.r;
        sum_eta += m.eta;
        sum_rho += m.rho;
        sum_cond += m.cond_num;
        sum_rel += m.max_sigma_rel_err;
        sum_abs += m.max_sigma_abs_err;
        sum_log_det += m.log_det_val;
        sum_smin += m.sigma_min;

        if (m.time_ms > max_time) max_time = m.time_ms;
        if (m.omega_U > max_wU) max_wU = m.omega_U;
        if (m.omega_V > max_wV) max_wV = m.omega_V;
        if (m.r > max_r) max_r = m.r;
        if (m.eta > max_eta) max_eta = m.eta;
        if (m.rho > max_rho) max_rho = m.rho;
        if (m.cond_num > max_cond) max_cond = m.cond_num;
        if (m.max_sigma_rel_err > max_rel) max_rel = m.max_sigma_rel_err;
        if (m.max_sigma_abs_err > max_abs) max_abs = m.max_sigma_abs_err;

        valid_count++;
      }
    }

    if (valid_count > 0) {
      mean_res[a].time_ms = sum_time / valid_count;
      mean_res[a].omega_U = sum_wU / valid_count;
      mean_res[a].omega_V = sum_wV / valid_count;
      mean_res[a].r = sum_r / valid_count;
      mean_res[a].eta = sum_eta / valid_count;
      mean_res[a].rho = sum_rho / valid_count;
      mean_res[a].cond_num = sum_cond / valid_count;
      mean_res[a].max_sigma_rel_err = sum_rel / valid_count;
      mean_res[a].max_sigma_abs_err = sum_abs / valid_count;
      mean_res[a].log_det_val = sum_log_det / valid_count;
      mean_res[a].sigma_min = sum_smin / valid_count;

      max_res[a].time_ms = max_time;
      max_res[a].omega_U = max_wU;
      max_res[a].omega_V = max_wV;
      max_res[a].r = max_r;
      max_res[a].eta = max_eta;
      max_res[a].rho = max_rho;
      max_res[a].cond_num = max_cond;
      max_res[a].max_sigma_rel_err = max_rel;
      max_res[a].max_sigma_abs_err = max_abs;
    }
  }

  print_table(family_name + " [MEAN across " + std::to_string(num_instances) + " matrices]", mean_res, show_log_det);
  print_table(family_name + " [MAX / WORST-CASE across " + std::to_string(num_instances) + " matrices]", max_res, show_log_det);
}

// Generate spectrum with customizable cluster centers and internal deltas
Eigen::VectorXd generate_clustered_spectrum(
    int p,
    const std::vector<double>& cluster_centers,
    const std::vector<double>& internal_deltas) {
  int K = static_cast<int>(cluster_centers.size());
  Eigen::VectorXd S_true(p);
  int cur = 0;
  for (int k = 0; k < K; ++k) {
    int n_k = (k == K - 1) ? (p - cur) : (p / K);
    double C_k = cluster_centers[k];
    double delta = (internal_deltas.size() == 1) ? internal_deltas[0] : internal_deltas[k];
    for (int j = 0; j < n_k; ++j) {
      double rel_offset = (n_k > 1) ? (static_cast<double>(j) / (n_k - 1)) * delta : 0.0;
      S_true(cur + j) = C_k * (1.0 + rel_offset);
    }
    cur += n_k;
  }
  std::sort(S_true.data(), S_true.data() + p, std::greater<double>());
  return S_true;
}

std::vector<MatrixInstance> create_matrix_instances(
    int m, int n, const Eigen::VectorXd& S_true, int num_trials, uint64_t base_seed) {
  int p = std::min(m, n);
  std::vector<MatrixInstance> instances(num_trials);
  for (int t = 0; t < num_trials; ++t) {
    std::mt19937_64 trial_rng(base_seed + t * 7);
    Eigen::MatrixXd U = generate_orthogonal_matrix(m, trial_rng);
    Eigen::MatrixXd V = generate_orthogonal_matrix(n, trial_rng);
    instances[t].A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
    clamp_min_entries(instances[t].A);
    instances[t].S_true = S_true;
  }
  return instances;
}

}  // namespace

int main() {
  std::cout << "####################################################################################################\n";
  std::cout << "#      DEDICATED BENCHMARK: Clustered High Dynamic Range Matrices (10^-6 to 10^+6)                 #\n";
  std::cout << "#      Scale: 500x500 | 10 Distinct Haar-Orthogonal Rotations Per Test                             #\n";
  std::cout << "#      Internal Delta: 1e-15 to 1e-10 | External Spacing: 1e-10 to 1e+2                            #\n";
  std::cout << "#      Algorithms: Halley (TridiagDC, TwoSidedJacobi), LAPACK (DGESDD, DGESVD, DBDSVDX), Jacobi   #\n";
  std::cout << "####################################################################################################\n";

  const int NUM_TRIALS = 10;
  const int m = 500;
  const int n = 500;
  const int p = std::min(m, n);

  // ==============================================================================================
  // Test 1: Unified Clustered HDR Matrix
  // Spanning 10^-6 to 10^+6 (cond ~ 1e12).
  // 10 clusters of 50 singular values each.
  // Internal deltas vary from 1e-15 to 1e-10 across clusters.
  // External spacings vary from 1e-10 to 1e+2 (100.0).
  // ==============================================================================================
  {
    const std::vector<double> cluster_centers = {
        1e-6,                    // C0: Bottom anchor (1e-6)
        1e-6 + 1e-10,            // C1: Spacing = 1e-10
        1e-4,                    // C2: Intermediate low
        1e-2,                    // C3: Intermediate
        1.0,                     // C4: Unit anchor
        1.0 + 1e-10,             // C5: Ultratight unit gap = 1e-10
        1.0 + 1e-4,              // C6: Moderate gap = 1e-4
        2.0,                     // C7: External gap = 1.0 (1e0)
        102.0,                   // C8: External gap = 100.0 (1e+2)
        1e6                      // C9: Top anchor (1e6)
    };

    const std::vector<double> internal_deltas = {
        1e-15,                   // C0: internal delta = 1e-15
        1e-14,                   // C1: internal delta = 1e-14
        1e-13,                   // C2: internal delta = 1e-13
        1e-12,                   // C3: internal delta = 1e-12
        1e-15,                   // C4: internal delta = 1e-15
        1e-13,                   // C5: internal delta = 1e-13
        1e-11,                   // C6: internal delta = 1e-11
        1e-10,                   // C7: internal delta = 1e-10
        1e-10,                   // C8: internal delta = 1e-10
        1e-10                    // C9: internal delta = 1e-10
    };

    Eigen::VectorXd S_true = generate_clustered_spectrum(p, cluster_centers, internal_deltas);
    auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 10001);

    run_suite_on_instances(
        "Test 1: Unified Clustered HDR Matrix (delta: 1e-15 -> 1e-10, spacing: 1e-10 -> 1e+2)",
        instances);
  }

  // ==============================================================================================
  // Test 2: Parametric Analysis of Internal Delta across the Clustered HDR Spectrum
  // Cluster centers fixed (10^-6 to 10^+6 with variable spacings 1e-10 to 1e+2).
  // Sweeping internal delta: 1e-15 vs 1e-12 vs 1e-10.
  // ==============================================================================================
  {
    const std::vector<double> cluster_centers = {
        1e-6, 1e-6 + 1e-10, 1e-4, 1e-2, 1.0, 1.0 + 1e-10, 1.0 + 1e-4, 2.0, 102.0, 1e6
    };

    // Sub-test 2A: Uniform ultratight internal delta = 1e-15
    {
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, cluster_centers, {1e-15});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 20001);
      run_suite_on_instances("Test 2A: HDR Spectrum with Uniform Internal Delta = 1e-15", instances);
    }

    // Sub-test 2B: Uniform medium internal delta = 1e-12
    {
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, cluster_centers, {1e-12});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 21001);
      run_suite_on_instances("Test 2B: HDR Spectrum with Uniform Internal Delta = 1e-12", instances);
    }

    // Sub-test 2C: Uniform relaxed internal delta = 1e-10
    {
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, cluster_centers, {1e-10});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 22001);
      run_suite_on_instances("Test 2C: HDR Spectrum with Uniform Internal Delta = 1e-10", instances);
    }
  }

  // ==============================================================================================
  // Test 3: Parametric Analysis of External Spacing across the Clustered HDR Spectrum
  // Fixing internal delta at 1e-15 and sweeping the external spacing around 1.0:
  // 1e-10 vs 1e-2 vs 1e+2.
  // ==============================================================================================
  {
    // Sub-test 3A: Ultratight external spacing = 1e-10
    {
      std::vector<double> centers = {1e-6, 1e-4, 1e-2, 1.0, 1.0 + 1e-10, 2.0, 10.0, 1e2, 1e4, 1e6};
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, centers, {1e-15});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 30001);
      run_suite_on_instances("Test 3A: HDR Spectrum with External Spacing = 1e-10 (around 1.0)", instances);
    }

    // Sub-test 3B: Moderate external spacing = 1e-2
    {
      std::vector<double> centers = {1e-6, 1e-4, 1e-2, 1.0, 1.0 + 1e-2, 2.0, 10.0, 1e2, 1e4, 1e6};
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, centers, {1e-15});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 31001);
      run_suite_on_instances("Test 3B: HDR Spectrum with External Spacing = 1e-2 (around 1.0)", instances);
    }

    // Sub-test 3C: Wide external spacing = 1e+2 (100.0)
    {
      std::vector<double> centers = {1e-6, 1e-4, 1e-2, 1.0, 101.0, 200.0, 1e3, 1e4, 1e5, 1e6};
      Eigen::VectorXd S_true = generate_clustered_spectrum(p, centers, {1e-15});
      auto instances = create_matrix_instances(m, n, S_true, NUM_TRIALS, 32001);
      run_suite_on_instances("Test 3C: HDR Spectrum with External Spacing = 1e+2 (around 1.0 -> 101.0)", instances);
    }
  }

  return 0;
}

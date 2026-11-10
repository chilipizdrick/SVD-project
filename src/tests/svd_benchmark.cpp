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

// Compute standard SVD error metrics for any m x n matrix
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

}  // namespace

int main() {
  std::cout << "####################################################################################################\n";
  std::cout << "#      UNIFIED SVD BENCHMARK: Halley vs LAPACK (DGESDD, DGESVD, DBDSVDX) & Jacobi                 #\n";
  std::cout << "#      Pathological Families, Clustered Spectra, 500x500 Scale                                     #\n";
  std::cout << "#      Consistency Evaluation: 10 Distinct Matrices Per Family                                    #\n";
  std::cout << "####################################################################################################\n";

  const int NUM_TRIALS = 10;
  int m = 500;
  int n = 500;
  int p = std::min(m, n);

  std::cout << "\n\n>>>>>>>>>>>>>>>>>>>>>>>>  Square 500x500 (10 matrices per family)  <<<<<<<<<<<<<<<<<<<<<<<<\n";

  // --------------------------------------------------------------------------------
  // Family 1: Twin Mega-Clusters (10 instances)
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    Eigen::VectorXd S_true(p);
    int half = p / 2;
    for (int i = 0; i < half; ++i) S_true(i) = 1.0 + 1e-14;
    for (int i = half; i < p; ++i) S_true(i) = 1.0;

    for (int t = 0; t < NUM_TRIALS; ++t) {
      std::mt19937_64 trial_rng(1001 + t * 7);
      Eigen::MatrixXd U = generate_orthogonal_matrix(m, trial_rng);
      Eigen::MatrixXd V = generate_orthogonal_matrix(n, trial_rng);
      instances[t].A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
      clamp_min_entries(instances[t].A);
      instances[t].S_true = S_true;
    }
    run_suite_on_instances("Family 1: Twin Mega-Clusters (1.0 vs 1.0 + 1e-14)", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 2: 10 Tight Clusters (10 instances)
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    Eigen::VectorXd S_true(p);
    for (int i = 0; i < p; ++i) {
      int cluster_id = i % 10;
      double base = 1.0 + cluster_id * 1e-14;
      double intra = (i / 10) * 1e-15;
      S_true(i) = base + intra;
    }
    std::sort(S_true.data(), S_true.data() + p, std::greater<double>());

    for (int t = 0; t < NUM_TRIALS; ++t) {
      std::mt19937_64 trial_rng(2001 + t * 7);
      Eigen::MatrixXd U = generate_orthogonal_matrix(m, trial_rng);
      Eigen::MatrixXd V = generate_orthogonal_matrix(n, trial_rng);
      instances[t].A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
      clamp_min_entries(instances[t].A);
      instances[t].S_true = S_true;
    }
    run_suite_on_instances("Family 2: 10 Tight Clusters (delta = 1e-15, spacing = 1e-14)", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 3: Wide Dynamic Range (10 instances)
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    Eigen::VectorXd S_true(p);
    for (int i = 0; i < p; ++i) {
      double t_frac = static_cast<double>(i) / std::max(1, p - 1);
      S_true(i) = std::pow(10.0, 9.0 - 18.0 * t_frac);
    }

    for (int t = 0; t < NUM_TRIALS; ++t) {
      std::mt19937_64 trial_rng(3001 + t * 7);
      Eigen::MatrixXd U = generate_orthogonal_matrix(m, trial_rng);
      Eigen::MatrixXd V = generate_orthogonal_matrix(n, trial_rng);
      instances[t].A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
      clamp_min_entries(instances[t].A);
      instances[t].S_true = S_true;
    }
    run_suite_on_instances("Family 3: Wide Dynamic Range (10^-9 to 10^+9, cond ~ 1e18)", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 4: Läuchli Matrix (10 instances with varied mu in [1e-9, 1e-7])
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      double mu = 1e-8 * std::pow(10.0, -1.0 + 2.0 * t / (NUM_TRIALS - 1));
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
      A.row(0).setOnes();
      for (int i = 1; i < p; ++i) {
        A(i, i) = mu;
      }
      clamp_min_entries(A);
      instances[t].A = A;
    }
    run_suite_on_instances("Family 4: Lauchli Matrix (Row 0 = 1, diag = mu in [1e-9, 1e-7])", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 5: Golub Integer Matrix (10 independent random integer factorizations)
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    std::uniform_int_distribution<int> int_dist(-10, 10);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      std::mt19937_64 trial_rng(5001 + t * 13);
      Eigen::MatrixXd L = Eigen::MatrixXd::Zero(m, p);
      Eigen::MatrixXd U_mat = Eigen::MatrixXd::Zero(p, n);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
          if (i == j) L(i, j) = 1.0;
          else if (i > j) L(i, j) = static_cast<double>(int_dist(trial_rng));
        }
      }
      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < n; ++j) {
          if (i == j) U_mat(i, j) = 1.0;
          else if (j > i) U_mat(i, j) = static_cast<double>(int_dist(trial_rng));
        }
      }
      instances[t].A = L * U_mat;
      clamp_min_entries(instances[t].A);
    }
    run_suite_on_instances("Family 5: Golub Integer Matrix (A = L * U, all entries in Z)", instances, (m == n));
  }

  // --------------------------------------------------------------------------------
  // Family 6: Kahan Matrix (10 instances with varied theta in [1.10, 1.28])
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      double theta = 1.10 + 0.02 * t;
      double s = std::sin(theta);
      double c = std::cos(theta);
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < p; ++i) {
        double scale = std::pow(s, i);
        A(i, i) = scale;
        for (int j = i + 1; j < n; ++j) {
          A(i, j) = -c * scale;
        }
      }
      clamp_min_entries(A);
      instances[t].A = A;
    }
    run_suite_on_instances("Family 6: Kahan Matrix (theta in [1.10, 1.28], upper triangular)", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 7: Hilbert Matrix (10 instances with varied Cauchy shift alpha in [0.5, 1.4])
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      double alpha = 0.5 + 0.1 * t;
      Eigen::MatrixXd A(m, n);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i, j) = 1.0 / (static_cast<double>(i + j) + alpha);
        }
      }
      clamp_min_entries(A);
      instances[t].A = A;
    }
    run_suite_on_instances("Family 7: Hilbert Matrix (Cauchy shift alpha in [0.5, 1.4])", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 8: Non-Normal Jordan Block (10 instances with varied eps in [1e-15, 1e-13])
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      double eps = 1e-14 * std::pow(10.0, -1.0 + 2.0 * t / (NUM_TRIALS - 1));
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < p - 1; ++i) {
        A(i, i + 1) = 1.0;
      }
      A(p - 1, 0) = eps;
      clamp_min_entries(A);

      Eigen::VectorXd S_true(p);
      for (int i = 0; i < p - 1; ++i) S_true(i) = 1.0;
      S_true(p - 1) = eps;

      instances[t].A = A;
      instances[t].S_true = S_true;
    }
    run_suite_on_instances("Family 8: Non-Normal Jordan Block (perturbation eps in [1e-15, 1e-13])", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 9: Golub-Kahan Bidiagonal (10 instances with varied diag in [0.38, 0.62])
  // --------------------------------------------------------------------------------
  {
    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      double d = 0.38 + (0.62 - 0.38) * t / (NUM_TRIALS - 1);
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < p; ++i) {
        A(i, i) = d;
        if (i < n - 1) A(i, i + 1) = 1.0;
      }
      clamp_min_entries(A);
      instances[t].A = A;
    }
    run_suite_on_instances("Family 9: Golub-Kahan Bidiagonal (diag in [0.38, 0.62], superdiag = 1.0)", instances);
  }

  // --------------------------------------------------------------------------------
  // Family 10: Clustered High Dynamic Range (10 independent random instances)
  // --------------------------------------------------------------------------------
  {
    std::vector<double> cluster_centers = {
        1e-6,                    // Cluster 0: Bottom anchor (1e-6)
        1e-6 * (1.0 + 1e-8),     // Cluster 1: Close to bottom (rel gap 1e-8)
        1e-3,                    // Cluster 2: Intermediate low
        1e-3 * (1.0 + 1e-4),     // Cluster 3: Rel gap 1e-4
        1.0,                     // Cluster 4: Unit anchor
        1.0 + 1e-14,             // Cluster 5: Ultratight external gap (1e-14)
        1.0 + 1e-1,              // Cluster 6: Moderate gap (0.1)
        2.0,                     // Cluster 7: External gap 1.0 (1e0)
        1e6 * (1.0 - 1e-14),     // Cluster 8: Ultratight top cluster (rel gap 1e-14)
        1e6                      // Cluster 9: Top anchor (1e6)
    };

    int K = static_cast<int>(cluster_centers.size());
    Eigen::VectorXd S_true(p);
    int cur = 0;
    for (int k = 0; k < K; ++k) {
      int n_k = (k == K - 1) ? (p - cur) : (p / K);
      double C_k = cluster_centers[k];
      for (int j = 0; j < n_k; ++j) {
        double rel_offset = (n_k > 1) ? (static_cast<double>(j) / (n_k - 1)) * 1e-15 : 0.0;
        S_true(cur + j) = C_k * (1.0 + rel_offset);
      }
      cur += n_k;
    }
    std::sort(S_true.data(), S_true.data() + p, std::greater<double>());

    std::vector<MatrixInstance> instances(NUM_TRIALS);
    for (int t = 0; t < NUM_TRIALS; ++t) {
      std::mt19937_64 trial_rng(10001 + t * 7);
      Eigen::MatrixXd U = generate_orthogonal_matrix(m, trial_rng);
      Eigen::MatrixXd V = generate_orthogonal_matrix(n, trial_rng);
      instances[t].A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
      clamp_min_entries(instances[t].A);
      instances[t].S_true = S_true;
    }

    run_suite_on_instances("Family 10: Clustered High Dynamic Range (10^-6 to 10^6, internal delta = 1e-15, external gaps 1e-14 to 1e0)",
                           instances);
  }

  return 0;
}

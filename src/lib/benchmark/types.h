#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace SVD_Project::Benchmark {

inline constexpr int kSchemaVersion = 1;
inline constexpr int kMetricsSchemaVersion = 2;

enum class ConcurrencyModel {
  Serial = 0,          // Pure single-threaded implementation
  LapackInternal = 1,  // Externally parallelized via LAPACK/BLAS/OpenMP
  LapackDiag = 2       // Single-threaded polar phase + LAPACK diagonalizer
};

inline std::string to_string(ConcurrencyModel m) {
  switch (m) {
    case ConcurrencyModel::Serial:
      return "serial";
    case ConcurrencyModel::LapackInternal:
      return "lapack_internal";
    case ConcurrencyModel::LapackDiag:
      return "lapack_diag";
  }
  return "serial";
}

inline ConcurrencyModel concurrency_from_string(const std::string& s) {
  if (s == "lapack_internal") return ConcurrencyModel::LapackInternal;
  if (s == "lapack_diag") return ConcurrencyModel::LapackDiag;
  return ConcurrencyModel::Serial;
}

inline bool is_internally_parallel(ConcurrencyModel m) {
  return m != ConcurrencyModel::Serial;
}

enum class RunStatus { Ok = 0, Failed = 1, NonFinite = 2, Unsupported = 3 };

inline std::string to_string(RunStatus s) {
  switch (s) {
    case RunStatus::Ok:
      return "ok";
    case RunStatus::Failed:
      return "failed";
    case RunStatus::NonFinite:
      return "non_finite";
    case RunStatus::Unsupported:
      return "unsupported";
  }
  return "failed";
}

inline RunStatus status_from_string(const std::string& s) {
  if (s == "ok") return RunStatus::Ok;
  if (s == "non_finite") return RunStatus::NonFinite;
  if (s == "unsupported") return RunStatus::Unsupported;
  return RunStatus::Failed;
}

struct MatrixSize {
  int rows{0};
  int cols{0};

  std::string label() const {
    return std::to_string(rows) + "x" + std::to_string(cols);
  }

  bool operator==(const MatrixSize& o) const {
    return rows == o.rows && cols == o.cols;
  }
  bool operator<(const MatrixSize& o) const {
    int64_t area_a = static_cast<int64_t>(rows) * cols;
    int64_t area_b = static_cast<int64_t>(o.rows) * o.cols;
    if (area_a != area_b) return area_a < area_b;
    if (rows != o.rows) return rows < o.rows;
    return cols < o.cols;
  }
};

inline std::optional<MatrixSize> parse_matrix_size(const std::string& s) {
  auto pos = s.find('x');
  if (pos == std::string::npos) pos = s.find('X');
  try {
    if (pos == std::string::npos) {
      int n = std::stoi(s);
      if (n <= 0) return std::nullopt;
      return MatrixSize{n, n};
    }
    int r = std::stoi(s.substr(0, pos));
    int c = std::stoi(s.substr(pos + 1));
    if (r <= 0 || c <= 0) return std::nullopt;
    return MatrixSize{r, c};
  } catch (...) {
    return std::nullopt;
  }
}

struct TestCoord {
  std::string algorithm;
  std::string diagonalizer;  // "none" for non-polar baselines
  std::string matrix_kind;
  MatrixSize size;
  std::string dtype{"double"};
  int trial{0};

  std::string coord_key() const {
    return algorithm + "|" + diagonalizer + "|" + matrix_kind + "|" +
           size.label() + "|" + dtype + "|" + std::to_string(trial);
  }

  std::string solver_key() const {
    if (diagonalizer == "none" || diagonalizer.empty()) return algorithm;
    return algorithm + "/" + diagonalizer;
  }

  std::string problem_key() const {
    return matrix_kind + " (" + size.label() + ")";
  }
};

struct MetricInfo {
  std::string key;
  std::string label;
  std::string unit;
  bool lower_is_better{true};
  bool directional{true};  // false for neutral descriptors like cond_num
};

inline const std::vector<MetricInfo>& all_metrics_info() {
  static const std::vector<MetricInfo> kMetrics = {
      {"time_ms", "Time (ms)", "ms", true, true},
      {"omega_U", "omega_U", "", true, true},
      {"omega_V", "omega_V", "", true, true},
      {"r", "r", "", true, true},
      {"eta", "eta", "", true, true},
      {"rho", "rho", "", true, true},
      {"kappa", "kappa", "", true, false},
      {"delta", "delta", "", true, true},
      {"Delta", "Delta", "", true, true},
      {"log_det", "log_det", "", true, false},
      {"sigma_min", "sigma_min", "", true, false},
      {"sigma_max", "sigma_max", "", true, false},
  };
  return kMetrics;
}

inline std::string canonical_metric_key(const std::string& key) {
  if (key == "resid_r") return "r";
  if (key == "rel_eta") return "eta";
  if (key == "recon_rho") return "rho";
  if (key == "cond_num") return "kappa";
  if (key == "rel_err_s") return "delta";
  if (key == "abs_err_s") return "Delta";
  return key;
}

inline std::optional<MetricInfo> find_metric_info(const std::string& key) {
  std::string ck = canonical_metric_key(key);
  for (const auto& m : all_metrics_info()) {
    if (m.key == ck) return m;
  }
  return std::nullopt;
}

struct MetricValues {
  double time_ms{std::numeric_limits<double>::quiet_NaN()};
  double omega_U{std::numeric_limits<double>::quiet_NaN()};
  double omega_V{std::numeric_limits<double>::quiet_NaN()};
  double r{std::numeric_limits<double>::quiet_NaN()};
  double eta{std::numeric_limits<double>::quiet_NaN()};
  double rho{std::numeric_limits<double>::quiet_NaN()};
  double kappa{std::numeric_limits<double>::quiet_NaN()};
  double delta{std::numeric_limits<double>::quiet_NaN()};
  double Delta{std::numeric_limits<double>::quiet_NaN()};
  double log_det{std::numeric_limits<double>::quiet_NaN()};
  double sigma_min{std::numeric_limits<double>::quiet_NaN()};
  double sigma_max{std::numeric_limits<double>::quiet_NaN()};

  double get(const std::string& raw_key) const {
    std::string key = canonical_metric_key(raw_key);
    if (key == "time_ms") return time_ms;
    if (key == "omega_U") return omega_U;
    if (key == "omega_V") return omega_V;
    if (key == "r") return r;
    if (key == "eta") return eta;
    if (key == "rho") return rho;
    if (key == "kappa") return kappa;
    if (key == "delta") return delta;
    if (key == "Delta") return Delta;
    if (key == "log_det") return log_det;
    if (key == "sigma_min") return sigma_min;
    if (key == "sigma_max") return sigma_max;
    return std::numeric_limits<double>::quiet_NaN();
  }

  void set(const std::string& raw_key, double val) {
    std::string key = canonical_metric_key(raw_key);
    if (key == "time_ms") time_ms = val;
    else if (key == "omega_U") omega_U = val;
    else if (key == "omega_V") omega_V = val;
    else if (key == "r") r = val;
    else if (key == "eta") eta = val;
    else if (key == "rho") rho = val;
    else if (key == "kappa") kappa = val;
    else if (key == "delta") delta = val;
    else if (key == "Delta") Delta = val;
    else if (key == "log_det") log_det = val;
    else if (key == "sigma_min") sigma_min = val;
    else if (key == "sigma_max") sigma_max = val;
  }
};

struct BenchmarkRecord {
  TestCoord coord;
  uint64_t seed{0};
  std::string matrix_hash;
  ConcurrencyModel concurrency{ConcurrencyModel::Serial};
  bool internally_parallel{false};
  int runner_jobs{1};
  bool blas_threads_pinned{false};
  RunStatus status{RunStatus::Ok};
  std::string error_msg;
  bool has_ground_truth{false};
  MetricValues metrics;
};

struct MatrixInstance {
  Eigen::MatrixXd A;
  Eigen::VectorXd S_true;
  bool has_ground_truth{false};
  uint64_t seed{0};
  std::string matrix_hash;
};

struct SVDOutput {
  Eigen::MatrixXd U;
  Eigen::VectorXd S;
  Eigen::MatrixXd V;
};

}  // namespace SVD_Project::Benchmark

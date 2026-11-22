#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "metrics.h"
#include "types.h"

namespace SVD_Project::Benchmark {

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t derive_instance_seed(uint64_t base_seed,
                                     const std::string& kind, int m, int n,
                                     int trial) {
  uint64_t h = fnv1a_update(14695981039346656037ULL, kind.data(), kind.size());
  uint64_t combined = splitmix64(base_seed ^ h);
  combined = splitmix64(combined ^ (static_cast<uint64_t>(m) << 32) ^
                        static_cast<uint64_t>(n));
  combined = splitmix64(combined ^ static_cast<uint64_t>(trial + 1));
  return combined;
}

inline void clamp_min_entries(Eigen::MatrixXd& A) {
  for (Eigen::Index j = 0; j < A.cols(); ++j) {
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
      if (std::abs(A(i, j)) < 1e-16) {
        A(i, j) = (A(i, j) >= 0.0) ? 1e-16 : -1e-16;
      }
    }
  }
}

// Haar-distributed random orthogonal matrix via Householder QR with R_ii > 0
inline Eigen::MatrixXd generate_orthogonal_matrix(int n, std::mt19937_64& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      M(i, j) = dist(rng);
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
  for (int i = 0; i < n; ++i) {
    if (R(i, i) < 0.0) {
      Q.col(i) = -Q.col(i);
    }
  }
  return Q;
}

inline MatrixInstance build_from_spectrum(int m, int n,
                                          const Eigen::VectorXd& S_true_in,
                                          uint64_t seed) {
  int p = std::min(m, n);
  Eigen::VectorXd S_true = S_true_in;
  std::sort(S_true.data(), S_true.data() + p, std::greater<double>());

  std::mt19937_64 rng(seed);
  Eigen::MatrixXd U = generate_orthogonal_matrix(m, rng);
  Eigen::MatrixXd V = generate_orthogonal_matrix(n, rng);

  MatrixInstance inst;
  inst.A = U.leftCols(p) * S_true.asDiagonal() * V.leftCols(p).transpose();
  clamp_min_entries(inst.A);
  inst.S_true = std::move(S_true);
  inst.has_ground_truth = true;
  inst.seed = seed;
  inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, true);
  return inst;
}

struct MatrixFamilyEntry {
  std::string name;
  std::string description;
  bool has_ground_truth{false};
  std::function<MatrixInstance(int m, int n, int trial, int total_trials,
                               uint64_t seed)>
      generator;
};

class MatrixRegistry {
 public:
  static const MatrixRegistry& instance() {
    static const MatrixRegistry reg = create_default();
    return reg;
  }

  const std::vector<MatrixFamilyEntry>& entries() const { return entries_; }

  const MatrixFamilyEntry* find(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) return nullptr;
    return &entries_[it->second];
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& e : entries_) out.push_back(e.name);
    return out;
  }

  void register_family(MatrixFamilyEntry entry) {
    index_[entry.name] = entries_.size();
    entries_.push_back(std::move(entry));
  }

 private:
  std::vector<MatrixFamilyEntry> entries_;
  std::map<std::string, size_t> index_;

  static MatrixRegistry create_default() {
    MatrixRegistry reg;

    // 1. Twin Mega-Clusters
    reg.register_family(
        {"twin_clusters",
         "Two mega-clusters: half at 1.0 + 1e-14, half at 1.0", true,
         [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           Eigen::VectorXd S(p);
           int half = p / 2;
           for (int i = 0; i < half; ++i) S(i) = 1.0 + 1e-14;
           for (int i = half; i < p; ++i) S(i) = 1.0;
           return build_from_spectrum(m, n, S, seed);
         }});

    // 2. 10 Tight Clusters
    reg.register_family(
        {"tight_10_clusters",
         "10 tight clusters around 1.0 (delta=1e-15, spacing=1e-14)", true,
         [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           Eigen::VectorXd S(p);
           for (int i = 0; i < p; ++i) {
             int cid = i % 10;
             double base = 1.0 + cid * 1e-14;
             double intra = (i / 10) * 1e-15;
             S(i) = base + intra;
           }
           return build_from_spectrum(m, n, S, seed);
         }});

    // 3. Wide Dynamic Range (10^-9 to 10^+9)
    reg.register_family(
        {"wide_hdr", "Log-spaced spectrum from 1e-9 to 1e+9 (cond ~ 1e18)",
         true, [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           Eigen::VectorXd S(p);
           for (int i = 0; i < p; ++i) {
             double t = static_cast<double>(i) / std::max(1, p - 1);
             S(i) = std::pow(10.0, 9.0 - 18.0 * t);
           }
           return build_from_spectrum(m, n, S, seed);
         }});

    // 4. Lauchli Matrix
    reg.register_family(
        {"lauchli", "Lauchli matrix (row 0 = 1, diag = mu in [1e-9, 1e-7])",
         false, [](int m, int n, int trial, int total, uint64_t seed) {
           int p = std::min(m, n);
           double frac =
               (total > 1) ? static_cast<double>(trial) / (total - 1) : 0.5;
           double mu = 1e-8 * std::pow(10.0, -1.0 + 2.0 * frac);
           Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
           A.row(0).setOnes();
           for (int i = 1; i < p; ++i) A(i, i) = mu;
           clamp_min_entries(A);
           MatrixInstance inst;
           inst.A = std::move(A);
           inst.has_ground_truth = false;
           inst.seed = seed;
           inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, false);
           return inst;
         }});

    // 5. Golub Integer Matrix (A = L * U with integer entries)
    reg.register_family(
        {"golub_int", "Golub integer matrix A = L*U (unit triangular in [-10,10])",
         false, [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           std::mt19937_64 rng(seed);
           std::uniform_int_distribution<int> dist(-10, 10);
           Eigen::MatrixXd L = Eigen::MatrixXd::Zero(m, p);
           Eigen::MatrixXd U = Eigen::MatrixXd::Zero(p, n);
           for (int i = 0; i < m; ++i) {
             for (int j = 0; j < p; ++j) {
               if (i == j) L(i, j) = 1.0;
               else if (i > j) L(i, j) = static_cast<double>(dist(rng));
             }
           }
           for (int i = 0; i < p; ++i) {
             for (int j = 0; j < n; ++j) {
               if (i == j) U(i, j) = 1.0;
               else if (j > i) U(i, j) = static_cast<double>(dist(rng));
             }
           }
           MatrixInstance inst;
           inst.A = L * U;
           clamp_min_entries(inst.A);
           inst.has_ground_truth = false;
           inst.seed = seed;
           inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, false);
           return inst;
         }});

    // 6. Kahan Matrix
    reg.register_family(
        {"kahan", "Kahan upper-triangular matrix (theta in [1.10, 1.28])",
         false, [](int m, int n, int trial, int total, uint64_t seed) {
           int p = std::min(m, n);
           double frac =
               (total > 1) ? static_cast<double>(trial) / (total - 1) : 0.5;
           double theta = 1.10 + 0.18 * frac;
           double s = std::sin(theta);
           double c = std::cos(theta);
           Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
           for (int i = 0; i < p; ++i) {
             double scale = std::pow(s, i);
             A(i, i) = scale;
             for (int j = i + 1; j < n; ++j) A(i, j) = -c * scale;
           }
           clamp_min_entries(A);
           MatrixInstance inst;
           inst.A = std::move(A);
           inst.has_ground_truth = false;
           inst.seed = seed;
           inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, false);
           return inst;
         }});

    // 7. Hilbert / Cauchy Matrix
    reg.register_family(
        {"hilbert", "Hilbert-Cauchy matrix 1/(i+j+alpha), alpha in [0.5, 1.4]",
         false, [](int m, int n, int trial, int total, uint64_t seed) {
           double frac =
               (total > 1) ? static_cast<double>(trial) / (total - 1) : 0.5;
           double alpha = 0.5 + 0.9 * frac;
           Eigen::MatrixXd A(m, n);
           for (int i = 0; i < m; ++i) {
             for (int j = 0; j < n; ++j) {
               A(i, j) = 1.0 / (static_cast<double>(i + j) + alpha);
             }
           }
           clamp_min_entries(A);
           MatrixInstance inst;
           inst.A = std::move(A);
           inst.has_ground_truth = false;
           inst.seed = seed;
           inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, false);
           return inst;
         }});

    // 8. Non-Normal Jordan Block
    reg.register_family(
        {"jordan", "Perturbed Jordan block (superdiag=1, corner eps in [1e-15,1e-13])",
         true, [](int m, int n, int trial, int total, uint64_t seed) {
           int p = std::min(m, n);
           double frac =
               (total > 1) ? static_cast<double>(trial) / (total - 1) : 0.5;
           double eps = 1e-14 * std::pow(10.0, -1.0 + 2.0 * frac);
           Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
           for (int i = 0; i < p - 1; ++i) A(i, i + 1) = 1.0;
           if (p > 0) A(p - 1, 0) = eps;
           clamp_min_entries(A);

           Eigen::VectorXd S(p);
           for (int i = 0; i < p - 1; ++i) S(i) = 1.0;
           if (p > 0) S(p - 1) = eps;

           MatrixInstance inst;
           inst.A = std::move(A);
           inst.S_true = std::move(S);
           inst.has_ground_truth = (m == n);
           inst.seed = seed;
           inst.matrix_hash =
               hash_matrix_instance(inst.A, inst.S_true, inst.has_ground_truth);
           return inst;
         }});

    // 9. Golub-Kahan Bidiagonal
    reg.register_family(
        {"golub_kahan", "Golub-Kahan bidiagonal (diag in [0.38, 0.62], superdiag=1)",
         false, [](int m, int n, int trial, int total, uint64_t seed) {
           int p = std::min(m, n);
           double frac =
               (total > 1) ? static_cast<double>(trial) / (total - 1) : 0.5;
           double d = 0.38 + (0.62 - 0.38) * frac;
           Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
           for (int i = 0; i < p; ++i) {
             A(i, i) = d;
             if (i < n - 1) A(i, i + 1) = 1.0;
           }
           clamp_min_entries(A);
           MatrixInstance inst;
           inst.A = std::move(A);
           inst.has_ground_truth = false;
           inst.seed = seed;
           inst.matrix_hash = hash_matrix_instance(inst.A, inst.S_true, false);
           return inst;
         }});

    // 10. Clustered High Dynamic Range (10^-6 to 10^+6, delta 1e-15..1e-10, spacing 1e-10..1e+2)
    reg.register_family(
        {"clustered_hdr",
         "Clustered HDR 1e-6..1e6 (internal delta 1e-15..1e-10, spacing 1e-10..1e2)",
         true, [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           const std::vector<double> centers = {
               1e-6,        1e-6 + 1e-10, 1e-4, 1e-2, 1.0,
               1.0 + 1e-10, 1.0 + 1e-4,   2.0,  102.0, 1e6};
           const std::vector<double> deltas = {
               1e-15, 1e-14, 1e-13, 1e-12, 1e-15,
               1e-13, 1e-11, 1e-10, 1e-10, 1e-10};
           int K = static_cast<int>(centers.size());
           Eigen::VectorXd S(p);
           int cur = 0;
           for (int k = 0; k < K && cur < p; ++k) {
             int n_k = (k == K - 1) ? (p - cur) : std::max(1, p / K);
             if (cur + n_k > p) n_k = p - cur;
             double C_k = centers[k];
             double delta = deltas[k];
             for (int j = 0; j < n_k; ++j) {
               double rel =
                   (n_k > 1) ? (static_cast<double>(j) / (n_k - 1)) * delta : 0.0;
               S(cur + j) = C_k * (1.0 + rel);
             }
             cur += n_k;
           }
           return build_from_spectrum(m, n, S, seed);
         }});

    // 11. Random Uniform Spectrum
    reg.register_family(
        {"random_uniform", "Uniform singular values in [1.0, 100.0] with Haar U, V",
         true, [](int m, int n, int /*trial*/, int /*total*/, uint64_t seed) {
           int p = std::min(m, n);
           std::mt19937_64 rng(splitmix64(seed ^ 0x55aa55aaULL));
           std::uniform_real_distribution<double> dist(1.0, 100.0);
           Eigen::VectorXd S(p);
           for (int i = 0; i < p; ++i) S(i) = dist(rng);
           if (p >= 2) {
             S(0) = 1.0;
             S(1) = 100.0;
           }
           return build_from_spectrum(m, n, S, seed);
         }});

    return reg;
  }
};

}  // namespace SVD_Project::Benchmark

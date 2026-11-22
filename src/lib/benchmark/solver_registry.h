#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../dbdsvdx/dbdsvdx.h"
#include "../dgesdd/dgesdd.h"
#include "../dgesvd/dgesvd.h"
#include "../polar_decomposition/polar_svd.h"
#include "types.h"

namespace SVD_Project::Benchmark {

struct SolverEntry {
  std::string algorithm;
  std::string diagonalizer;  // "none" for non-polar baselines
  std::string dtype{"double"};
  std::string description;
  bool is_polar{false};
  ConcurrencyModel concurrency{ConcurrencyModel::Serial};
  std::function<SVDOutput(const Eigen::MatrixXd&)> solve;

  std::string key() const { return algorithm + "|" + diagonalizer + "|" + dtype; }
};

class SolverRegistry {
 public:
  static const SolverRegistry& instance() {
    static const SolverRegistry reg = create_default();
    return reg;
  }

  const std::vector<SolverEntry>& entries() const { return entries_; }

  const SolverEntry* find(const std::string& algo, const std::string& diag,
                          const std::string& dtype = "double") const {
    auto it = index_.find(algo + "|" + diag + "|" + dtype);
    if (it == index_.end()) return nullptr;
    return &entries_[it->second];
  }

  std::vector<std::string> algorithms() const {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& e : entries_) {
      if (seen.insert(e.algorithm).second) out.push_back(e.algorithm);
    }
    return out;
  }

  std::vector<std::string> diagonalizers() const {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& e : entries_) {
      if (seen.insert(e.diagonalizer).second) out.push_back(e.diagonalizer);
    }
    return out;
  }

  std::vector<std::string> polar_diagonalizers() const {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& e : entries_) {
      if (e.is_polar && seen.insert(e.diagonalizer).second) {
        out.push_back(e.diagonalizer);
      }
    }
    return out;
  }

  void register_polar(const std::string& algo, const std::string& diag,
                      ConcurrencyModel concurrency,
                      const std::string& description,
                      std::function<SVDOutput(const Eigen::MatrixXd&)> fn) {
    SolverEntry e;
    e.algorithm = algo;
    e.diagonalizer = diag;
    e.dtype = "double";
    e.description = description;
    e.is_polar = true;
    e.concurrency = concurrency;
    e.solve = std::move(fn);
    index_[e.key()] = entries_.size();
    entries_.push_back(std::move(e));
  }

  void register_baseline(const std::string& algo, ConcurrencyModel concurrency,
                         const std::string& description,
                         std::function<SVDOutput(const Eigen::MatrixXd&)> fn) {
    SolverEntry e;
    e.algorithm = algo;
    e.diagonalizer = "none";
    e.dtype = "double";
    e.description = description;
    e.is_polar = false;
    e.concurrency = concurrency;
    e.solve = std::move(fn);
    index_[e.key()] = entries_.size();
    entries_.push_back(std::move(e));
  }

 private:
  std::vector<SolverEntry> entries_;
  std::map<std::string, size_t> index_;

  static SolverRegistry create_default() {
    SolverRegistry reg;

    // ------------------------------------------------------------------------
    // Polar-decomposition SVD strategies (Halley, Newton, Newton-Accelerated)
    // ------------------------------------------------------------------------
    auto register_polar_family =
        [&]<SVD_Project::PolarAlgorithm PolarAlg>(
            const std::string& algo_name, const std::string& polar_label) {
          auto register_one =
              [&]<SVD_Project::DiagonalizationAlgorithm DiagAlg>(
                  const std::string& diag_name, ConcurrencyModel concurrency,
                  const std::string& diag_desc) {
                reg.register_polar(
                    algo_name, diag_name, concurrency,
                    polar_label + " polar + " + diag_desc,
                    [](const Eigen::MatrixXd& A) -> SVDOutput {
                      SVD_Project::Polar_SVD<Eigen::MatrixXd, PolarAlg, DiagAlg>
                          svd;
                      svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                      return {svd.matrixU(), svd.singularValues(),
                              svd.matrixV()};
                    });
              };

          register_one.template operator()<
              SVD_Project::DiagonalizationAlgorithm::TridiagDC>(
              "tridiag_dc", ConcurrencyModel::LapackDiag,
              "Householder tridiagonal D&C (LAPACK dsyevd)");
          register_one.template operator()<
              SVD_Project::DiagonalizationAlgorithm::TwoSidedJacobi>(
              "two_sided_jacobi", ConcurrencyModel::Serial,
              "classical two-sided cyclic Jacobi on H");
          register_one.template operator()<
              SVD_Project::DiagonalizationAlgorithm::CholeskyOneSided>(
              "cholesky_onesided", ConcurrencyModel::Serial,
              "pivoted Cholesky & one-sided Jacobi on H");
          register_one.template operator()<
              SVD_Project::DiagonalizationAlgorithm::QDWH_EIG>(
              "qdwh_eig", ConcurrencyModel::Serial,
              "spectral divide & conquer (QDWH-EIG) on H");
        };

    register_polar_family.template operator()<
        SVD_Project::PolarAlgorithm::Halley>("halley", "Halley (QDWH)");
    register_polar_family.template operator()<
        SVD_Project::PolarAlgorithm::Newton>("newton", "Newton");
    register_polar_family.template operator()<
        SVD_Project::PolarAlgorithm::NewtonAccelerated>(
        "newton_accel", "Newton accelerated");

    // ------------------------------------------------------------------------
    // Non-Polar Baselines (diagonalizer = "none")
    // ------------------------------------------------------------------------
    reg.register_baseline(
        "dgesdd", ConcurrencyModel::LapackInternal,
        "LAPACK dgesdd (bidiagonal divide & conquer)",
        [](const Eigen::MatrixXd& A) -> SVDOutput {
          DGESDD_SVD<Eigen::MatrixXd> svd(
              A, Eigen::ComputeFullU | Eigen::ComputeFullV);
          return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
        });

    reg.register_baseline(
        "dgesvd", ConcurrencyModel::LapackInternal,
        "LAPACK dgesvd (bidiagonal implicit QR)",
        [](const Eigen::MatrixXd& A) -> SVDOutput {
          DGESVD_SVD<Eigen::MatrixXd> svd(
              A, Eigen::ComputeFullU | Eigen::ComputeFullV);
          return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
        });

    reg.register_baseline(
        "dbdsvdx", ConcurrencyModel::LapackInternal,
        "LAPACK dgesvdx/dbdsvdx (bisection & inverse iteration)",
        [](const Eigen::MatrixXd& A) -> SVDOutput {
          DBDSVDX_SVD<Eigen::MatrixXd> svd(
              A, Eigen::ComputeFullU | Eigen::ComputeFullV);
          return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
        });

    reg.register_baseline(
        "eigen_jacobi", ConcurrencyModel::Serial,
        "Eigen::JacobiSVD (two-sided cyclic Jacobi directly on A)",
        [](const Eigen::MatrixXd& A) -> SVDOutput {
          Eigen::JacobiSVD<Eigen::MatrixXd> svd(
              A, Eigen::ComputeFullU | Eigen::ComputeFullV);
          return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
        });

    return reg;
  }
};

}  // namespace SVD_Project::Benchmark

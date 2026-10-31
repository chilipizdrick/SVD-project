#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <lapacke.h>
#include <ostream>

namespace SVD_Project {

/**
 * @brief Diagonalization strategy for the symmetric polar factor H (or matrix A)
 * in Halley SVD.
 */
enum class HalleyDiagonalizer {
  TwoSidedJacobi = 0,    // Classical two-sided cyclic Jacobi on H
  CholeskyOneSided = 1,  // Complete pivoted Cholesky + Drmac-Veselic one-sided Jacobi
  QDWH_EIG = 2,          // Spectral Divide & Conquer via QDWH sign function
  TridiagDC = 3          // Tridiagonalization + Gu-Eisenstat D&C (LAPACK dsyevd)
};

template <typename _MatrixType,
          HalleyDiagonalizer _DiagMethod = HalleyDiagonalizer::TwoSidedJacobi>
class Halley_SVD;

}  // namespace SVD_Project

namespace Eigen {
namespace internal {
template <typename _MatrixType, SVD_Project::HalleyDiagonalizer _DiagMethod>
struct traits<SVD_Project::Halley_SVD<_MatrixType, _DiagMethod>>
    : public traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};
}  // namespace internal
}  // namespace Eigen

namespace SVD_Project {

/**
 * @brief High-precision SVD based on Halley's method (QDWH) for polar decomposition
 * followed by configurable diagonalization algorithms.
 *
 * @tparam _MatrixType Matrix type, e.g. Eigen::MatrixXd.
 * @tparam _DiagMethod Diagonalization algorithm policy.
 */
template <typename _MatrixType, HalleyDiagonalizer _DiagMethod>
class Halley_SVD : public Eigen::SVDBase<Halley_SVD<_MatrixType, _DiagMethod>> {
 public:
  using Base = Eigen::SVDBase<Halley_SVD<_MatrixType, _DiagMethod>>;
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;

  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

  Halley_SVD();

  Halley_SVD(const MatrixType& A,
             unsigned int computationOptions = Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);

  Halley_SVD(const MatrixType& A, std::ostream* os,
             unsigned int computationOptions = Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);

  Halley_SVD& compute(const MatrixType& A,
                      unsigned int computationOptions = Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);

  /// Access computed unitary polar factor Q
  const MatrixDynamic& matrixQ() const { return m_matrixQ; }

  /// Access computed symmetric positive semi-definite polar factor H
  const MatrixDynamic& matrixH() const { return m_matrixH; }

  /// Set relative threshold for numerical null-space truncation
  Halley_SVD& setThreshold(RealScalar threshold) {
    m_threshold = threshold;
    return *this;
  }

  /// Get current relative rank threshold
  RealScalar threshold() const { return m_threshold; }

  /// Set runtime diagonalization policy
  Halley_SVD& setDiagonalizer(HalleyDiagonalizer method) {
    m_diagonalizer = method;
    return *this;
  }

  /// Get current diagonalization policy
  HalleyDiagonalizer diagonalizer() const { return m_diagonalizer; }

 private:
  MatrixDynamic m_matrixQ;
  MatrixDynamic m_matrixH;
  std::ostream* m_divOstream = nullptr;
  RealScalar m_threshold = RealScalar(0);
  HalleyDiagonalizer m_diagonalizer = _DiagMethod;

  void compute_qdwh_polar(const MatrixDynamic& A, MatrixDynamic& Q,
                          MatrixDynamic& H);
  RealScalar estimate_inv_r1_norm(const MatrixDynamic& R);

  // Diagonalization implementations
  void compute_two_sided_jacobi(const MatrixDynamic& H,
                                VectorDynamic& singularValues,
                                MatrixDynamic& V);

  void compute_cholesky_onesided_evd(const MatrixDynamic& H,
                                     VectorDynamic& singularValues,
                                     MatrixDynamic& V);

  void compute_qdwh_eig_evd(const MatrixDynamic& H,
                            VectorDynamic& singularValues,
                            MatrixDynamic& V);

  void compute_qdwh_eig_step(const MatrixDynamic& H_sub,
                             VectorDynamic& lambdas,
                             MatrixDynamic& V_sub);

  void compute_tridiag_dc_evd(const MatrixDynamic& H,
                              VectorDynamic& singularValues,
                              MatrixDynamic& V);
};

// Aliases for explicit parametrization
template <typename _MatrixType>
using Halley_SVD_Jacobi = Halley_SVD<_MatrixType, HalleyDiagonalizer::TwoSidedJacobi>;

template <typename _MatrixType>
using Halley_SVD_Cholesky = Halley_SVD<_MatrixType, HalleyDiagonalizer::CholeskyOneSided>;

template <typename _MatrixType>
using Halley_SVD_QDWH_EIG = Halley_SVD<_MatrixType, HalleyDiagonalizer::QDWH_EIG>;

template <typename _MatrixType>
using Halley_SVD_TridiagDC = Halley_SVD<_MatrixType, HalleyDiagonalizer::TridiagDC>;

}  // namespace SVD_Project

#include "halley_svd.cpp"

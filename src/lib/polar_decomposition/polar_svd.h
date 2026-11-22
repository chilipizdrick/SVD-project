#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <ostream>

#include "diagonalizers.h"
#include "polar_algorithms.h"
#include "polar_enums.h"

namespace SVD_Project {

template <typename _MatrixType,
          PolarAlgorithm _PolarMethod = PolarAlgorithm::Halley,
          DiagonalizationAlgorithm _DiagMethod =
              DiagonalizationAlgorithm::TwoSidedJacobi>
class Polar_SVD;

}  // namespace SVD_Project

namespace Eigen {
namespace internal {
template <typename _MatrixType, SVD_Project::PolarAlgorithm _PolarMethod,
          SVD_Project::DiagonalizationAlgorithm _DiagMethod>
struct traits<
    SVD_Project::Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>>
    : public traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};
}  // namespace internal
}  // namespace Eigen

namespace SVD_Project {

/**
 * @brief Generalized Polar-Decomposition SVD solver.
 *
 * Computes A = U * Sigma * V^T by first computing a polar decomposition
 * A = Q * H via the chosen PolarAlgorithm and then diagonalizing the
 * symmetric positive semi-definite factor H = V * Sigma * V^T via the
 * chosen DiagonalizationAlgorithm (with U = Q * V).
 *
 * @tparam _MatrixType Matrix type, e.g. Eigen::MatrixXd.
 * @tparam _PolarMethod Polar decomposition algorithm policy.
 * @tparam _DiagMethod Symmetric diagonalization algorithm policy.
 */
template <typename _MatrixType, PolarAlgorithm _PolarMethod,
          DiagonalizationAlgorithm _DiagMethod>
class Polar_SVD
    : public Eigen::SVDBase<Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>> {
 public:
  using Base =
      Eigen::SVDBase<Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>>;
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;

  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

  Polar_SVD();

  Polar_SVD(const MatrixType& A,
            unsigned int computationOptions = Eigen::ComputeThinU |
                                              Eigen::ComputeThinV);

  Polar_SVD(const MatrixType& A, std::ostream* os,
            unsigned int computationOptions = Eigen::ComputeThinU |
                                              Eigen::ComputeThinV);

  Polar_SVD& compute(const MatrixType& A,
                     unsigned int computationOptions = Eigen::ComputeThinU |
                                                       Eigen::ComputeThinV);

  /// Access computed unitary polar factor Q
  const MatrixDynamic& matrixQ() const { return m_matrixQ; }

  /// Access computed symmetric positive semi-definite polar factor H
  const MatrixDynamic& matrixH() const { return m_matrixH; }

  /// Set relative threshold for numerical null-space truncation
  Polar_SVD& setThreshold(RealScalar threshold) {
    m_threshold = threshold;
    return *this;
  }

  /// Get current relative rank threshold
  RealScalar threshold() const { return m_threshold; }

  /// Set runtime polar decomposition policy
  Polar_SVD& setPolarAlgorithm(PolarAlgorithm method) {
    m_polarAlgorithm = method;
    return *this;
  }

  /// Get current polar decomposition policy
  PolarAlgorithm polarAlgorithm() const { return m_polarAlgorithm; }

  /// Set runtime diagonalization policy
  Polar_SVD& setDiagonalizer(DiagonalizationAlgorithm method) {
    m_diagonalizer = method;
    return *this;
  }

  /// Get current diagonalization policy
  DiagonalizationAlgorithm diagonalizer() const { return m_diagonalizer; }

 private:
  MatrixDynamic m_matrixQ;
  MatrixDynamic m_matrixH;
  std::ostream* m_divOstream = nullptr;
  RealScalar m_threshold = RealScalar(0);
  PolarAlgorithm m_polarAlgorithm = _PolarMethod;
  DiagonalizationAlgorithm m_diagonalizer = _DiagMethod;
};

// Convenience Halley aliases preserving concise instantiation syntax
using HalleyDiagonalizer = DiagonalizationAlgorithm;

template <typename _MatrixType,
          DiagonalizationAlgorithm _DiagMethod =
              DiagonalizationAlgorithm::TwoSidedJacobi>
using Halley_SVD = Polar_SVD<_MatrixType, PolarAlgorithm::Halley, _DiagMethod>;

template <typename _MatrixType>
using Halley_SVD_Jacobi =
    Polar_SVD<_MatrixType, PolarAlgorithm::Halley,
              DiagonalizationAlgorithm::TwoSidedJacobi>;

template <typename _MatrixType>
using Halley_SVD_Cholesky =
    Polar_SVD<_MatrixType, PolarAlgorithm::Halley,
              DiagonalizationAlgorithm::CholeskyOneSided>;

template <typename _MatrixType>
using Halley_SVD_QDWH_EIG =
    Polar_SVD<_MatrixType, PolarAlgorithm::Halley,
              DiagonalizationAlgorithm::QDWH_EIG>;

template <typename _MatrixType>
using Halley_SVD_TridiagDC =
    Polar_SVD<_MatrixType, PolarAlgorithm::Halley,
              DiagonalizationAlgorithm::TridiagDC>;

// Convenience Newton polar SVD aliases
template <typename _MatrixType,
          DiagonalizationAlgorithm _DiagMethod =
              DiagonalizationAlgorithm::TwoSidedJacobi>
using Newton_Polar_SVD =
    Polar_SVD<_MatrixType, PolarAlgorithm::Newton, _DiagMethod>;

template <typename _MatrixType,
          DiagonalizationAlgorithm _DiagMethod =
              DiagonalizationAlgorithm::TwoSidedJacobi>
using Newton_Polar_SVD_Acceleration =
    Polar_SVD<_MatrixType, PolarAlgorithm::NewtonAccelerated, _DiagMethod>;

}  // namespace SVD_Project

#include "polar_svd.cpp"

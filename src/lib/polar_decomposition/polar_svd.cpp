#pragma once

#include "polar_svd.h"

namespace SVD_Project {

template <typename _MatrixType, PolarAlgorithm _PolarMethod,
          DiagonalizationAlgorithm _DiagMethod>
Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>::Polar_SVD() {}

template <typename _MatrixType, PolarAlgorithm _PolarMethod,
          DiagonalizationAlgorithm _DiagMethod>
Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>::Polar_SVD(
    const MatrixType& A, unsigned int computationOptions) {
  compute(A, computationOptions);
}

template <typename _MatrixType, PolarAlgorithm _PolarMethod,
          DiagonalizationAlgorithm _DiagMethod>
Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>::Polar_SVD(
    const MatrixType& A, std::ostream* os, unsigned int computationOptions)
    : m_divOstream(os) {
  compute(A, computationOptions);
}

template <typename _MatrixType, PolarAlgorithm _PolarMethod,
          DiagonalizationAlgorithm _DiagMethod>
Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>&
Polar_SVD<_MatrixType, _PolarMethod, _DiagMethod>::compute(
    const MatrixType& A, unsigned int computationOptions) {
  Index m = A.rows();
  Index n = A.cols();

  if (m < n) {
    Polar_SVD<MatrixType, _PolarMethod, _DiagMethod> svd_adj;
    svd_adj.setPolarAlgorithm(m_polarAlgorithm);
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

  PolarDecomposition<MatrixType>::compute(m_polarAlgorithm, A, m_matrixQ,
                                          m_matrixH, m_divOstream);

  VectorDynamic S;
  MatrixDynamic V;
  SymmetricDiagonalizer<MatrixType>::diagonalize(m_diagonalizer, m_matrixH, S,
                                                 V);

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

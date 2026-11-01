#ifndef NEWTON_POLAR_HPP
#define NEWTON_POLAR_HPP

#include <Eigen/LU>

#include "newton_polar.h"

namespace SVD_Project {

template <typename MatrixType>
MatrixType Newton_Polar<MatrixType>::compute(const MatrixType& A,
                                              Index max_iter,
                                              RealScalar tol) {
  using Matrix = MatrixType;
  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton polar decomposition requires square matrix");

  Matrix X = A;
  if (X.norm() == RealScalar(0)) return X;

  const Matrix I = Matrix::Identity(m, n);

  for (Index k = 0; k < max_iter; ++k) {
    Matrix Y = X.adjoint().fullPivLu().solve(I);
    if (!Y.allFinite()) return X;

    Matrix delta = RealScalar(0.5) * (Y - X);
    Matrix X_new = X + delta;

    RealScalar scale = X_new.norm();
    if (scale == RealScalar(0)) scale = RealScalar(1);

    X = X_new;

    if (delta.norm() <= tol * scale) break;
  }

  return X;
}



template <typename _MatrixType>
Newton_Polar_SVD<_MatrixType>::Newton_Polar_SVD(
    const _MatrixType& matrix, unsigned int computationOptions) {
  compute(matrix);
}

template <typename _MatrixType>
Newton_Polar_SVD<_MatrixType>& Newton_Polar_SVD<_MatrixType>::compute(
    const _MatrixType& A) {
  using MatrixType = _MatrixType;
  using Scalar     = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const Eigen::Index m = A.rows();
  const Eigen::Index n = A.cols();
  eigen_assert(m == n && "Newton_Polar_SVD requires square matrix");

  Newton_Polar<MatrixType> np;
  MatrixType Up = np.compute(A);          

  MatrixType H = Up.adjoint() * A;        
  H = Scalar(0.5) * (H + H.adjoint());    

  Eigen::SelfAdjointEigenSolver<MatrixType> es(H);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("Newton_Polar_SVD: eigensolver failed");

  auto evals = es.eigenvalues();          
  auto V     = es.eigenvectors();

  std::vector<int> idx(evals.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int i, int j) { return evals[i] > evals[j]; });

  MatrixType Vsorted(V.rows(), V.cols());
  Eigen::VectorXd S(evals.size());
  for (int i = 0; i < (int)evals.size(); ++i) {
    Vsorted.col(i) = V.col(idx[i]);
    S(i) = std::max(RealScalar(0), evals[idx[i]]);
  }

  m_matrixU        = Up * Vsorted;
  m_matrixV        = Vsorted;
  m_singularValues = S;

  m_isInitialized = true;
  m_computeFullU  = true;
  m_computeFullV  = true;

  return *this;
}

}  // namespace SVD_Project

#endif  // NEWTON_POLAR_HPP
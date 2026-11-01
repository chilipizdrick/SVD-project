#ifndef NEWTON_POLAR_H
#define NEWTON_POLAR_H

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SVD_Project {

template <typename MatrixType>
class Newton_Polar {
  public:
    using Scalar     = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index      = Eigen::Index;

    Newton_Polar() = default;

    MatrixType compute(
        const MatrixType& A,
        Index max_iter = 100,
        RealScalar tol = std::numeric_limits<RealScalar>::epsilon()
    );
};

template <typename _MatrixType>
class Newton_Polar_SVD : public Eigen::SVDBase<Newton_Polar_SVD<_MatrixType>> {
  public:
    using MatrixType = _MatrixType;
    using Scalar     = typename MatrixType::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    Newton_Polar_SVD(const _MatrixType& matrix, unsigned int computationOptions);

    Newton_Polar_SVD<_MatrixType>& compute(const _MatrixType& matrix);

    ~Newton_Polar_SVD() = default;

    const Eigen::MatrixXd& matrixU()        const { return m_matrixU; }
    const Eigen::MatrixXd& matrixV()        const { return m_matrixV; }
    const Eigen::VectorXd& singularValues() const { return m_singularValues; }

  private:
    Eigen::MatrixXd m_matrixU;
    Eigen::MatrixXd m_matrixV;
    Eigen::VectorXd m_singularValues;
    bool m_isInitialized = false;
    bool m_computeFullU  = false;
    bool m_computeFullV  = false;
};

} 

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::Newton_Polar_SVD<_MatrixType>>
    : public Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "newton_polar.hpp"

#endif  // NEWTON_POLAR_H
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SVD_Project {

template <typename _MatrixType>
class Newton_Polar {
 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar() {}

  MatrixType compute(
      const MatrixType& A,
      Index max_iter = 100,
      RealScalar tol = std::numeric_limits<RealScalar>::epsilon());

 private:
  static RealScalar norm1(const MatrixType& M) {
    return M.cwiseAbs().colwise().sum().maxCoeff();
  }
};

}  // namespace SVD_Project

#include "newton.hpp"

#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SVD_Project {
template <typename _MatrixType>
class Newton_Polar_Acceleration {
 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar_Acceleration() = default;

  void       setScalingThreshold(RealScalar tau) { m_tau = tau; }
  RealScalar scalingThreshold() const            { return m_tau; }

  Index steps()       const { return m_steps; }
  Index scaledSteps() const { return m_scaled_steps; }

  MatrixType compute(
      const MatrixType& A,
      Index max_iter = 100,
      RealScalar tol = std::numeric_limits<RealScalar>::epsilon());

 private:
  static RealScalar norm1  (const MatrixType& M);
  static RealScalar normInf(const MatrixType& M);

  RealScalar m_tau          = RealScalar(1e-2);
  Index      m_steps        = 0;
  Index      m_scaled_steps = 0;
};

}  // namespace SVD_Project

#include "newton_acceleration.hpp"

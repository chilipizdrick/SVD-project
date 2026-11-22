#pragma once

namespace SVD_Project {

/**
 * @brief Polar decomposition algorithm used to compute A = Q * H.
 */
enum class PolarAlgorithm {
  Halley = 0  // Dynamically weighted Halley iteration (QDWH)
};

/**
 * @brief Diagonalization strategy for the symmetric polar factor H.
 */
enum class DiagonalizationAlgorithm {
  TwoSidedJacobi = 0,    // Classical two-sided cyclic Jacobi on H
  CholeskyOneSided = 1,  // Complete pivoted Cholesky + Drmac-Veselic one-sided Jacobi
  QDWH_EIG = 2,          // Spectral Divide & Conquer via QDWH sign function
  TridiagDC = 3          // Tridiagonalization + Gu-Eisenstat D&C (LAPACK dsyevd)
};

}  // namespace SVD_Project


#include <Eigen/Core>
#include <vector>


#include "dualres/eigen_slicing.h"


#ifndef _DUALRES_PERTURBATION_MATRIX_
#define _DUALRES_PERTURBATION_MATRIX_


namespace dualres {
  /*! @addtogroup SmoothnessEstimation
   * @{
   */



  Eigen::MatrixXi perturbations_3d_unit_directions() {
    const int nrow = 14, ncol = 3;
    Eigen::MatrixXi P(nrow, ncol);
    P << 0,  0,  0,
      1,  0,  0,
      0,  1,  0,
      1,  1,  0,
     -1,  1,  0,
      0,  0,  1,
      1,  0,  1,
      0,  1,  1,
      1,  1,  1,
     -1,  1,  1,
      1,  0, -1,
      0,  1, -1,
      1,  1, -1,
     -1,  1, -1;
    return P;
  };



  /*!
   * Return all 3-element combinations of voxel offsets up to n
   */
  Eigen::MatrixXi perturbation_permutations_3d(const int n) {
    const int M = n * n * n;
    const int D = 3;
    Eigen::MatrixXi grid(M, D);
    Eigen::MatrixXi col_1 = Eigen::RowVectorXi::LinSpaced(n, 1, n).replicate(n, n);
    Eigen::MatrixXi col_2 = Eigen::RowVectorXi::LinSpaced(n, 1, n).replicate(n * n, 1);
    grid.col(0) = Eigen::VectorXi::LinSpaced(n, 1, n).replicate(n * n, 1);
    grid.col(1) = Eigen::Map<Eigen::VectorXi>(col_1.data(), M, 1);
    grid.col(2) = Eigen::Map<Eigen::VectorXi>(col_2.data(), M, 1);
    return grid;
  };

  

  /*!
   * Generate 3D ijk perturbation matrix to assist minimum contrast estimation.
   */
  Eigen::MatrixXi perturbation_matrix_3d() {
    const int nvox_angles = 18;
    const int nvox_raster = 25;
    Eigen::MatrixXi U = dualres::perturbations_3d_unit_directions();
    Eigen::MatrixXi A = dualres::perturbation_permutations_3d(nvox_angles);
    const int Np = U.rows() * A.rows() + 3 * (nvox_raster - nvox_angles);
    Eigen::MatrixXi P = Eigen::MatrixXi::Zero(Np, 3);
    std::vector<bool> duplicate(Np, false);
    int unique_count = 0;
    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < U.rows(); j++) {
	P.row(i * U.rows() + j) = (U.row(j).array() * A.row(i).array())
	  .matrix().transpose();
      }
    }
    for (int i = 0; i < (nvox_raster - nvox_angles); i++) {
      for (int j = 0; j < 3; j++) {
	P(A.rows() * U.rows() + i * 3 + j, j) = nvox_angles + i + 1;
      }
    }
    // Remove duplicate rows
    for (int i = 0; i < (Np - 1); i++) {
      if (!duplicate[i]) {
	for (int j = i + 1; j < Np; j++) {
	  if (!duplicate[j]) {
	    duplicate[j] = (P(i, 0) == P(j, 0)) &&
	      (P(i, 1) == P(j, 1)) && (P(i, 2) == P(j, 2));
	  }
	}
	unique_count++;
      }
    }
    if (!duplicate.back())  unique_count++;
    Eigen::VectorXi unique_index(unique_count);
    Eigen::Vector3i col_index;
    col_index << 0, 1, 2;
    int j = 0;
    for (int i = 0; i < duplicate.size(); i++) {
      if (!duplicate[i]) {
	unique_index[j] = i;
	j++;
      }
    }
    return dualres::nullary_index(P, unique_index, col_index);
  };

  /*! @} */
}



#endif  // _DUALRES_PERTURBATION_MATRIX_

 

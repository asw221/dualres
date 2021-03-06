
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <stdexcept>
#include <vector>

#include "dualres/defines.h"
#include "dualres/eigen_slicing.h"
#include "dualres/kernels.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/utilities.h"


#ifndef _DUALRES_KRIGING_MATRIX_
#define _DUALRES_KRIGING_MATRIX_


namespace dualres {
  /*! @addtogroup GaussianProcessModels
   * @{
   */
  

  template< typename T = float >
  struct kriging_matrix_data {
    std::vector<T> _Data;
    std::vector<int> column_indices;
    std::vector<int> cum_row_counts;
    int ncol;
    int nrow;
  };





  Eigen::MatrixXi expand_grid_3d(const Eigen::Vector3i &pos_griddims) {
    const Eigen::Vector3i _NN = (2 * pos_griddims.array() + 1).matrix();
    const int D = 3, _nn_prod = _NN.prod();
    if (_nn_prod <= 0)
      throw std::logic_error("expand_grid_3d: pos_griddims must all be >= 0");
    Eigen::MatrixXi grid(_nn_prod, D);
    Eigen::MatrixXi col1 = Eigen::RowVectorXi::LinSpaced(
      _NN[1], -pos_griddims[1], pos_griddims[1]).replicate(_NN[0], _NN[2]);
    Eigen::MatrixXi col2 = Eigen::RowVectorXi::LinSpaced(
      _NN[2], -pos_griddims[2], pos_griddims[2]).replicate(_NN[0] * _NN[1], 1);
    grid.col(0) = Eigen::VectorXi::LinSpaced(_NN[0], -pos_griddims[0], pos_griddims[0])
      .replicate(_NN[1] * _NN[2], 1);
    grid.col(1) = Eigen::Map<Eigen::VectorXi>(col1.data(), _nn_prod, 1);
    grid.col(2) = Eigen::Map<Eigen::VectorXi>(col2.data(), _nn_prod, 1);
    return grid;
  };




  

  Eigen::MatrixXi neighborhood_perturbation(
    const dualres::qform_type &Qform,  // 
    const float &radius                // in mm units
  ) {
    if (radius < 0)
      throw std::domain_error("neighborhood_perturbation: radius cannot be negative");
    const Eigen::Vector3f voxel_dims = dualres::voxel_dimensions(Qform);
    const Eigen::Vector3i nbr_range =
      (radius / voxel_dims.array()).cast<int>().matrix();
    Eigen::MatrixXi P = dualres::expand_grid_3d(nbr_range);
    Eigen::MatrixXf Q = P.cast<float>();
    Q.conservativeResize(P.rows(), P.cols() + 1);
    Q.col(Q.cols() - 1) = Eigen::VectorXf::Zero(Q.rows(), 1);
    Q = (Q * Qform.transpose()).eval();
    Q.conservativeResize(Q.rows(), Q.cols() - 1);
    // std::vector<int> ball_indices(P.rows());
    std::vector<int> ball_indices;
    std::vector<int> all_cols{0, 1, 2};
    int within_radius_count = 0;
    ball_indices.reserve(P.rows());
    for (int i = 0; i < P.rows(); i++) {
      if (Q.row(i).norm() <= radius) {
	ball_indices.push_back(i);
	within_radius_count++;
      }
    }
    ball_indices.shrink_to_fit();
    if (1.0 - (double)within_radius_count / P.rows() > 0.55) {
      // Sphere within a cube: volume of sphere is about 52.4%
      //   so > 0.55 is a very rough check
      std::cout << "Warning: neighborhood_perturbation: "
		<< (1.0 - (double)within_radius_count / P.rows()) * 100
		<< "% of perturbations outside radius "
		<< radius << std::endl;
      if (within_radius_count == 0)
	throw std::logic_error(
          "neighborhood_perturbation: no perturbations within radius");
    }
    Eigen::MatrixXi Psub(within_radius_count, 3);
    for (int i = 0; i < within_radius_count; i++) {
      Psub.row(i) = P.row(ball_indices[i]);
    }
    return Psub;
  };

  

  
  template< typename T = float >
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> perturbation_kernel_distances(
    const Eigen::MatrixXi &P,
    const dualres::qform_type &Qform,
    const std::vector<T> &kernel_params
  ) {
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> DistMatType;
    DistMatType Dist = DistMatType::Ones(P.rows(), P.rows());
    Eigen::MatrixXf P_ext = P.cast<float>();
    P_ext.conservativeResize(P_ext.rows(), P_ext.cols() + 1);
    P_ext.col(P_ext.cols() - 1) = Eigen::VectorXf::Zero(P_ext.rows());
    P_ext = (P_ext * Qform.transpose()).eval();
    P_ext.conservativeResize(P_ext.rows(), P_ext.cols() - 1);
    T value;
    if (P.rows() > 1) {
      // Loop over lower triangle
      for (int i = 1; i < P.rows(); i++) {
	for (int j = 0; j < i; j++) {
	  value = dualres::kernels::rbf(
            (T)(P_ext.row(i) - P_ext.row(j)).norm(),
	    kernel_params[1], kernel_params[2], kernel_params[0]);
	  Dist(i, j) = value;
	  Dist(j, i) = value;
	}
      }
    }
    return Dist;
  };


  


  
  

  template< typename ScalarType = float, typename ImageType = float >
  dualres::kriging_matrix_data<ScalarType> get_sparse_kriging_matrix_data(
    const nifti_image* const high_res,
    const nifti_image* const std_res,
    const std::vector<ScalarType> &rbf_params,
    const ScalarType &neighborhood_radius
  ) {
    // Compute kriging matrix data formatted to translate to sparse,
    // compressed, row-oriented matrix types. Resultant column indices correspond
    // to High Res data in bounded brain space (removing 0/nan padding around
    // the brain)
    //

    const float eps0 = 1e-5;
    const ImageType* const data_ptr = (ImageType*)high_res->data;    
    const int nvox_hr = (int)high_res->nvox;
    const dualres::qform_type Qform_hr = dualres::qform_matrix(high_res);
    ImageType voxel_value_hr;

    std::cout << "Computing neighborhoods" << std::endl;
    const float nhood_radius = (float)neighborhood_radius;
    const Eigen::MatrixXi P = dualres::neighborhood_perturbation(
      Qform_hr, nhood_radius);
    // std::cout << P << std::endl;
    
    if (P.rows() > 5e3)
      throw std::logic_error(
      "get_sparse_kriging_array_data: neighborhood >5e3 voxels");
    
    
    std::cout << "Computing kernel distance matrix" << std::endl;
    std::vector<float> covar_params(rbf_params.size());
    for (int i = 0; i < (int)rbf_params.size(); i++)
      covar_params[i] = (float)rbf_params[i];
    const Eigen::MatrixXf K =
      dualres::perturbation_kernel_distances<>(P, Qform_hr, covar_params);
    // std::cout << "\n" << K << "\n" << std::endl;


    const dualres::nifti_bounding_box bb_hr =
      dualres::get_bounding_box(high_res);
    const Eigen::Vector3i bhr_dims = bb_hr.ijk_max - bb_hr.ijk_min +
      Eigen::Vector3i::Ones();
    std::vector<int> bounded_nonzero_index_hr =
      dualres::get_bounding_box_nonzero_flat_index(high_res);
    
    dualres::qform_type Qform_std = dualres::qform_matrix(std_res);
    dualres::qform_type map_ijk_std_to_ijk_hr = Qform_hr.inverse() * Qform_std;
    
    Eigen::MatrixXi ijk_std = dualres::get_nonzero_indices(std_res);
    // ijk_std.conservativeResize(ijk_std.rows(), ijk_std.cols() + 1);
    // ijk_std.col(ijk_std.cols() - 1) = Eigen::VectorXi::Ones(ijk_std.rows());
    // ijk_std = (((ijk_std.cast<float>() * Qform_std.transpose()).rowwise() +
    // 		(-Qform_std.col(3) + Qform_hr.col(3)).transpose()) *
    // 	       Qform_hr.inverse().transpose()).cast<int>().eval();
    // ijk_std.conservativeResize(ijk_std.rows(), ijk_std.cols() - 1);

    // Eigen::Vector4f current_ijk_std = Eigen::Vector4f::Zero();  // <- Zeros (!)
    // Eigen::Vector4i current_ijk_hr = Eigen::Vector4i::Zero();
    Eigen::Vector4i current_ijk_hr_base = Eigen::Vector4i::Zero();
    Eigen::Vector4f current_ijk_std = Eigen::Vector4f::Ones();  // <- Zeros (!)
    Eigen::Vector4i current_ijk_hr = Eigen::Vector4i::Ones();
    // Eigen::Vector4i current_ijk_hr_base = Eigen::Vector4i::Ones();

    // Qform_std.col(3) = Qform_hr.col(3);
    
    Eigen::VectorXf k_prime;    // 
    Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> _w_;
      // one row of kriging matrix weights
    
    dualres::kriging_matrix_data<ScalarType> kmd;
    kmd.cum_row_counts.resize(ijk_std.rows() + 1);
    kmd.cum_row_counts[0] = 0;
    // kmd.ncol = bhr_dims.prod();
    kmd.ncol = bb_hr.nnz;
    kmd.nrow = ijk_std.rows();

    // Create variables to help construct output:
    //  - bounded_hr_index and kernel_distances will be concatenated to the
    //    ends of kmd.column_indices, and kmd._Data, respectively
    //  - perturbation_index is used to help subset K
    std::vector<int> bounded_hr_index, perturbation_index;
    std::vector<float> kernel_distances;
    
    
    float dist;
    ScalarType w_sum;
    int empty_row_count = 0, row_count = 0, voxel_offset_hr, bhr_i;


    
    dualres::utilities::progress_bar pb(ijk_std.rows());
    std::cout << "Computing kriging matrix rows" << std::endl;
    
    // Loop over Std Res voxels
    for (int i = 0; i < ijk_std.rows(); i++) {
      bounded_hr_index.clear();
      perturbation_index.clear();
      kernel_distances.clear();
      bounded_hr_index.reserve(P.rows());
      perturbation_index.reserve(P.rows());
      kernel_distances.reserve(P.rows());
      row_count = 0;


      // current_ijk_std.head<3>() = ijk_std.row(i).cast<float>();
      for (int k = 0; k < 3; k++)
	current_ijk_std[k] = (float)ijk_std(i, k);
      current_ijk_hr_base = (map_ijk_std_to_ijk_hr * current_ijk_std).cast<int>();
      //
      // Loop over perturbations
      for (int j = 0; j < P.rows(); j++) {
	// current_ijk_hr.head<3>() = current_ijk_hr_base.head<3>() + P.row(j);
	for (int k = 0; k < 3; k++)
	  current_ijk_hr[k] = current_ijk_hr_base[k] + P(j, k);

	voxel_offset_hr = current_ijk_hr[2] * high_res->ny * high_res->nx +
	  current_ijk_hr[1] * high_res->nx + current_ijk_hr[0];

	if (voxel_offset_hr >= 0 && voxel_offset_hr < nvox_hr) {
	  
	  voxel_value_hr = *(data_ptr + voxel_offset_hr);
	  if (!(isnan(voxel_value_hr) || voxel_value_hr == 0)) {
	    // High Res voxel is non-0/nan and within Neighborhood
	    // Store: (i) flat index of High Res voxel in bounded brain space
	    //        (ii) index (in rows of P) of High Res voxel and
	    //        (iii) distance from High Res voxel to Std Res voxel
	    //
	    bhr_i = 
              (current_ijk_hr[2] - bb_hr.ijk_min[2]) * bhr_dims[0] * bhr_dims[1]  +
	      (current_ijk_hr[1] - bb_hr.ijk_min[1]) * bhr_dims[0] +
	      (current_ijk_hr[0] - bb_hr.ijk_min[0]);
	    // bhr_i =
	    //   (current_ijk_hr[2] - bb_hr.ijk_min[2]) +
	    //   (current_ijk_hr[1] - bb_hr.ijk_min[1]) * bhr_dims[2] +
	    //   (current_ijk_hr[0] - bb_hr.ijk_min[0]) * bhr_dims[2] * bhr_dims[1];
	    // bounded_hr_index.push_back(bhr_i);  // (i)
	    if (bounded_nonzero_index_hr[bhr_i] >= 0) {
	      // (i)
	      bounded_hr_index.push_back(bounded_nonzero_index_hr[bhr_i]);
	      // (ii)
	      perturbation_index.push_back(j);
	      dist = (Qform_std * current_ijk_std -
		      Qform_hr * current_ijk_hr.cast<float>()).norm() +
		eps0;
	      // (iii)
	      kernel_distances.push_back(dualres::kernels::rbf(dist,
                covar_params[1], covar_params[2], covar_params[0]));
	      row_count++;
	    }
	  }
	}
      }  // for (int j = 0; j < P.rows() ...
      bounded_hr_index.shrink_to_fit();
      perturbation_index.shrink_to_fit();
      kernel_distances.shrink_to_fit();
      
      kmd.cum_row_counts[i + 1] = kmd.cum_row_counts[i];
      if (row_count == 0) {
	empty_row_count++;
      }
      else {
	k_prime = Eigen::Map<Eigen::VectorXf>(
          kernel_distances.data(), kernel_distances.size());
	_w_ = dualres::eigen_select_symmetric(K, perturbation_index)
	  .colPivHouseholderQr().solve(k_prime)
	  .template cast<ScalarType>();
	// _w_ = dualres::nullary_index(K,
        //   Eigen::Map<Eigen::VectorXi>(perturbation_index.data(), perturbation_index.size()),
        //   Eigen::Map<Eigen::VectorXi>(perturbation_index.data(), perturbation_index.size()))
	//   .colPivHouseholderQr().solve(k_prime);
	w_sum = _w_.sum();
	if (w_sum == 0)  w_sum = 1;
	if (!isnan(w_sum)) {
	  _w_ /= w_sum;
	  kmd._Data.insert(kmd._Data.end(), std::make_move_iterator(_w_.data()),
			   std::make_move_iterator(_w_.data() + _w_.size()));
	  kmd.column_indices.insert(kmd.column_indices.end(),
            bounded_hr_index.begin(), bounded_hr_index.end());
	  kmd.cum_row_counts[i + 1] += _w_.size();
	}
      }
      
      pb++;
      std::cout << pb;
    }  // for (int i = 0; i < ijk_std.rows() ...
    pb.finish();

    if ((double)empty_row_count / ijk_std.rows() > 0.05) {
      std::cerr << "Warning: " << std::setprecision(2) << std::fixed
		<< ((double)empty_row_count / ijk_std.rows() * 100)
		<< "% of kriging matrix rows are all zero"
		<< std::endl;
    }

    return kmd;
  };



  


  
  /*! @} */  
}  // namespace dualres 


#endif  // _DUALRES_KRIGING_MATRIX_





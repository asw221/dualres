
#include <arrayfire.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <vector>

#include "defines.h"
#include "kernels.h"
#include "nifti_manipulation.h"
#include "utilities.h"


#ifndef _DUALRES_KRIGGING_MATRIX_
#define _DUALRES_KRIGGING_MATRIX_


namespace dualres {  
  

  template< typename T = float >
  struct krigging_matrix_data {
    std::vector<T> _Data;
    std::vector<int> column_indices;
    std::vector<int> cum_row_counts;
    int ncol;
    int nrow;
  };





  template< typename T >
  af::array construct_sparse_krigging_array(
    const dualres::krigging_matrix_data<T> &kmd
  ) {
    return af::sparse(kmd.nrow, kmd.ncol, kmd._Data.size(),
		      kmd._Data.data(), kmd.cum_row_counts.data(),
		      kmd.column_indices.data());
  };
  

  

  template< typename T >
  void re_grid_krigging_matrix_data(
    dualres::krigging_matrix_data<T>& kmd,
    const af::dim4& old_grid_dim,
    const af::dim4& new_grid_dim
  ) {
    // re-compute number columns
    int new_ncol = 1;
    std::vector<int> ijk(3);
    for (int i = 0; i < 3; i++) {  // assume 3D grid
      new_ncol *= new_grid_dim[i];
      if (old_grid_dim[i] > new_grid_dim[i])
	throw std::logic_error(
          "re_grid_krigging_matrix_data: old grid should be <= new grid");
    }
    kmd.ncol = new_ncol;
    // re-compute column indices
    // (same ijk indices, different grid dimensions)
    for (std::vector<int>::iterator it = kmd.column_indices.begin();
	 it != kmd.column_indices.end(); ++it) {
      ijk[0] = (*it) % old_grid_dim[0];
      ijk[1] = ((*it) / old_grid_dim[0]) % old_grid_dim[1];
      ijk[2] = ((*it) / (old_grid_dim[0] * old_grid_dim[1])) % old_grid_dim[2];
      *it = ijk[2] * (new_grid_dim[1] * new_grid_dim[0]) +
	ijk[1] * new_grid_dim[0] + ijk[0];
    }
  };
  

  

  


  af::array expand_grid(const af::array &pos_griddims) {
    const int D = pos_griddims.dims(0);
    if (D > 4) {
      throw std::logic_error("expand_grid: pos_griddims must be <= 4 in size");
    }
    std::vector<int> N(4), _NN(4);
    int _nn_prod = 1;
    for (int j = 0; j < 4; j++) {
      if (j < D) {
	N[j] = pos_griddims(j).scalar<int>();
	_NN[j] = 2 * N[j] + 1;
	_nn_prod *= _NN[j];
      }
      else {
	N[j] = 1;
	_NN[j] = 1;
      }
    }
    const af::dim4 _d0(_NN[0], _NN[1], _NN[2], _NN[3]);
    const af::dim4 _d1(_nn_prod);
    af::array grid(_nn_prod, D);
    for (int j = 0; j < D; j++)
      grid.col(j) = af::moddims(af::range(_d0, j) - N[j], _d1);
    return grid;
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
	throw std::logic_error("neighborhood_perturbation: no perturbations within radius");
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




  // Eigen::MatrixXi reorient_ijk(const Eigen::MatrixXi &ijk, const dualres::qform_type &Qform) {
  //   Eigen::MatrixXf ijk_float = ijk.cast<float>();
  //   Eigen::MatrixXi ijk_new;
  //   ijk_float.conservativeResize(ijk.rows(), ijk.cols() + 1);
  //   ijk_float.col(ijk_float.cols() - 1) = Eigen::VectorXf::Ones(ijk.rows());
  //   ijk_new = (ijk_float * Qform.transpose()).cast<int>();
  //   ijk_new.conservativeResize(ijk.rows(), ijk.cols());
  //   return ijk_new;
  // };

  


  
  

  template< typename Scalar = float >
  dualres::krigging_matrix_data<float> get_sparse_krigging_array_data(
    const nifti_image* const high_res,
    const nifti_image* const std_res,
    const std::vector<float> &rbf_params,
    const float &neighborhood_radius
  ) {
    // Compute krigging matrix data formatted to translate to sparse,
    // compressed, row-oriented matrix types. Resultant column indices correspond
    // to High Res data in bounded brain space (removing 0/nan padding around
    // the brain)
    //

    
    const Scalar* const data_ptr = (Scalar*)high_res->data;    
    const int nvox_hr = (int)high_res->nvox;
    const dualres::qform_type Qform_hr = dualres::qform_matrix(high_res);

    std::cout << "Computing neighborhoods" << std::endl;
    const Eigen::MatrixXi P = neighborhood_perturbation(Qform_hr, neighborhood_radius);
    // std::cout << P << std::endl;
    
    if (P.rows() > 5e3)
      throw std::logic_error("get_sparse_krigging_array_data: neighborhood >5e3 voxels");
    
    
    std::cout << "Computing kernel distance matrix" << std::endl;
    const Eigen::MatrixXf K =
      dualres::perturbation_kernel_distances<>(P, Qform_hr, rbf_params);
    // std::cout << "\n" << K << "\n" << std::endl;


    const dualres::nifti_bounding_box bb_hr = dualres::get_bounding_box(high_res);
    const Eigen::Vector3i bhr_dims = bb_hr.ijk_max - bb_hr.ijk_min +
      Eigen::Vector3i::Ones();
    
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
    Eigen::VectorXf _w_;        //
    
    dualres::krigging_matrix_data<float> kmd;
    kmd.cum_row_counts.resize(ijk_std.rows() + 1);
    kmd.cum_row_counts[0] = 0;
    kmd.ncol = bhr_dims.prod();
    kmd.nrow = ijk_std.rows();

    // Create variables to help construct output:
    //  - bounded_hr_index and kernel_distances will be concatenated to the
    //    ends of kmd.column_indices, and kmd._Data, respectively
    //  - perturbation_index is used to help subset K
    std::vector<int> bounded_hr_index, perturbation_index;
    std::vector<float> kernel_distances;
    
    
    float dist, w_sum;
    int empty_row_count = 0, row_count = 0, voxel_offset_hr;

    Scalar voxel_value_hr;

    
    dualres::utilities::progress_bar pb(ijk_std.rows());
    std::cout << "Computing krigging matrix rows" << std::endl;
    
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
	    bounded_hr_index.push_back(
	      (current_ijk_hr[2] - bb_hr.ijk_min[2]) * bhr_dims[0] * bhr_dims[1]  +
	      (current_ijk_hr[1] - bb_hr.ijk_min[1]) * bhr_dims[0] +
	      (current_ijk_hr[0] - bb_hr.ijk_min[0]));                  // (i)
	    perturbation_index.push_back(j);                            // (ii)
	    dist = (Qform_std * current_ijk_std -
		    Qform_hr * current_ijk_hr.cast<float>()).norm();
	    kernel_distances.push_back(dualres::kernels::rbf(dist,
              rbf_params[1], rbf_params[2], rbf_params[0]));            // (iii)
	    row_count++;
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
	k_prime = Eigen::Map<Eigen::VectorXf>(kernel_distances.data(), kernel_distances.size());
	_w_ = dualres::utilities::eigen_select_symmetric(K, perturbation_index)
	  .colPivHouseholderQr().solve(k_prime);
	w_sum = _w_.sum();
	if (w_sum == 0) w_sum = 1;
	if (!isnan(w_sum)) {
	  _w_ /= w_sum;
	  kmd._Data.insert(kmd._Data.end(), std::make_move_iterator(_w_.data()),
			   std::make_move_iterator(_w_.data() + _w_.size()));
	  kmd.column_indices.insert(kmd.column_indices.end(), bounded_hr_index.begin(),
				    bounded_hr_index.end());
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
		<< "% of krigging matrix rows are all zero"
		<< std::endl;
    }

    return kmd;
  };



  


  template< typename Scalar = float >
  af::array sparse_krigging_array(
    const nifti_image* const high_res,
    const nifti_image* const std_res,
    const std::vector<float> &rbf_params,
    const float &neighborhood_radius,
    const af::dim4 &extended_grid_dims
  ) {
    const dualres::nifti_bounding_box bb_hr = dualres::get_bounding_box(high_res);
    const Eigen::Vector3i bhr_dims = bb_hr.ijk_max - bb_hr.ijk_min +
      Eigen::Vector3i::Ones();
    const af::dim4 bhr_grid(bhr_dims[0], bhr_dims[1], bhr_dims[2]);
    for (int i = 0; i < 3; i++) {
      if (bhr_grid[i] > extended_grid_dims[i])
	throw std::logic_error(
          "sparse_krigging_array: old grid should be <= new grid");
    }
    dualres::krigging_matrix_data<float> kmd =
      dualres::get_sparse_krigging_array_data<Scalar>(
        high_res, std_res, rbf_params, neighborhood_radius);
    dualres::re_grid_krigging_matrix_data<float>(kmd, bhr_grid, extended_grid_dims);
    return dualres::construct_sparse_krigging_array<float>(kmd);
  };



  
  
}


#endif  // _DUALRES_KRIGGING_MATRIX_


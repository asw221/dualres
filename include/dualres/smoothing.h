
#include <cmath>
#include <Eigen/Core>
#include <stdexcept>

#include "dualres/eigen_slicing.h"
#include "dualres/kernels.h"
#include "dualres/kriging_matrix.h"
#include "dualres/nifti_manipulation.h"



namespace dualres {

  

  template< typename ImageType = float >
  void local_rbf_smooth_impl(
    nifti_image* nii,
    const ImageType neighborhood_radius,
    const ImageType bandwidth,
    const ImageType exponent = 2.0  // default gaussian kernel
  ) {
    typedef typename Eigen::Matrix<ImageType, Eigen::Dynamic, 1> VectorType;
    
    const int nvox = (int)nii->nvox;
    const dualres::qform_type Qform = dualres::qform_matrix(nii);
    const std::vector<ImageType> rbf_params{1.0, bandwidth, exponent};
    ImageType* voxel_ptr = (ImageType*)nii->data;
    ImageType voxel_value;

    const Eigen::MatrixXi P = dualres::neighborhood_perturbation(
      Qform, neighborhood_radius);

    if (P.rows() > 5e3)
      throw std::logic_error("local_rbf_smooth_impl: neighborhood >5e3 voxels");

    VectorType weights = (P.cast<ImageType>() *
      dualres::nullary_index(Qform, Eigen::VectorXi::LinSpaced(3, 0, 2),
			     Eigen::VectorXi::LinSpaced(3, 0, 2))
			  .cast<ImageType>().transpose()).rowwise().norm();
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = dualres::kernels::rbf(weights[i], bandwidth, exponent);
    }
    

    const Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
    Eigen::Vector3i current_ijk;
    int empty_set_count = 0, neighborhood_count = 0, voxel_index;

    VectorType _local_data(P.rows());  // 
    VectorType _smoothed(ijk.rows());  // result
    VectorType _w_(P.rows());          // smoothing weights
    ImageType _sum_w_;

    for (int i = 0; i < ijk.rows(); i++) {
      _local_data.setZero();
      _w_ = weights;
      neighborhood_count = 0;
      
      // Loop over perturbations
      for (int j = 0; j < P.rows(); j++) {
	current_ijk = ijk.row(i) + P.row(j);
	voxel_index = current_ijk[2] * nii->nx * nii->ny +
	  current_ijk[1] * nii->nx + current_ijk[0];
	
	if (voxel_index >= 0 && voxel_index < nvox) {
	  voxel_value = *(voxel_ptr + voxel_index);
	  if (!(isnan(voxel_value) || voxel_value == 0)) {
	    _local_data[j] = voxel_value;
	    neighborhood_count++;
	  }
	  else {
	    _w_[j] = 0;
	  }
	}
      }  // for (int j = 0; j < P.rows(); ...
      
      _sum_w_ = _w_.sum();
      if (_sum_w_ == 0) {  // should never happen
	throw std::runtime_error("Fatal error: local_rbf_smooth_impl: bad weights");
      }
      
      _smoothed[i] = (_w_.transpose() * _local_data)[0] / _sum_w_;
      if (neighborhood_count <= 1) {
	empty_set_count++;
      }
    }  // for (int i = 0; i < ijk.rows(); ...
    
    if (empty_set_count > 0) {
      std::cerr << "Warning: local_rbf_smooth_impl: " << empty_set_count
		<< " voxels had no apparent surrounding data ("
		<< ((double)empty_set_count / ijk.rows() * 100)
		<< "%)" << std::endl;
    }
    
    dualres::emplace_nonzero_data<ImageType>(nii, _smoothed);
  };



  template< typename T >
  void local_rbf_smooth(
    nifti_image* nii,
    const T neighborhood_radius,
    const T bandwidth,
    const T exponent = 2.0  // default gaussian kernel
  ) {
    if (dualres::is_float(nii))
      dualres::local_rbf_smooth_impl<float>(
        nii, (float)neighborhood_radius, (float)bandwidth, (float)exponent);
    else if (dualres::is_double(nii))
      dualres::local_rbf_smooth_impl<double>(
        nii, (double)neighborhood_radius, (double)bandwidth, (double)exponent);
    else
      throw std::runtime_error("local_rbf_smooth: unrecognized image data type");
  };
  
  
}





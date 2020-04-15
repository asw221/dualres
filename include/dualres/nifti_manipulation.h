
#include <Eigen/Core>
#include <nifti1_io.h>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include "dualres/defines.h"
#include "dualres/utilities.h"


#ifndef _DUALRES_NIFTI_MANIPULATION_
#define _DUALRES_NIFTI_MANIPULATION_


namespace dualres {

  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> qform_type;

  
  struct nifti_bounding_box {
    Eigen::Vector3i ijk_min;
    Eigen::Vector3i ijk_max;
    int nnz;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  ::nifti_image* nifti_image_read(const std::string &hname, int read_data) {
    // ::nifti_image_read is defined in nifti1_io.h
    const dualres::__internals::path _initial_path = dualres::current_path();
    dualres::__internals::path hpath(hname);
    dualres::current_path(hpath.parent_path());
    ::nifti_image* _nii = ::nifti_image_read(hpath.filename().c_str(), read_data);
    dualres::current_path(_initial_path);
    return _nii;
  };


  void nifti_image_write(::nifti_image* nii, std::string new_filename = "") {
    // ::nifti_image_write is defined in nifti1_io.h
    const dualres::__internals::path _initial_path = dualres::current_path();
    if (new_filename.empty()) {
      new_filename = std::string(nii->fname);
    }
    dualres::__internals::path hpath(new_filename);
    dualres::current_path(hpath.parent_path());
    remove(hpath.c_str());
    ::nifti_set_filenames(nii, hpath.filename().c_str(), 1, 1);
    ::nifti_image_write(nii);
    dualres::current_path(_initial_path);
  };

  

  dualres::nifti_data_type nii_data_type(const ::nifti_image* const nii) {
    dualres::nifti_data_type __dt = dualres::nifti_data_type::OTHER;
    try {
      __dt = static_cast<dualres::nifti_data_type>(nii->datatype);
    }
    catch (...) { ; }
    return __dt;
  };



  bool is_nifti_file(const std::string &fname) {
    // ::is_nifti_file defined in nifti1_io.h
    const dualres::__internals::path _initial_path = dualres::current_path();
    dualres::__internals::path fpath(fname);
    bool is_nifti = dualres::utilities::file_exists(fname);
    if (is_nifti) {
      dualres::current_path(fpath.parent_path());
      is_nifti = (::is_nifti_file(fpath.filename().c_str()) == 1);
      dualres::current_path(_initial_path);
    }
    return is_nifti;
  };



  bool is_float(const ::nifti_image* const nii) {
    return dualres::nii_data_type(nii) ==
      dualres::nifti_data_type::FLOAT;
  };
  
  bool is_double(const ::nifti_image* const nii) {
    return dualres::nii_data_type(nii) ==
      dualres::nifti_data_type::DOUBLE;
  };
  



  bool same_data_types(
    const ::nifti_image* const first_img,
    const ::nifti_image* const second_img
  ) {
    return (static_cast<dualres::nifti_data_type>(first_img->datatype) ==
	    static_cast<dualres::nifti_data_type>(second_img->datatype));
  };

  
  

  dualres::qform_type qform_matrix(const ::nifti_image* const img) {
    // Use of Eigen::Map means the nifti_image* input cannot be marked const
    // return Eigen::Map<dualres::qform_type>(&img->qto_xyz.m[0][0]);
    std::vector<float> m(16);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
	m[i * 4 + j] = img->qto_xyz.m[i][j];
    return Eigen::Map<dualres::qform_type>(m.data());
  };


  Eigen::Vector3f voxel_dimensions(const dualres::qform_type &Q) {
    const Eigen::Matrix<float, 4, 3> I = Eigen::Matrix<float, 4, 3>::Identity();
    return (Q * I).colwise().norm();
  };




  
  template< typename ImageType = float >
  Eigen::MatrixXi get_nonzero_indices_impl(const ::nifti_image* const nii) {
    if (nii->ndim < 3)
      throw std::logic_error("NIfTI image has dim < 3");
    const int nx = nii->nx, ny = nii->ny, nz = nii->nz, nvox = (int)nii->nvox;
    ImageType* dataPtr = (ImageType*)nii->data;
    std::vector<int> indices;
    for (int i = 0; i < nvox; i++) {
      if (!(isnan(*dataPtr) || *dataPtr == 0)) {
	indices.push_back(i);
      }
      dataPtr++;
    }
    Eigen::MatrixXi ijk(indices.size(), 3);
    for (int i = 0; i < indices.size(); i++) {
      ijk(i, 0) = indices[i] % nx;  
      ijk(i, 1) = (indices[i] / nx) % ny;
      ijk(i, 2) = (indices[i] / (nx * ny)) % nz;
      // ijk(i, 0) = (indices[i] / (nz * ny)) % nx;
      // ijk(i, 1) = (indices[i] / nz) % ny;
      // ijk(i, 2) = indices[i] % nz;
    }
    return ijk;
  };


  
  Eigen::MatrixXi get_nonzero_indices(const ::nifti_image* const nii) {
    if (dualres::is_float(nii))
      return dualres::get_nonzero_indices_impl<>(nii);
    else if (dualres::is_double(nii))
      return dualres::get_nonzero_indices_impl<double>(nii);
    else
      throw std::runtime_error("get_nonzero_indices: unrecognized image data type");
    
    // Not reached:
    return Eigen::MatrixXi::Zero(1, 1);
  };


  
  Eigen::MatrixXi get_nonzero_indices_bounded(const ::nifti_image* const nii) {
    Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
    return (ijk.rowwise() - ijk.colwise().minCoeff());
  };




  dualres::nifti_bounding_box get_bounding_box(const ::nifti_image* const nii) {
    dualres::nifti_bounding_box nbb;
    Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
    nbb.ijk_min = ijk.colwise().minCoeff();
    nbb.ijk_max = ijk.colwise().maxCoeff();
    nbb.nnz = ijk.rows();
    return nbb;
  };




  template< typename ResultType = float, typename ImageType = float >
  std::vector<ResultType> get_nonzero_data_impl(const ::nifti_image* const nii) {
    const int nvox = (int)nii->nvox;
    std::vector<ResultType> _data;
    _data.reserve(nvox);
    ImageType* data_ptr = (ImageType*)nii->data;
    ResultType voxel_value;
    for (int i = 0; i < nvox; i++) {
      voxel_value = (ResultType)(*(data_ptr + i));
      if (!(isnan(voxel_value) || voxel_value == 0)) {
	_data.push_back(voxel_value);
      }
    }
    _data.shrink_to_fit();
    return _data;
  };


  template< typename ResultType = float >
  std::vector<ResultType> get_nonzero_data(const ::nifti_image* const nii) {
    if (dualres::is_float(nii))
      return dualres::get_nonzero_data_impl<ResultType, float>(nii);
    else if (dualres::is_double(nii))
      return dualres::get_nonzero_data_impl<ResultType, double>(nii);
    else
      throw std::logic_error("get_nonzero_data: unrecognized image data type");
    
    // Not reached:
    return std::vector<ResultType>{};
  };
  
  



  template< typename ImageType = float >
  std::vector<int> get_bounding_box_nonzero_flat_index_impl(const ::nifti_image* const nii) {
    const dualres::nifti_bounding_box bb = dualres::get_bounding_box(nii);
    const Eigen::Vector3i dims = bb.ijk_max - bb.ijk_min + Eigen::Vector3i::Ones();
    const ImageType* const data_ptr = (ImageType*)nii->data;
    const int nx = nii->nx, ny = nii->ny;  // , nz = nii->nz;
    ImageType voxel_value;
    int nii_index, bounded_index, count = 0;
    std::vector<int> index(dims.prod());
    for (int k = 0; k < dims[2]; k++) {
      for (int j = 0; j < dims[1]; j++) {
	for (int i = 0; i < dims[0]; i++) {
	  nii_index = (k + bb.ijk_min[2]) * nx * ny +
	    (j + bb.ijk_min[1]) * nx + (i + bb.ijk_min[0]);
	  bounded_index = k * dims[0] * dims[1] + j * dims[0] + i;
	  // nii_index = (k + bb.ijk_min[2]) + (j + bb.ijk_min[1]) * nz +
	  //   (i + bb.ijk_min[0]) * nz * ny;
	  // bounded_index = k + j * dims[2] + i * dims[2] * dims[3];
	  voxel_value = *(data_ptr + nii_index);
	  if (!(isnan(voxel_value) || voxel_value == 0)) {
	    index[bounded_index] = count;
	    count++;
	  }
	  else {
	    index[bounded_index] = -1;
	  }
	}
      }
    }
    return index;
  };

  
  std::vector<int> get_bounding_box_nonzero_flat_index(const ::nifti_image* const nii) {
    if (dualres::is_float(nii))
      return dualres::get_bounding_box_nonzero_flat_index_impl<float>(nii);
    else if (dualres::is_double(nii))
      return dualres::get_bounding_box_nonzero_flat_index_impl<double>(nii);
    else
      throw std::logic_error("get_bounding_box_nonzero_flat_index: unrecognized image data type");
    
    // Not reached:
    return std::vector<int>{};
  };

  



  template< typename ImageType = float >
  int count_nonzero_voxels_impl(const ::nifti_image* const nii) {
    const ImageType* const data_ptr = (ImageType*)nii->data;
    const int nvox = (int)nii->nvox;
    int count = 0;
    ImageType voxel_value;
    for (int i = 0; i < nvox; i++) {
      voxel_value = *(data_ptr + i);
      if (!(isnan(voxel_value) || voxel_value == 0))  count++;
    }
    return count;
  };


  int count_nonzero_voxels(const ::nifti_image* const nii) {
    if (dualres::is_float(nii))
      return dualres::count_nonzero_voxels_impl<float>(nii);
    else if (dualres::is_double(nii))
      return dualres::count_nonzero_voxels_impl<double>(nii);
    else
      throw std::logic_error("count_nonzero_voxels: unrecognized image data type");
    
    // Not reached:
    return 0;
  };




  template< typename ImageType = float, typename DataType >
  void emplace_nonzero_data_impl(
    ::nifti_image* nii,
    const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
  ) {
    const int nvox = (int)nii->nvox;
    ImageType* nii_ptr = (ImageType*)nii->data;
    ImageType voxel_value;
    for (int i = 0, j = 0; i < nvox && j < nzdat.size(); i++) {
      voxel_value = (*nii_ptr);
      if (!(isnan(voxel_value) || voxel_value == 0)) {
	(*nii_ptr) = (ImageType)nzdat[j];
	j++;
      }
      ++nii_ptr;
    }
  };


  template< typename DataType >
  void emplace_nonzero_data(
    ::nifti_image* nii,
    const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
  ) {
    if (dualres::is_float(nii))
      dualres::emplace_nonzero_data_impl<float, DataType>(nii, nzdat);
    else if (dualres::is_double(nii))
      dualres::emplace_nonzero_data_impl<double, DataType>(nii, nzdat);
    else
      throw std::runtime_error("emplace_nonzero_data: unrecognized image data type");
  };

  
}  // namespace dualres












  // af::array qform_matrix_to_array(const dualres::qform_type &Q) {
  //   af::array qnew = af::array(Q.cols(), Q.rows(), Q.data()).T();
  //   return qnew;
  // };

  // af::array voxel_dimensions(const af::array &Q) {
  //   if (!(Q.dims(0) == 4 && Q.dims(1) == 4)) {
  //     throw std::logic_error("voxel_dimensions: input is not of correct dimensions");
  //   }
  //   return af::sqrt(af::sum(af::pow(af::matmul(Q, af::identity(4, 3)), 2)));
  // };


  // template< typename ResultType = float >
  // af::array get_nonzero_data_array(const nifti_image* const nii) {
  //   std::vector<ResultType> _data = dualres::get_nonzero_data<ResultType>(nii);
  //   return af::array(_data.size(), _data.data());
  // };


  // template< typename ResultType = float >
  // af::array put_data_in_extended_grid(
  //   const nifti_image* const nii,
  //   const af::dim4 &grid_dim
  // ) {
  //   // std::vector<ResultType> nz_data = dualres::get_nonzero_data<ResultType>(nii);
  //   af::array _y = dualres::get_nonzero_data_array<ResultType>(nii);
  //   Eigen::MatrixXi ijk = dualres::get_nonzero_indices_bounded(nii);
  //   Eigen::Vector3i ijk_max = ijk.colwise().maxCoeff();
  //   int elements = 1;
  //   for (int i = 0; i < 3; i++) {
  //     elements *= grid_dim[i];
  //     if (ijk_max[i] >= grid_dim[i]) {	
  // 	throw std::logic_error(
  //         "put_data_in_extended_grid: original grid is larger than the extended");
  //     }
  //   }
  //   Eigen::VectorXi flat_index = ijk.col(2) * grid_dim[1] * grid_dim[0] +
  //     ijk.col(1) * grid_dim[0] + ijk.col(0);
  //   af::array _index(flat_index.size(), flat_index.data());
  //   // af::array _y(nz_data.size(), nz_data.data());
  //   af::array _y_ext = af::constant(0, elements, dualres::data_types<ResultType>::af_dtype);
  //   _y_ext(_index) = _y;
  //   return _y_ext;
  // };







#endif  // _DUALRES_NIFTI_MANIPULATION_

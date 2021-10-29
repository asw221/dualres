
#include <Eigen/Core>
#include <nifti1_io.h>
#include <string>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include "dualres/defines.h"
#include "dualres/eigen_slicing.h"
#include "dualres/utilities.h"






// --- A -------------------------------------------------------------

void dualres::add_to(
  ::nifti_image* first_img,
  const ::nifti_image* const second_img
) {
  if (!dualres::same_orientation(first_img, second_img)) {
    throw std::domain_error(
      "Image math does not make sense: different orientations");
  }
  if (first_img->nvox != second_img->nvox) {
    throw std::domain_error(
      "Image math does not make sense: different grid sizes");
  }
  if (dualres::is_float(first_img) &&
      dualres::is_float(second_img)) {
    dualres::add_to_impl<>(first_img, second_img);
  }
  else if (dualres::is_float(first_img) &&
	   dualres::is_double(second_img)) {
    dualres::add_to_impl<float, double>(first_img, second_img);
  }
  else if (dualres::is_double(first_img) &&
	   dualres::is_float(second_img)) {
    dualres::add_to_impl<double, float>(first_img, second_img);
  }
  else if (dualres::is_double(first_img) &&
	   dualres::is_double(second_img)) {
    dualres::add_to_impl<double, double>(first_img, second_img);
  }
  else {
    throw std::domain_error("Image math not implemented: datatype");
  }
};



template< typename ImageType, typename OtherImageType >
void dualres::add_to_impl(
  ::nifti_image* const A,
  const ::nifti_image* const B
) {
  // A <- A + B
  ImageType* A_data = (ImageType*)A->data;
  OtherImageType* B_data = (OtherImageType*)B->data;
  for (int i = 0; i < (int)A->nvox; i++, ++A_data, ++B_data) {
    (*A_data) += static_cast<ImageType>( *B_data );
  }
};








void dualres::apply_mask(
  ::nifti_image* const img,
  const ::nifti_image* const mask
) {
  if (!dualres::same_orientation(img, mask)) {
    throw std::domain_error(
      "Image masking does not make sense: different orientations");
  }
  if (img->nvox != mask->nvox) {
    throw std::domain_error(
      "Image masking does not make sense: different grid sizes");
  }
  if (dualres::is_float(img) && dualres::is_float(mask)) {
    dualres::apply_mask_impl<>(img, mask);
  }
  else if (dualres::is_float(img) && dualres::is_double(mask)) {
    dualres::apply_mask_impl<float, double>(img, mask);
  }
  else if (dualres::is_float(img) && dualres::is_int(mask)) {
    dualres::apply_mask_impl<float, int>(img, mask);
  }
  else if (dualres::is_double(img) && dualres::is_double(mask)) {
    dualres::apply_mask_impl<double, double>(img, mask);
  }
  else if (dualres::is_double(img) && dualres::is_float(mask)) {
    dualres::apply_mask_impl<double, float>(img, mask);
  }
  else if (dualres::is_double(img) && dualres::is_int(mask)) {
    dualres::apply_mask_impl<double, int>(img, mask);
  }
  else {
    throw std::domain_error("Image masking not implemented: datatype");
  }
};




template< typename ImageType, typename MaskType >
void dualres::apply_mask_impl(
  ::nifti_image* const img,
  const ::nifti_image* const mask
) {
  ImageType * img_ptr = (ImageType*)img->data;
  MaskType * mask_ptr = (MaskType*)mask->data;
  for (int i = 0; i < (int)img->nvox; i++, ++img_ptr, ++mask_ptr) {
    if ((*mask_ptr) == (MaskType)0  ||  isnan(*img_ptr)) {
      (*img_ptr) = (ImageType)0;
    }
  }
};





// --- C -------------------------------------------------------------
  

int dualres::count_nonzero_voxels(const ::nifti_image* const nii) {
  if (dualres::is_float(nii))
    return dualres::count_nonzero_voxels_impl<float>(nii);
  else if (dualres::is_double(nii))
    return dualres::count_nonzero_voxels_impl<double>(nii);
  else
    throw std::logic_error(
      "count_nonzero_voxels: unrecognized image data type");
    
  // Not reached:
  return 0;
};



template< typename ImageType >
int dualres::count_nonzero_voxels_impl(const ::nifti_image* const nii) {
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





// --- E -------------------------------------------------------------


template< typename DataType >
void dualres::emplace_nonzero_data(
  ::nifti_image* nii,
  const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
) {
  if (dualres::is_float(nii))
    dualres::emplace_nonzero_data_impl<float, DataType>(nii, nzdat);
  else if (dualres::is_double(nii))
    dualres::emplace_nonzero_data_impl<double, DataType>(nii, nzdat);
  else
    throw std::runtime_error(
      "emplace_nonzero_data: unrecognized image data type");
};



template< typename ImageType, typename DataType >
void dualres::emplace_nonzero_data_impl(
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





// --- G -------------------------------------------------------------


dualres::nifti_bounding_box dualres::get_bounding_box(
  const ::nifti_image* const nii
) {
  dualres::nifti_bounding_box nbb;
  Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
  nbb.ijk_min = ijk.colwise().minCoeff();
  nbb.ijk_max = ijk.colwise().maxCoeff();
  nbb.nnz = ijk.rows();
  return nbb;
};





std::vector<int> dualres::get_bounding_box_nonzero_flat_index(
  const ::nifti_image* const nii
) {
  if (dualres::is_float(nii))
    return dualres::get_bounding_box_nonzero_flat_index_impl
      <float>(nii);
  else if (dualres::is_double(nii))
    return dualres::get_bounding_box_nonzero_flat_index_impl
      <double>(nii);
  else
    throw std::logic_error(
      "get_bounding_box_nonzero_flat_index: unrecognized image data type");
    
  // Not reached:
  return std::vector<int>{};
};


template< typename ImageType >
std::vector<int> dualres::get_bounding_box_nonzero_flat_index_impl(
  const ::nifti_image* const nii
) {
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



  

template< typename ResultType >
std::vector<ResultType> dualres::get_nonzero_data(
  const ::nifti_image* const nii
) {
  if (dualres::is_float(nii))
    return dualres::get_nonzero_data_impl<ResultType, float>(nii);
  else if (dualres::is_double(nii))
    return dualres::get_nonzero_data_impl<ResultType, double>(nii);
  else
    throw std::logic_error("get_nonzero_data: unrecognized image data type");
    
  // Not reached:
  return std::vector<ResultType>{};
};


template< typename ResultType, typename ImageType >
std::vector<ResultType> dualres::get_nonzero_data_impl(
  const ::nifti_image* const nii
) {
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
  
  


Eigen::MatrixXi dualres::get_nonzero_indices(
  const ::nifti_image* const nii
) {
  if (dualres::is_float(nii))
    return dualres::get_nonzero_indices_impl<>(nii);
  else if (dualres::is_double(nii))
    return dualres::get_nonzero_indices_impl<double>(nii);
  else
    throw std::runtime_error("get_nonzero_indices: unrecognized image data type");
    
  // Not reached:
  return Eigen::MatrixXi::Zero(1, 1);
};


template< typename ImageType >
Eigen::MatrixXi dualres::get_nonzero_indices_impl(
  const ::nifti_image* const nii
) {
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
  for (int i = 0; i < (int)indices.size(); i++) {
    // Column-major order
    ijk(i, 0) = indices[i] % nx;  
    ijk(i, 1) = (indices[i] / nx) % ny;
    ijk(i, 2) = (indices[i] / (nx * ny)) % nz;
    // ijk(i, 0) = (indices[i] / (nz * ny)) % nx;
    // ijk(i, 1) = (indices[i] / nz) % ny;
    // ijk(i, 2) = indices[i] % nz;
  }
  return ijk;
};



  
Eigen::MatrixXi dualres::get_nonzero_indices_bounded(
  const ::nifti_image* const nii
) {
  Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
  return (ijk.rowwise() - ijk.colwise().minCoeff());
};






Eigen::MatrixXi dualres::get_nonzero_indices_bounded_by_box(
  const ::nifti_image* const nii,
  const dualres::nifti_bounding_box& bbox
) {
  Eigen::MatrixXi ijk = dualres::get_nonzero_indices(nii);
  Eigen::VectorXi in_box(ijk.rows());
  const Eigen::VectorXi all_cols_ =
    Eigen::VectorXi::LinSpaced(ijk.cols(), 0, ijk.cols() - 1);
  int n = 0;
  bool row_ok;
  for (int i = 0; i < ijk.rows(); i++) {
    row_ok = (ijk.row(i).transpose().array() >= bbox.ijk_min.array()).all() &&
      (ijk.row(i).transpose().array() <= bbox.ijk_max.array()).all();
    if (row_ok) {
      ijk.row(i) -= bbox.ijk_min;
      in_box[n] = i;
      n++;
    }
  }
  in_box.conservativeResize(n);
  return dualres::nullary_index(ijk, in_box, all_cols_);
};







Eigen::MatrixXi dualres::get_nonzero_indices_bounded_by_mask(
  const ::nifti_image* const nii,
  const ::nifti_image* const mask
) {
  if (!dualres::same_orientation(nii, mask)) {
    throw std::domain_error(
      "Bounded masking does not make sense: different orientations");
  }
  if (nii->nvox != mask->nvox) {
    throw std::domain_error(
      "Bounded masking does not make sense: different grid sizes");
  }
  dualres::nifti_bounding_box bbmask = dualres::get_bounding_box(mask);
  return dualres::get_nonzero_indices_bounded_by_box(nii, bbmask);
};




  




// --- I -------------------------------------------------------------



bool dualres::is_double(const ::nifti_image* const nii) {
  return dualres::nii_data_type(nii) ==
    dualres::nifti_data_type::DOUBLE;
};
  


bool dualres::is_float(const ::nifti_image* const nii) {
  return dualres::nii_data_type(nii) ==
    dualres::nifti_data_type::FLOAT;
};


bool dualres::is_int(const ::nifti_image* const nii) {
  return dualres::nii_data_type(nii) ==
    dualres::nifti_data_type::INT;
};




bool dualres::is_nifti_file(const std::string &fname) {
  // ::is_nifti_file defined in nifti1_io.h
  const dualres::path _initial_path = dualres::current_path();
  dualres::path fpath(fname);
  bool is_nifti = dualres::utilities::file_exists(fname);
  if (is_nifti) {
    dualres::current_path(fpath.parent_path());
    is_nifti = (::is_nifti_file(fpath.filename().c_str()) == 1);
    dualres::current_path(_initial_path);
  }
  return is_nifti;
};



bool dualres::is_unknown_datatype(const ::nifti_image* const nii) {
  return dualres::nii_data_type(nii) ==
    dualres::nifti_data_type::OTHER;
};





// --- N -------------------------------------------------------------


std::string dualres::nifti_datatype_string(
  const ::nifti_image* const nii
) {
  std::string __dt(::nifti_datatype_string(nii->datatype));
  return __dt;
};



::nifti_image* dualres::nifti_image_read(
  const std::string &hname,
  int read_data
) {
  // ::nifti_image_read is defined in nifti1_io.h
  const dualres::path _initial_path = dualres::current_path();
  dualres::path hpath(hname);
  dualres::current_path(hpath.parent_path());
  ::nifti_image* _nii = ::nifti_image_read(hpath.filename().c_str(), read_data);
  dualres::current_path(_initial_path);
  return _nii;
};


  
void dualres::nifti_image_write(
  ::nifti_image* nii,
  std::string new_filename
) {
  // ::nifti_image_write is defined in nifti1_io.h
  const dualres::path _initial_path = dualres::current_path();
  if (new_filename.empty()) {
    new_filename = std::string(nii->fname);
  }
  dualres::path hpath(new_filename);
  dualres::current_path(hpath.parent_path());
  remove(hpath.filename().c_str());
  ::nifti_set_filenames(nii, hpath.filename().c_str(), 1, 1);
  ::nifti_image_write(nii);
  dualres::current_path(_initial_path);
};

  

dualres::nifti_data_type dualres::nii_data_type(
  const ::nifti_image* const nii
) {
  dualres::nifti_data_type __dt = dualres::nifti_data_type::OTHER;
  try {
    __dt = static_cast<dualres::nifti_data_type>(nii->datatype);
  }
  catch (...) { ; }
  return __dt;
};





// --- Q -------------------------------------------------------------
  

dualres::qform_type dualres::qform_matrix(
  const ::nifti_image* const img
) {
  std::vector<float> m(16);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m[i * 4 + j] = img->qto_xyz.m[i][j];
  return Eigen::Map<dualres::qform_type>(m.data());
};





// --- S -------------------------------------------------------------


bool dualres::same_data_types(
  const ::nifti_image* const first_img,
  const ::nifti_image* const second_img
) {
  return (static_cast<dualres::nifti_data_type>
	  (first_img->datatype) ==
	  static_cast<dualres::nifti_data_type>
	  (second_img->datatype));
};



bool dualres::same_orientation(
  const ::nifti_image* const first_img,
  const ::nifti_image* const second_img,
  const float tol
) {
  const float m = ( dualres::qform_matrix(first_img) -
		    dualres::qform_matrix(second_img) )
    .cwiseAbs().maxCoeff();
  return ( m <= tol );
};





// --- V -------------------------------------------------------------


Eigen::Vector3f dualres::voxel_dimensions(
  const ::nifti_image* const nii
) {
  Eigen::Vector3f dims_;
  dims_ << nii->dx, nii->dy, nii->dz;
  return dims_;
};



Eigen::Vector3f dualres::voxel_dimensions(
  const dualres::qform_type &Q
) {
  const Eigen::Matrix<float, 4, 3> I = Eigen::Matrix<float, 4, 3>::Identity();
  return (Q * I).colwise().norm();
};





  

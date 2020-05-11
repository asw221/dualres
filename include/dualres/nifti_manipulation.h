
#include <Eigen/Core>
#include <nifti1_io.h>
#include <string>
#include <vector>

#include "dualres/defines.h"


#ifndef _DUALRES_NIFTI_MANIPULATION_
#define _DUALRES_NIFTI_MANIPULATION_


/*! @defgroup NiftiManipulation
 * 
 * Collection of functions to extract data and summary information
 * from \c nifti_image structures (defined in 
 * <a href="https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1_io.h">nifti1_io.h</a>).
 */



namespace dualres {
  /*! @addtogroup NiftiManipulation
   * @{
   */


  // === Type/Object definitions =====================================
  
  /*!
   * NIfTI \b Q-form matrix type
   */
  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> qform_type;


  /*!
   * Information about where brain data lives on a NIfTI grid
   */
  struct nifti_bounding_box {
    Eigen::Vector3i ijk_min;  /*!< Smallest (i, j, k) grid index with 
			       * nonzero brain data */
    Eigen::Vector3i ijk_max;  /*!< Largest (i, j, k) grid index with 
			       * nonzero brain data */
    int nnz;                  /*!< Number of nonzero brain voxels */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  // === Function definitions ========================================


  // --- C -----------------------------------------------------------

  int count_nonzero_voxels(const ::nifti_image* const nii);


  
  // --- E -----------------------------------------------------------

  template< typename DataType >
  void emplace_nonzero_data(
    ::nifti_image* nii,
    const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
  );  


  
  // --- G -----------------------------------------------------------

  dualres::nifti_bounding_box get_bounding_box(const ::nifti_image* const nii);
  
  
  std::vector<int> get_bounding_box_nonzero_flat_index(
    const ::nifti_image* const nii
  );


  template< typename ResultType = float >
  std::vector<ResultType> get_nonzero_data(const ::nifti_image* const nii);


  Eigen::MatrixXi get_nonzero_indices(const ::nifti_image* const nii);

  
  Eigen::MatrixXi get_nonzero_indices_bounded(const ::nifti_image* const nii);
  


  
  // --- I -----------------------------------------------------------
  
  bool is_double(const ::nifti_image* const nii);
  

  bool is_float(const ::nifti_image* const nii);


  bool is_nifti_file(const std::string &fname);





  // --- N -----------------------------------------------------------
  
  std::string nifti_datatype_string(const ::nifti_image* const nii);
  

  /*!
   * Read information from a NIfTI file.
   * 
   * Output is a pointer to a \c nifti_image structure, as defined in
   * <a href="https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1_io.h">nifti1_io.h</a>.
   *
   * If the input parameter \c read_data is set to \c 1, NIfTI "brick" 
   * data will be read into the output's \c data field, and the 
   * resulting \c nifti_image* pointer must be deallocated with 
   * \c ::nifti_image_free. Data is read in column-major order.
   * If this parameter is set to \c 0,
   * only the file's header information will be read and an error 
   * will be thrown if later deallocated with \c ::nifti_image_free.
   *
   * Defined in file nifti_manipulation.h
   * 
   * @param hname Path to a NIfTI header file.
   * @param read_data The value \c 0 instructs to read header info; 
   *   value \c 1 additionally reads NIfTI "brick" data
   */
  ::nifti_image * nifti_image_read(const std::string &hname, int read_data);

  
  void nifti_image_write(::nifti_image* nii, std::string new_filename = "");
  

  dualres::nifti_data_type nii_data_type(const ::nifti_image* const nii);


  
  // --- Q -----------------------------------------------------------

  dualres::qform_type qform_matrix(const ::nifti_image* const img);


  
  // --- S -----------------------------------------------------------

  bool same_data_types(
    const ::nifti_image* const first_img,
    const ::nifti_image* const second_img
  );


  
  // --- V -----------------------------------------------------------
  
  Eigen::Vector3f voxel_dimensions(const ::nifti_image* const nii);


  Eigen::Vector3f voxel_dimensions(const dualres::qform_type &Q);



  

  
  /// @cond IMPL
  /*
   * Implementation of \c dualres::emplace_nonzero_data
   */
  template< typename ImageType = float, typename DataType >
  void emplace_nonzero_data_impl(
    ::nifti_image* nii,
    const Eigen::Matrix<DataType, Eigen::Dynamic, 1> &nzdat
  );

  
  /*
   * Implementation of \c dualres::get_bounding_box_nonzero_flat_index
   */
  template< typename ImageType = float >
  std::vector<int> get_bounding_box_nonzero_flat_index_impl(
    const ::nifti_image* const nii
  );

  
  /*
   * Implementation of \c dualres::get_nonzero_data
   */
  template< typename ResultType = float, typename ImageType = float >
  std::vector<ResultType> get_nonzero_data_impl(const ::nifti_image* const nii);

  
  /*
   * Implementation of \c dualres::get_nonzero_indices
   */
  template< typename ImageType = float >
  Eigen::MatrixXi get_nonzero_indices_impl(const ::nifti_image* const nii);

  
  /*
   * Implementation of \c dualres::count_nonzero_voxels
   */
  template< typename ImageType = float >
  int count_nonzero_voxels_impl(const ::nifti_image* const nii);

  /// @endcond



  /*! @} */
}  // namespace dualres





#include "dualres/impl/nifti_manipulation.inl"


#endif  // _DUALRES_NIFTI_MANIPULATION_

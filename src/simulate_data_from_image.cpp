
#include <algorithm>
#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "dualres/CommandParser.h"
#include "dualres/kernels.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/smoothing.h"
#include "dualres/utilities.h"


int main(int argc, char *argv[]) {
  typedef float scalar_type;
  typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> VectorType;

  const dualres::SmoothingCommandParser<scalar_type> inputs(argc, argv);
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  /*
    - Read input data file
    - Use RBF smoothing as "nonparametric" estimate of true activation
    - Add permutation of actual residuals back to smooth
    - Write out '*_activation.nii' and '*_simdata.nii' files
   */
  
  
  const scalar_type bandwidth = dualres::kernels::rbf_fwhm_to_bandwidth(
    inputs.fwhm(), inputs.exponent());

  bool error_status = false;
  scalar_type radius = inputs.radius();
  std::ostringstream new_fname_stream;
  ::nifti_image* _nii;
  std::vector<scalar_type> _nii_data, _nii_smoothed_data;

  try {
    _nii = dualres::nifti_image_read(inputs.image_file(), 1);
    _nii_data = dualres::get_nonzero_data<scalar_type>(_nii);
  
    if (radius <= 0) {
      radius = 3 * (scalar_type)dualres::voxel_dimensions(
        dualres::qform_matrix(_nii)).array().maxCoeff();
    }

    new_fname_stream << ::nifti_makebasename(_nii->fname) << "_"
		     << ((int)inputs.fwhm()) << "mm_fwhm";

    

    // Use RBF smoothing as a "nonparametric" estimate of
    // true activation
    dualres::local_rbf_smooth(_nii, radius, bandwidth, inputs.exponent());

    // Write "true" activation image
    dualres::nifti_image_write(_nii,
      (new_fname_stream.str() + "_activation.nii"));


    // --- Data simulation -------------------------------------------
    // Compute "residuals" into _nii_data
    _nii_smoothed_data = dualres::get_nonzero_data<scalar_type>(_nii);
    for (int i = 0; i < _nii_data.size(); i++) {
      _nii_data[i] -= _nii_smoothed_data[i];
    }
    
    // Permute residuals
    std::shuffle(_nii_data.begin(), _nii_data.end(), dualres::rng());

    // Add mean field back
    for (int i = 0; i < _nii_data.size(); i++) {
      _nii_data[i] += _nii_smoothed_data[i];
    }
    dualres::emplace_nonzero_data<scalar_type>(
      _nii, Eigen::Map<VectorType>(_nii_data.data(), _nii_data.size()));

    // Write simulated data file
    dualres::nifti_image_write(_nii,
      (new_fname_stream.str() + "_simdata.nii"));
    
    ::nifti_image_free(_nii);
  }
  catch (const std::exception &__err) {
    error_status = true;
    std::cerr << "Exception caught with message:\n'"
	      << __err.what() << "'\n"
	      << std::endl;
  }
  catch (...) {
    error_status = true;
    std::cerr << "Unknown error\n";
  }

  if (error_status)  return 1;
}




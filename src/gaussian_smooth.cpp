
#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdio.h>
#include <string>

#include "dualres/CommandParser.h"
#include "dualres/kernels.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/smoothing.h"


int main(int argc, char *argv[]) {
  typedef float scalar_type;

  const dualres::SmoothingCommandParser<scalar_type> inputs(argc, argv);
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  
  const scalar_type bandwidth = dualres::kernels::rbf_fwhm_to_bandwidth(
    inputs.fwhm(), inputs.exponent());

  bool error_status = false;
  scalar_type radius = inputs.radius();
  std::ostringstream new_fname_stream;
  nifti_image* _nii;

  try {
    _nii = dualres::nifti_image_read(inputs.image_file(), 1);
  
    if (radius <= 0) {
      radius = 3 * (scalar_type)dualres::voxel_dimensions(
        dualres::qform_matrix(_nii)).array().maxCoeff();
    }

    new_fname_stream << nifti_makebasename(_nii->fname) << "_"
		     << ((int)inputs.fwhm()) << "mm_fwhm.nii";

    dualres::local_rbf_smooth(_nii, radius, bandwidth, inputs.exponent());
    dualres::nifti_image_write(_nii, new_fname_stream.str());
    
    nifti_image_free(_nii);
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




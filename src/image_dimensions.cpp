
#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <nifti1_io.h>
#include <string>

#include "dualres/nifti_manipulation.h"



int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "\nUsage:\n\timage_dimensions /path/to/img\n";
    return 1;
  }
  
  const std::string _input_image_file(argv[1]);
  bool error_status = false;
  ::nifti_image* _nii;
  Eigen::Vector3f _voxel_dims;
  
  if (!dualres::is_nifti_file(_input_image_file)) {
    std::cerr << "\nimage_dimensions: requires one NIfTI file as input\n";
    return 1;
  }

  try {
    _nii = dualres::nifti_image_read(_input_image_file, 0);
    _voxel_dims = dualres::voxel_dimensions(dualres::qform_matrix(_nii));
    
    std::cout << _input_image_file << ":\n"
	      << "  Grid size       -  ("
	      << _nii->nx << ", " << _nii->ny << ", " << _nii->nz
	      << ")\n"
	      << "  Voxel dim (mm)  -  ("
	      << _voxel_dims[0] << ", " << _voxel_dims[1] << ", " << _voxel_dims[2]
	      << ")\n";

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




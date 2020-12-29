
#include <Eigen/Core>
#include <exception>
#include <iostream>
#include <nifti1_io.h>
#include <string>

#include "dualres/ansi.h"
#include "dualres/nifti_manipulation.h"



/*
 * Print basic information about an input NIfTI file
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "\nUsage:\n\timage_info /path/to/img\n";
    return 1;
  }
  
  const std::string _input_image_file(argv[1]);
  bool error_status = false;
  bool print_qform = false;
  // ::nifti_image* _nii;
  // Eigen::Vector3f _voxel_dims;
  
  if (!dualres::is_nifti_file(_input_image_file)) {
    std::cerr << "image_info: requires one NIfTI file as input\n";
    return 1;
  }
  if (argc > 2 && std::string(argv[2]) == "--qform") {
    print_qform = true;
  }
  

  try {
    ::nifti_image* _nii = dualres::nifti_image_read(_input_image_file, 0);
    // _voxel_dims = dualres::voxel_dimensions(_nii);
    
    std::cout << "\n"
	      << ansi::foreground_bold
	      << _input_image_file << ":\n"
	      << ansi::reset
	      << "---------------\n"
	      << "  Data type       -  "
	      << dualres::nifti_datatype_string(_nii)
	      << "\n"
	      << "  Grid size       -  ("
	      << _nii->nx << ", " << _nii->ny << ", " << _nii->nz
	      << ")\n"
	      << "  Voxel dim (mm)  -  ("
              // << _voxel_dims[0] << ", " << _voxel_dims[1] << ", " << _voxel_dims[2]
	      // << ")\n"
	      // << "  Grid spacings   -  ("
	      << _nii->dx << ", " << _nii->dy << ", " << _nii->dz
	      << ")\n";

    if (print_qform) {
      std::cout << "\n"
		<< ansi::foreground_bold
		<< "Qform matrix:\n"
		<< ansi::reset
		<< "---------------\n"
		<< dualres::qform_matrix(_nii)
		<< "\n\n";
    }

    ::nifti_image_free(_nii);
  }
  catch (const std::exception &__err) {
    error_status = true;
    std::cerr << ansi::foreground_bold_magenta
	      << "\nException caught with message:\n"
	      << ansi::reset
	      << "'" << __err.what() << "'\n"
	      << std::endl;
  }
  catch (...) {
    error_status = true;
    std::cerr << ansi::foreground_bold_magenta
	      << "\nUnknown error\n"
	      << ansi::reset;
  }

  if (error_status)  return 1;
}




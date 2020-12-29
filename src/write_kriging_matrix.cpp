
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <exception>
#include <fstream>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "dualres/CommandParser.h"
#include "dualres/kernels.h"
#include "dualres/kriging_matrix.h"
#include "dualres/nifti_manipulation.h"



int main(int argc, char *argv[]) {
  typedef float scalar_type;
  typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> VectorType;
  typedef typename Eigen::SparseMatrix<scalar_type, Eigen::RowMajor> SparseMatrixType;

  const dualres::KrigingCommandParser<scalar_type> inputs(argc, argv);
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  
  const scalar_type bandwidth = dualres::kernels::rbf_fwhm_to_bandwidth(
    inputs.fwhm(), inputs.exponent());

  bool error_status = false;
  scalar_type radius = inputs.radius();
  std::ostringstream new_fname_stream;
  ::nifti_image *_nii, *_output_nii;

  const std::vector<scalar_type> covariance_parameters{ 1,
      bandwidth, inputs.exponent() };

  dualres::kriging_matrix_data<scalar_type> kmd;

  
  try {
    _nii = dualres::nifti_image_read(inputs.image_file(), 1);
    new_fname_stream << nifti_makebasename(_nii->fname) << "_"
		     << ((int)inputs.fwhm()) << "mm_fwhm_kriged.nii";
    
    _output_nii = dualres::nifti_image_read(inputs.output_image(), 1);
  
    if (radius <= 0) {
      radius = 3 * (scalar_type)dualres::voxel_dimensions(_output_nii)
	.array().maxCoeff();
    }


    if (dualres::is_float(_nii)) {
      kmd = dualres::get_sparse_kriging_matrix_data<scalar_type, float>(
        _nii, _output_nii, covariance_parameters, radius);
    }
    else if (dualres::is_double(_nii)) {
      kmd = dualres::get_sparse_kriging_matrix_data<scalar_type, double>(
        _nii, _output_nii, covariance_parameters, radius);
    }
    else
      throw std::runtime_error("interpolate: input image is of unrecognized data type");


    // ---------------------------------------------------------------
    std::ofstream out("W.csv");
    
    std::cout << "Writing W:" << std::endl;
    
    out << "Data,ColumnIndex,RowCounts,NRow,Ncol" << std::endl;
    for (int i = 0; i < kmd._Data.size(); i++) {
      out << kmd._Data[i] << ",";
      if (i < kmd.column_indices.size())
	out << kmd.column_indices[i];
      out << ",";
      if (i < kmd.cum_row_counts.size())
	out << kmd.cum_row_counts[i];
      out << ",";
      if (i == 0) {
	out << kmd.nrow << "," << kmd.ncol;
      }
      else {
	out << ",";
      }
      out << std::endl;
    }
    out.close();
    std::cout << "W.csv written" << std::endl;
    // ---------------------------------------------------------------
    
    ::nifti_image_free(_nii);
    ::nifti_image_free(_output_nii);
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




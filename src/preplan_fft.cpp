
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <string>


#include "dualres/CommandParser.h"
#include "dualres/defines.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/sized_fft.h"
#include "dualres/utilities.h"




int main (int argc, char* argv[]) {
#ifdef DUALRES_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  typedef typename std::complex<scalar_type> complex_type;
  typedef typename Eigen::Array<complex_type, Eigen::Dynamic, 1>
    ComplexArrayType;
  
  dualres::FFTWWisdomCommandParser<scalar_type> inputs(argc, argv);
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  dualres::initialize_temporary_directory();
  dualres::set_number_of_threads(inputs.threads());
  // ::fftwf_plan_with_nthreads(dualres::threads());
  std::cout << "[dualres running on " << dualres::threads()
	    << " cores]\n";

  
  // Read image header and find correct eigen vector grid dimensions
  nifti_image* __nii;
  dualres::nifti_bounding_box bbx;
  try {
    __nii = dualres::nifti_image_read(inputs.image_file(), 1);
    bbx = dualres::get_bounding_box(__nii);
    ::nifti_image_free(__nii);
  }
  catch (...) {
    std::cerr << "Error reading data from file: " << inputs.image_file()
	      << std::endl;
    return 1;
  }
  
  Eigen::Array3d temp_grid_dims = 2 *
    (bbx.ijk_max.cast<double>() - bbx.ijk_min.cast<double>()).array();
  if (!inputs.native_grid()) {
    temp_grid_dims = Eigen::pow(2.0,
      (temp_grid_dims.log() / std::log(2.0)).ceil()).eval();
  }
  const Eigen::Vector3i grid_dims =
    temp_grid_dims.cast<int>().matrix();



  // // Import any existing wisdom
  //    ^^ now taken care of by sized_fft class
  // if (dualres::utilities::file_exists(dualres::fftw_wisdom_file().string()))
  //   fftwf_import_wisdom_from_filename(dualres::fftw_wisdom_file().c_str());

  
  // Adopt requested planner flag (patient/exhaustive search)
  unsigned int __flag = FFTW_PATIENT;
  if (inputs.flagged_exhaustive()) {
    __flag = FFTW_EXHAUSTIVE;
  }


  // --- Identify optimal plans --------------------------------------
  // dualres algorithms compute eigen vectors in column-major order
  // (in keeping with indexing of nifti_image->data);
  // -> construct FFTW plans assuming such an order
  std::cout << "Finding optimal FFT plans on grid of size ("
	    << grid_dims[0] << " x " << grid_dims[1] << " x "
	    << grid_dims[2] << ")...  "
	    << std::flush;
  auto start_time = std::chrono::high_resolution_clock::now();
  dualres::sized_fft<scalar_type> __dft(
    grid_dims[0], grid_dims[1], grid_dims[2], __flag);
  // fftwf_plan __forward_fft_plan = fftwf_plan_dft_3d(
  //   grid_dims[2], grid_dims[1], grid_dims[0],
  //   reinterpret_cast<fftwf_complex*>(_lambda.data()),
  //   reinterpret_cast<fftwf_complex*>(_lambda.data()),
  //   FFTW_FORWARD, __flag
  // );
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>
    (stop_time - start_time);
  auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>
    (stop_time - start_time);
  std::cout << "Done!\n\t(computation took ";
  if (duration_sec.count() > 60) {
    std::cout << std::setprecision(2) << std::fixed
	      << ((double)duration_sec.count() / 60) << " min)"
	      << std::endl;
  }
  else {
    std::cout << std::setprecision(2) << std::fixed
	      << ((double)duration.count() / 1e6) << " sec)"
	      << std::endl;
  }

  
  // --- Wisdom export -----------------------------------------------
  int export_status = __dft.save_plans();
  // int export_status = fftwf_export_wisdom_to_filename(
  //   dualres::fftw_wisdom_file().c_str());
  if (export_status == 0) {
    std::cerr << "\nWarning: wisdom file not written!" << std::endl;
  }
  else {
    std::cout << "Wisdom earned: "
	      << dualres::fftw_wisdom_file().string()
	      << std::endl;
  }

  
  // --- Print FFT execution times -----------------------------------
  ComplexArrayType _lambda = ComplexArrayType::Constant(
    grid_dims.prod(), complex_type(0, 0));
  start_time = std::chrono::high_resolution_clock::now();
  // fftwf_execute(__forward_fft_plan);
  __dft.forward(_lambda.data());
  stop_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>
    (stop_time - start_time);
  std::cout << "\nEach forward DFT will take approximately "
	    << std::setprecision(5) << std::fixed
	    << ((double)duration.count() / 1e6) << " sec"
	    << std::endl;
  
  start_time = std::chrono::high_resolution_clock::now();
  // fftwf_execute(__backward_fft_plan);
  __dft.inverse(_lambda.data());
  stop_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>
    (stop_time - start_time);
  std::cout << "Each inverse DFT will take approximately "
	    << std::setprecision(5) << std::fixed
	    << ((double)duration.count() / 1e6) << " sec\n"
	    << std::endl;


  // --- Cleanup -----------------------------------------------------
#ifdef DUALRES_SINGLE_PRECISION
  ::fftwf_cleanup_threads();
#else
  ::fftw_cleanup_threads();
#endif
}





  
  // // Backward --------------------------------------------------------
  // std::cout << "Finding optimal inverse FFT plan on grid of size ("
  // 	    << grid_dims[0] << " x " << grid_dims[1] << " x " << grid_dims[2]
  // 	    << ")...  " << std::flush;
  // start_time = std::chrono::high_resolution_clock::now();
  // fftwf_plan __backward_fft_plan = fftwf_plan_dft_3d(
  //   grid_dims[2], grid_dims[1], grid_dims[0],
  //   reinterpret_cast<fftwf_complex*>(_lambda.data()),
  //   reinterpret_cast<fftwf_complex*>(_lambda.data()),
  //   FFTW_BACKWARD, __flag
  // );
  // stop_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
  // duration_sec = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
  // std::cout << "Done!\n\t(computation took ";
  // if (duration_sec.count() > 60) {
  //   std::cout << std::setprecision(2) << std::fixed
  // 	      << ((double)duration_sec.count() / 60) << " min)"
  // 	      << std::endl;
  // }
  // else {
  //   std::cout << std::setprecision(2) << std::fixed
  // 	      << ((double)duration.count() / 1e6) << " sec)"
  // 	      << std::endl;
  // }




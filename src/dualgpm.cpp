
#include <Eigen/Core>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <vector>

#include "dualres/CommandParser.h"
#include "dualres/defines.h"
#include "dualres/estimate_kernel_parameters.h"
#include "dualres/gaussian_process_model.h"
#include "dualres/HMCParameters.h"
#include "dualres/kernels.h"
#include "dualres/kriging_matrix.h"
#include "dualres/MultiResData.h"
#include "dualres/MultiResParameters.h"
#include "dualres/nifti_manipulation.h"



// return 1 for error; return 0 for success


template< typename T >
bool compute_kernel_parameters_if_needed(
  std::vector<T> &theta,
  const nifti_image * const input_data
);

template< typename T >
bool valid_kernel_parameters(const std::vector<T> &theta);





int main(int argc, char *argv[]) {
  typedef float scalar_type;
  
  dualres::GPMCommandParser<scalar_type> inputs(argc, argv);
  // Errors if not given at least a "highres" image
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  
  dualres::set_seed(inputs.seed());

  dualres::set_number_of_threads(inputs.threads());
  omp_set_num_threads(dualres::threads());
  Eigen::setNbThreads(dualres::threads());
  fftwf_plan_with_nthreads(dualres::threads());
  std::cout << "[dualres running on " << dualres::threads() << " cores]" << std::endl;


  scalar_type neighborhood = inputs.neighborhood();
  // int n_datasets = 1;
  const bool _standard_resolution_available = !inputs.stdres_file().empty();
  const int _n_datasets = _standard_resolution_available ? 2 : 1;
  
  nifti_image* _high_res_ = nifti_image_read(inputs.highres_file().c_str(), 1);
  nifti_image* _std_res_;

  if (_standard_resolution_available) {
    _std_res_ = nifti_image_read(inputs.stdres_file().c_str(), 1);
    
    if (!dualres::same_data_types(_high_res_, _std_res_)) {
      std::cerr << "Cannot mix image data types!";
      return 1;
    }
  }

  
  std::vector<scalar_type> kernel_params = inputs.kernel_parameters();
  // If kernel parameters are not given, estimate them from inputs. Use
  // Std Res image first if available
  if (_standard_resolution_available) {
    if (!compute_kernel_parameters_if_needed(kernel_params, _std_res_))  return 1;
  }
  else if (!compute_kernel_parameters_if_needed(kernel_params, _high_res_)) {
    return 1;
  }
  if (!valid_kernel_parameters(kernel_params))  return 1;
    

  // If the (mm) neighborhood is not given, compute it from the kernel
  // parameters assuming sparsity after a cuttoff level of 0.1
  if (neighborhood <= 0) {
    // neighborhood = dualres::kernels::rbf_inverse(
    //   (scalar_type)0.1, kernel_params[1], kernel_params[2], kernel_params[0]);
    neighborhood = (scalar_type)dualres::voxel_dimensions(
      dualres::qform_matrix(_high_res_)).array().maxCoeff();
  }
  


  
  if (!inputs.highres_file().empty())
    std::cout << "High-resolution file: " << inputs.highres_file() << std::endl;
  if (!inputs.stdres_file().empty())
    std::cout << "Standard-resolution file: " << inputs.stdres_file() << std::endl;



  // "Body:"
  // const Eigen::MatrixXi ijk_high = dualres::get_nonzero_indices_bounded(_high_res_);  
  dualres::HMCParameters<scalar_type> _hmc_(
    inputs.mcmc_burnin(), inputs.mcmc_nsave(), inputs.mcmc_thin(),
    inputs.mcmc_leapfrog_steps()
  );


  std::vector<scalar_type> __Yh = dualres::get_nonzero_data<scalar_type>(_high_res_);
  std::vector<scalar_type> __Ys;
  dualres::MultiResData<scalar_type> _data_;
  
  if (!_standard_resolution_available) {
    _data_ = dualres::MultiResData<scalar_type>(_high_res_);
  }
  else {
    _data_ = dualres::MultiResData<scalar_type>(
      _high_res_, _std_res_, kernel_params, neighborhood);
  }


  
  
  std::cout << "Kernel parameters: ("
	    << kernel_params[0] << ", " << kernel_params[1] << ", " << kernel_params[2]
	    << ")\nNieghborhood: " << neighborhood << " (mm)"
	    << "\nBurnin   = " << _hmc_.burnin_iterations()
	    << "\nNSave    = " << _hmc_.n_save()
	    << "\nThin     = " << _hmc_.thin_iterations()
	    << "\nLeapfrog = " << inputs.mcmc_leapfrog_steps() // _hmc_.integrator_steps()
	    << std::endl;

  if (_standard_resolution_available) {
    std::cout << "Kriging matrix has dimension: ("
	      << _data_.W().rows() << ", " << _data_.W().cols() << ")"
	      << std::endl;
  }


  std::cout << "(Extended) size of Y's: " << _data_.Yh().size() << ", "
	    << _data_.Ys().size()
	    << std::endl;


  std::ofstream mcmc_samples_file("mcmc_samples.dat~");
  if (mcmc_samples_file) {
  
    dualres::gaussian_process::sor_approx::mcmc_mode<scalar_type> mcmc_summary =
      dualres::gaussian_process::sor_approx::fit_model<scalar_type>(
        _high_res_, _data_, _hmc_, mcmc_samples_file);

    mcmc_samples_file.close();
  }
  else {
    std::cerr << "Could not write to mcmc_samples.dat" << std::endl;
  }
  

  // Finish by calling  nifti_image_free()
  nifti_image_free(_high_res_);
  if (_standard_resolution_available)  nifti_image_free(_std_res_);
  fftwf_cleanup_threads();
}









template< typename T >
bool compute_kernel_parameters_if_needed(
  std::vector<T> &theta,
  const nifti_image * const input_data
) {
  bool success = true;
  if (theta.empty()) {
    std::vector<double> kernel_params_dtemp{1, 0.6, 1.5};
    // ^^ provide better starting values, especially for marginal variance

    std::cout << "Estimating kernel parameters... " << std::flush;
    int kp_success;  // 0 - success, 1 - error
    kp_success = dualres::compute_rbf_parameters(
      kernel_params_dtemp, dualres::compute_mce_summary_data(input_data));
    if (kp_success == 1) {
      std::cerr << "Could not estimate kernel parameters.\n"
		<< "Try re-running with the --kernel argument supplied"
		<< std::endl;
      success = false;
    }
    else {
      for (int i = 0; i < kernel_params_dtemp.size(); i++)
	theta.push_back((T)kernel_params_dtemp[i]);
      std::cout << "Done!" << std::endl;
    }
  }
  else {
  }
  return success;
};


 
template< typename T >
bool valid_kernel_parameters(const std::vector<T> &theta) {
  // Error check input kernel parameters:
  bool success = true;
  if (theta[0] <= 0) {
    std::cerr << "First kernel parameter (marginal variance) must be > 0\n";
    success = false;
  }
  if (theta[1] <= 0) {
    std::cerr << "Second kernel parameter (bandwidth) must be > 0\n";
    success = false;
  }
  if (theta[2] <= 0 || theta[2] > 2) {
    std::cerr << "Third kernel parameter (exponent) must be in (0, 2]\n";
    success = false;
  }
  return success;
};

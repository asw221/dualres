
#include <Eigen/Core>
#include <fftw3.h>
#include <iostream>
#include <nifti1_io.h>
#include <vector>

#include "CommandParser.h"
#include "defines.h"
#include "estimate_kernel_parameters.h"
#include "gaussian_process_model.h"
#include "HMCParameters.h"
#include "kernels.h"
#include "kriging_matrix.h"
#include "MultiResData.h"
#include "MultiResParameters.h"
#include "nifti_manipulation.h"



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


  dualres::set_number_of_threads(inputs.threads());
  Eigen::setNbThreads(6);
  fftw_plan_with_nthreads(dualres::internals::_N_THREADS_);
  
  
  if (!inputs.highres_file().empty())
    std::cout << "High-resolution file: " << inputs.highres_file() << std::endl;
  if (!inputs.stdres_file().empty())
    std::cout << "Standard-resolution file: " << inputs.stdres_file() << std::endl;


  dualres::set_seed(inputs.seed());

  scalar_type neighborhood = inputs.neighborhood();
  int n_datasets = 1;
  
  nifti_image* _high_res_ = nifti_image_read(inputs.highres_file().c_str(), 1);
  nifti_image* _std_res_;

  if (!inputs.stdres_file().empty()) {
    _std_res_ = nifti_image_read(inputs.stdres_file().c_str(), 1);
    n_datasets++;
    
    if (!dualres::same_data_types(_high_res_, _std_res_)) {
      std::cerr << "Cannot mix image data types!";
      return 1;
    }
  }

  
  std::vector<scalar_type> kernel_params = inputs.kernel_parameters();
  // If kernel parameters are not given, estimate them from inputs. Use
  // Std Res image first if available
  if (!inputs.stdres_file().empty()) {
    if (!compute_kernel_parameters_if_needed(kernel_params, _std_res_))  return 1;
  }
  else if (!compute_kernel_parameters_if_needed(kernel_params, _high_res_)) {
    return 1;
  }
  if (!valid_kernel_parameters(kernel_params))  return 1;
    

  // If the (mm) neighborhood is not given, compute it from the kernel
  // parameters assuming sparsity after a cuttoff level of 0.1
  if (neighborhood <= 0) {
    neighborhood = dualres::kernels::rbf_inverse(
      (scalar_type)0.1, kernel_params[1], kernel_params[2], kernel_params[0]);
  }


  // "Body:"
  // const Eigen::MatrixXi ijk_high = dualres::get_nonzero_indices_bounded(_high_res_);  
  dualres::HMCParameters<scalar_type> _hmc_(
    inputs.mcmc_burnin(), inputs.mcmc_nsave(), inputs.mcmc_thin(),
    inputs.mcmc_leapfrog_steps()
  );

  std::cout << "Constructing circulant base and initializing parameters... "
	    << std::flush;
  
  dualres::MultiResParameters<scalar_type> _theta_(
    n_datasets, kernel_params,
    dualres::get_nonzero_indices_bounded(_high_res_),
    dualres::qform_matrix(_high_res_),
    dualres::use_lambda_method::EXTENDED
  );
  std::cout << "Done!" << std::endl;

  dualres::MultiResData<scalar_type> _data_;
  _data_.push_back_data(dualres::put_data_in_extended_grid<scalar_type>(
    _high_res_, _theta_.lambda().dims()));
  
  if (!inputs.stdres_file().empty()) {
    _data_.push_back_data(dualres::get_nonzero_data_array<scalar_type>(_std_res_));
    // push_back_weight
    dualres::kriging_matrix_data<scalar_type> kmd =
      dualres::get_sparse_kriging_array_data<scalar_type>(
        _high_res_, _std_res_, kernel_params, neighborhood
    );
    _data_.push_back_weight(kmd);
  }

  
  
  std::cout << "Kernel parameters: ("
	    << kernel_params[0] << ", " << kernel_params[1] << ", " << kernel_params[2]
	    << ")\nNieghborhood: " << neighborhood << " (mm)"
	    << "\nBurnin   = " << _hmc_.burnin_iterations()
	    << "\nNSave    = " << _hmc_.n_save()
	    << "\nThin     = " << _hmc_.thin_iterations()
	    << "\nLeapfrog = " << inputs.mcmc_leapfrog_steps() // _hmc_.integrator_steps()
	    << std::endl;

  std::cout << "Eigen vector array has size ("
	    << _theta_.lambda().size() << ")\n\n"
	    << "Sum( Re(mu_0) ) = " 
	    << _theta_.mu().sum()
	    << std::endl;

  if (!inputs.stdres_file().empty()) {
    std::cout << "Kriging matrix has dimension: ("
	      << _data_.W(0).rows() << ", " << _data_.W(0).cols() << ")"
	      << std::endl;
  }


  std::cout << "(Extended) size of Y's: " << _data_.Y(0).size() << ", "
	    << _data_.Y(1).size()
	    << std::endl;
  
  // dualres::fit_dualres_gaussian_process_model<scalar_type>(_data_, _theta_, _hmc_);
  

  // Finish by calling  nifti_image_free()
  nifti_image_free(_high_res_);
  nifti_image_free(_std_res_);
  // fftw_cleanup_threads();
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
	kernel_params.push_back((scalar_type)kernel_params_dtemp[i]);
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

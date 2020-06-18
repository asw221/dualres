
#include <Eigen/Core>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nifti1_io.h>
#include <sstream>
#include <stdio.h>
#include <stdexcept>
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
#include "dualres/utilities.h"



// return 1 for error; return 0 for success


template< typename T >
bool compute_covar_parameters_if_needed(
  std::vector<T> &theta,
  const nifti_image * const input_data
);

template< typename T >
bool valid_covar_parameters(const std::vector<T> &theta);





int main(int argc, char *argv[]) {
#ifdef DUALRES_SINGLE_PRECISION
  typedef float scalar_type;
#else
  typedef double scalar_type;
#endif
  
  dualres::GPMCommandParser<scalar_type> inputs(argc, argv);
  // Errors if not given at least a "highres" image
  if (!inputs)
    return 1;
  else if (inputs.help_invoked())
    return 0;

  
  dualres::set_seed(inputs.seed());

  dualres::set_number_of_threads(inputs.threads());
  ::omp_set_num_threads(dualres::threads());
  Eigen::setNbThreads(dualres::threads());

  dualres::initialize_temporary_directory();


#ifdef DUALRES_SINGLE_PRECISION
  const int FFTW_STATUS = ::fftwf_init_threads();
#else
  const int FFTW_STATUS = ::fftw_init_threads();
#endif
  if (FFTW_STATUS == 0) {
    std::cerr
      << "FFTW thread initialization had abnormal exit status"
      << std::endl;
    return 1;
  }
  std::cout << "[dualres running on " << dualres::threads()
	    << " cores]" << std::endl;
  

  scalar_type neighborhood = inputs.neighborhood();
  // int n_datasets = 1;
  const bool _STANDARD_RESOLUTION_AVAILABLE =
    !inputs.stdres_file().empty();
  // const int _n_datasets = _STANDARD_RESOLUTION_AVAILABLE ? 2 : 1;

  const std::string _OUTPUT_FILE_SAMPLES = inputs.output_file(
    "_samples.dat");
  const std::string _OUTPUT_FILE_MEAN = inputs.output_file(
    "_posterior_mean.nii");
  const std::string _OUTPUT_FILE_VARIANCE = inputs.output_file(
    "_posterior_variance.nii");
  
  nifti_image* _high_res_;
  nifti_image* _std_res_;
  
  dualres::MultiResData<scalar_type> _data_;
  dualres::HMCParameters<scalar_type> _hmc_;
  
  std::vector<scalar_type> _covar_params = inputs.covariance_parameters();

  bool error_status = false;
  std::ostringstream error_stream;

  
  std::ofstream mcmc_samples_stream(_OUTPUT_FILE_SAMPLES.c_str());
  if (_OUTPUT_FILE_MEAN != inputs.highres_file())
    ::remove(_OUTPUT_FILE_MEAN.c_str());
  if (_OUTPUT_FILE_VARIANCE != inputs.highres_file())
    ::remove(_OUTPUT_FILE_VARIANCE.c_str());
  

  try {
    _high_res_ = nifti_image_read(inputs.highres_file().c_str(), 1);
    std::cout << "High-resolution file: " << inputs.highres_file()
	      << std::endl;
    

    if (!mcmc_samples_stream) {
      error_status = true;
      error_stream << "Error: will not be able to write output to\n  "
		   << _OUTPUT_FILE_MEAN;
      throw std::runtime_error(error_stream.str());
    }
    
    
    if (_STANDARD_RESOLUTION_AVAILABLE) {
      _std_res_ = nifti_image_read(inputs.stdres_file().c_str(), 1);
      std::cout << "Standard-resolution file: " << inputs.stdres_file()
		<< std::endl;
    
      if (!dualres::same_data_types(_high_res_, _std_res_)) {
	// Actually don't think this should result in an error
	error_status = true;
	error_stream << "Error: " << inputs.highres_file() << " and "
		     << inputs.stdres_file()
		     << " have different data types\n";
	throw std::domain_error(error_stream.str());
      }
    
      // If covar parameters are not given, estimate them from inputs. Use
      // Std Res image first if available  
      if (!compute_covar_parameters_if_needed(_covar_params, _std_res_)) {
	error_status = true;
	error_stream << "Error: could not compute covariance parameters from "
		     << inputs.stdres_file() << "\n";
	throw std::runtime_error(error_stream.str());
      }

      

      // If the (mm) neighborhood is not given, compute it from the covar
      // parameters assuming sparsity after a cuttoff level of 0.1
      //  - Only relevant for multi-resolution models
      if (neighborhood <= 0) {
	// neighborhood = dualres::kernels::rbf_inverse(
	//   (scalar_type)0.1, covar_params[1], covar_params[2], covar_params[0]);
	neighborhood = 3 * (scalar_type)dualres::voxel_dimensions(
          dualres::qform_matrix(_high_res_)).array().maxCoeff();
      }
      std::cout << "Neighborhood: " << neighborhood << " (mm)\n";
      
    }  // if (_STANDARD_RESOLUTION_AVAILABLE)

    
    if (!compute_covar_parameters_if_needed(_covar_params, _high_res_)) {
	error_status = true;
	error_stream << "Error: could not compute covariance parameters from "
		     << inputs.highres_file() << "\n";
	throw std::runtime_error(error_stream.str());
    }
    if (!valid_covar_parameters(_covar_params)) {
      error_status = true;
      error_stream << "Error: covariance parameters are not valid (";
      for (int i = 0; i < (int)_covar_params.size(); i++) {
	error_stream << _covar_params[i];
	if (i < ((int)_covar_params.size() - 1))  error_stream << ", ";
      }
      error_stream << ")\n";
      throw std::domain_error(error_stream.str());
    }

    //
    std::cout << "Covariance parameters: (" << _covar_params[0]
	      << ", " << _covar_params[1] << ", " << _covar_params[2] << ")"
	      << std::endl;
    //

    _hmc_ = dualres::HMCParameters<scalar_type>(
      inputs.mcmc_burnin(), inputs.mcmc_nsave(), inputs.mcmc_thin(),
      inputs.mcmc_leapfrog_steps(),
      (scalar_type)(1.0 / inputs.mcmc_leapfrog_steps()),  // starting eps
      inputs.mcmc_mhtarget()
      );
    dualres::set_monitor_simulations(inputs.monitor());
      
    std::cout << "\nMCMC settings:"
	      << "\n-----------------"
	      << "\nBurnin   = " << inputs.mcmc_burnin()
	      << "\nNSave    = " << inputs.mcmc_nsave()
	      << "\nThin     = " << inputs.mcmc_thin()
	      << "\nLeapfrog = " << inputs.mcmc_leapfrog_steps()
	      << "\n-----------------"
	      << std::endl;


    if (!_STANDARD_RESOLUTION_AVAILABLE) {
      _data_ = dualres::MultiResData<scalar_type>(
        _high_res_, _covar_params);
    }
    else {
      _data_ = dualres::MultiResData<scalar_type>(
        _high_res_, _std_res_, _covar_params, neighborhood);

      // Clear Standard Resolution data file after importing to _data_
      ::nifti_image_free(_std_res_);
    }
    // _data_.print_summary();

    

    // Fit model
    dualres::gaussian_process::sor_approx::mcmc_summary<scalar_type>
      model_output = dualres::gaussian_process::sor_approx::fit_model
      <scalar_type>(_high_res_, _data_, _hmc_, mcmc_samples_stream);

    mcmc_samples_stream.close();
    std::cout << _OUTPUT_FILE_SAMPLES << " written\n";

    // Write output
    // Posterior mean:
    ::nifti_set_filenames(_high_res_, _OUTPUT_FILE_MEAN.c_str(), 1, 1);
    if (std::string(_high_res_->fname) != inputs.highres_file()) {
      dualres::emplace_nonzero_data(_high_res_, model_output.mode_mu());
      
      ::nifti_image_write(_high_res_);
      std::cout << _OUTPUT_FILE_MEAN << " written\n";
    }
    else {
      error_stream << "Warning: posterior mean image would overwrite data. "
		   << "File not written\n";
      std::cerr << error_stream.str();
    }
    
    // Posterior variance:
    ::nifti_set_filenames(_high_res_, _OUTPUT_FILE_VARIANCE.c_str(), 1, 1);
    if (std::string(_high_res_->fname) != inputs.highres_file()) {
      dualres::emplace_nonzero_data(_high_res_, model_output.var_mu());
      
      ::nifti_image_write(_high_res_);
      std::cout << _OUTPUT_FILE_VARIANCE << " written\n";
    }
    else {
      error_stream << "Warning: posterior variance image would overwrite data. "
		   << "File not written\n";
      std::cerr << error_stream.str();
    }

    
    std::cout << "Sampling took " << model_output.sampling_time() << " (sec)\n"
	      << "Metropolis-Hastings rate was "
	      << (model_output.metropolis_hastings_rate() * 100)
	      << "%\n" << std::endl;
    
    // For simulations: write MH rate and sampling time to 'stats' file
    mcmc_samples_stream.open(inputs.output_file("_stats.csv"));
    if (mcmc_samples_stream.is_open()) {
      mcmc_samples_stream << "MHrate,SamplingTime,StepSize\n"
			  << model_output.metropolis_hastings_rate()
			  << ","
			  << model_output.sampling_time()
			  << ","
			  << _hmc_.eps_value()
			  << "\n";
      mcmc_samples_stream.close();
    }
    

    // 
    ::nifti_image_free(_high_res_);
  }  // try ...
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



  // Finish by calling  fftw_cleanup_threads()
#ifdef DUALRES_SINGLE_PRECISION
  ::fftwf_cleanup_threads();
#else
  ::fftw_cleanup_threads();
#endif
  if (error_status)
    return 1;
}









template< typename T >
bool compute_covar_parameters_if_needed(
  std::vector<T> &theta,
  const nifti_image * const input_data
) {
  bool success = true;
  if (theta.empty()) {
    std::vector<double> covar_params_dtemp{1, 0.6, 1.5};
    // ^^ provide better starting values, especially for marginal variance

    std::cout << "Estimating covariance parameters... " << std::flush;
    int kp_success;  // 0 - success, 1 - error
    kp_success = dualres::compute_rbf_parameters(
      covar_params_dtemp, dualres::compute_mce_summary_data(input_data));
    if (kp_success == 1) {
      std::cerr << "Could not estimate covar parameters.\n"
		<< "Try re-running with the --covariance argument supplied"
		<< std::endl;
      success = false;
    }
    else {
      for (int i = 0; i < (int)covar_params_dtemp.size(); i++)
	theta.push_back((T)covar_params_dtemp[i]);
      std::cout << "Done!" << std::endl;
    }
  }
  return success;
};


 
template< typename T >
bool valid_covar_parameters(const std::vector<T> &theta) {
  // Error check input covar parameters:
  bool success = true;
  if (theta[0] <= 0) {
    std::cerr << "First covariance parameter (marginal variance) must be > 0\n";
    success = false;
  }
  if (theta[1] <= 0) {
    std::cerr << "Second covariance parameter (bandwidth) must be > 0\n";
    success = false;
  }
  if (theta[2] <= 0 || theta[2] > 2) {
    std::cerr << "Third covariance parameter (exponent) must be in (0, 2]\n";
    success = false;
  }
  return success;
};

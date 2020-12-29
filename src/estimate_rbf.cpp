
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <fstream>
#include <iostream>
#include <nifti1_io.h>
#include <stdexcept>
#include <vector>

#include "dualres/ansi.h"
#include "dualres/CommandParser.h"
#include "dualres/estimate_kernel_parameters.h"
#include "dualres/kernels.h"
#include "dualres/nifti_manipulation.h"


int main(int argc, char *argv[]) {
  dualres::EstimRbfCommandParser inputs(argc, argv);
  if (inputs.error())
    return 1;
  else if (inputs.help_invoked())
    return 0;

  ::nifti_image* __nii;
  ::nifti_image* __mask;
  dualres::mce_data mce;
  // Eigen::Vector3d _grad;
  // Eigen::Matrix3d _Cov_approx;
  // Eigen::Matrix3d Hessian;
  try {
    __nii = dualres::nifti_image_read(inputs.image_file(), 1);
    if (!inputs.mask_file().empty()) {
      __mask = dualres::nifti_image_read(inputs.mask_file(), 1);
      if (!dualres::same_orientation(__nii, __mask)) {
	throw std::domain_error("Error: Image/mask orientation mismatch");
      }
      else {
	dualres::apply_mask(__nii, __mask);
	::nifti_image_free(__mask);
      }
    }
    std::cout << "Computing covariances across the image... " << std::flush;
  
    // Extract covariance/distance summary data
    mce = dualres::compute_mce_summary_data(__nii);
    std::cout << "Done!" << std::endl;

    // Cleanup
    ::nifti_image_free(__nii);
  }
  catch (const std::exception &__err) {
    std::cerr << ansi::foreground_bold_magenta
	      << "Exception caught with message:\n"
	      << ansi::reset << "'"
	      << __err.what() << "'\n"
	      << ansi::reset << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Error computing minimum contrast data from file:\n\t"
	      << inputs.image_file() << std::endl;
    return 1;
  }

  
  // Write output csv file, if requested
  if (!inputs.output_file().empty()) {
    std::ofstream csv(inputs.output_file());
    if (csv.is_open()) {
      try {
	csv << "Radial_Distance,Covariance,N_Pairs\n";
	for (int i = 0; i < mce.distance.size(); i++) {
	  csv << mce.distance[i] << ","
	      << mce.covariance[i] << ","
	      << mce.npairs[i];
	  if (i != (mce.distance.size() - 1))  csv << "\n";
	}
	csv.close();
      }
      catch (...) {
	std::cerr << "\nWarning: unable to write to '" << inputs.output_file()
		  << "'. Data will not be saved";
	csv.close();
	std::remove(inputs.output_file().c_str());
      }
    }
    else {
      std::cerr << "\nWarning: unable to open '" << inputs.output_file()
		<< "' for writing. Data will not be saved";
    }
  }
  
  
  
  // Estimate RBF parameters
  // Starting point can (should?) be made more robust by running an approximate
  // Global optimization first 
  // std::vector<double> theta{mce.covariance[0] * 0.8, 0.6, 1.5};
  std::vector<double> theta{mce.covariance[0] * 0.8, 0.6, 1.5};
  // std::vector<double> theta_copy(theta.size());
  // double dtheta, tmp;
  // const int maxiter = 30;
  
  std::cout << "Estimating smoothness (radial basis function approximation)... "
	    << std::flush;
  try {
    // First pass to get good but rough estimate
    dualres::compute_rbf_parameters(theta, mce,
      inputs.use_constraint(), 1e6,
      inputs.parameter(0), inputs.parameter(1), inputs.parameter(2),
      inputs.xtol_rel()
    );
    // std::cout << "\n(" << theta[0] << ", " << theta[1] << ", " << theta[2] << ")\n";
    // Second pass removing potentially noisy tail data
    // for (int i = 0; i < maxiter; i++) {
    //   dtheta = 0;
    //   theta_copy[0] = theta[0] ; theta_copy[1] = theta[1] ; theta_copy[2] = theta[2];
    //   if (inputs.use_constraint() && theta[2] <= theta[1]) {
    // 	theta[1] -= 1e-6;
    //   }
    //   dualres::compute_rbf_parameters(theta, mce,
    //     inputs.use_constraint(),
    //     dualres::kernels::rbf_inverse(0.05, theta[1], theta[2]),
    //     inputs.parameter(0), inputs.parameter(1), inputs.parameter(2),
    //     inputs.xtol_rel()
    //   );
    //   for (int i = 0; i < theta.size(); i++) {
    // 	tmp = theta[i] - theta_copy[i];
    // 	dtheta += tmp * tmp;
    //   }
    //   std::cout << "(" << theta[0] << ", " << theta[1] << ", " << theta[2]
    // 		<< ");  dt = " << dtheta << "\n";
    // }
    // std::cout << std::endl;
    //
    // _grad = dualres::_rbf_lsq_gradient(theta, mce);
    // _Cov_approx = (_grad * _grad.transpose()).completeOrthogonalDecomposition()
    //   .pseudoInverse();
    // _Cov_approx *= dualres::_rbf_mse(theta, mce);
    // for (int i = 0; i < theta.size(); i++) {
    //   if (inputs.parameter_fixed(i)) {
    // 	// _grad[i] = 0;
    // 	_Cov_approx.row(i) *= 0;
    // 	_Cov_approx.col(i) *= 0;
    //   }
    // }
    // Hessian = dualres::_rbf_lsq_hessian(theta, mce);
  }
  catch (const std::exception &__err) {
    std::cerr << "\n"
	      << ansi::foreground_bold_magenta
	      << "Exception caught with message:\n"
	      << ansi::reset << "'"
	      << __err.what() << "'\n"
	      << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "\n"
	      << ansi::foreground_bold_magenta
	      << "Error: unable to estimate covariance parameters "
	      << ansi::reset
	      << "(unknown cause)"
	      << std::endl;
    return 1;
  }
  std::cout << "Done!" << std::endl;
  std::cout << "  Marg. Var. = " << theta[0] << "\n"
	    << "  Bandwidth  = " << theta[1] << "\n"
	    << "  Exponent   = " << theta[2] << "\n"
	    << "  <FWHM      = "
	    << dualres::kernels::rbf_bandwidth_to_fwhm(theta[1], theta[2])
	    << " (mm)>\n"
    // << "Approximate Covariance:\n" << _Cov_approx << "\n"
	    << std::endl;
  
};

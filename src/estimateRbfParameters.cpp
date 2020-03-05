
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nifti1_io.h>
#include <vector>

#include "CommandParser.h"
#include "estimate_kernel_parameters.h"


int main(int argc, char *argv[]) {
  dualres::EstimRbfCommandParser inputs(argc, argv);
  if (inputs.error())
    return 1;
  else if (inputs.help_invoked())
    return 0;
  
  nifti_image* img = nifti_image_read(inputs.image_file().c_str(), 1);
  std::cout << "Computing covariances across the image... " << std::flush;
  
  // Extract covariance/distance summary data
  dualres::mce_data summ = dualres::compute_mce_summary_data(img);
  std::cout << "Done!" << std::endl;

  
  // Write output csv file, if requested
  if (!inputs.output_file().empty()) {
    std::ofstream csv(inputs.output_file());
    if (csv.is_open()) {
      try {
	csv << "Radial_Distance,Covariance,N_Pairs\n";
	for (int i = 0; i < summ.distance.size(); i++) {
	  csv << summ.distance[i] << ","
	      << summ.covariance[i] << ","
	      << summ.npairs[i];
	  if (i != (summ.distance.size() - 1))  csv << "\n";
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
  std::vector<double> theta{summ.covariance[0] * 0.8, 0.6, 1.5};
  
  std::cout << "Estimating smoothness (radial basis function approximation)... "
	    << std::flush;
  dualres::compute_rbf_parameters(theta, summ);
  std::cout << "Done!" << std::endl;
  std::cout << "  Marg. Var. = " << theta[0] << "\n"
	    << "  Bandwidth  = " << theta[1] << "\n"
	    << "  Exponent   = " << theta[2]
	    << std::endl;
  
  nifti_image_free(img);
};


#include <iostream>
#include <math.h>

#include "dualres/CommandParser.h"
#include "dualres/kernels.h"



int main(int argc, char *argv[]) {
  dualres::NeighborhoodCommandParser inputs(argc, argv);
  if (inputs.error())
    return 1;
  else if (inputs.help_invoked())
    return 0;

  
  const double neighborhood = dualres::kernels::rbf_inverse(
    inputs.rho(), inputs.parameter(1), inputs.parameter(2),
    inputs.parameter(0)
  );
  
  std::cout << neighborhood << std::endl;
  
  if (isnan(neighborhood)) {
    std::cerr << "\nWarning: make sure that kernel parameters fall within ranges:"
	      << "\n  variance > 0"
	      << "\n  bandwidth > 0"
	      << "\n  0 < exponent <= 2"
	      << std::endl;
  }
};


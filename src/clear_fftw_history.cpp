
#include <iostream>
#include <stdio.h>

#include "dualres/utilities.h"


int main () {
  if (remove(dualres::fftw_wisdom_file().c_str()) != 0) {
    std::cerr << "Wisdom has not been forgotten. "
	      << "Or was never remembered in the first place."
	      << std::endl;
  }
}

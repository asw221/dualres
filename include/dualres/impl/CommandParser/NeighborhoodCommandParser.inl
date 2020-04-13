
#include <iostream>
#include <string>
#include <vector>



template< typename T >
void dualres::NeighborhoodCommandParser<T>::show_usage() const {
  std::cerr << "\nUsage:\n"
	    << "\trbf_neighbhorood rho --covariance f1 f2 f3";
};



template< typename T >
dualres::NeighborhoodCommandParser<T>::NeighborhoodCommandParser(int argc, char **argv) {
  // argv[0] can be used to change kernel types at some point
  std::ifstream ifs;
  _status = call_status::success;

  const int K = 3;  // number of kernel parameters
  _kernel_params.resize(K);
  
  if (argc != (K + 3)) {
    show_usage();
    _status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if ((arg == "-h") || (arg == "--help")) {
	_status = call_status::help;
      }
      else if (arg == "--theta" || arg == "--covariance") {
	if ((i + K) <= argc) {
	  try {
	    for (int j = 0; j < K; j++) {
	      i++;
	      _kernel_params[j] = (scalar_type)std::stod(argv[i]);
	    }
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --covariance option requires"
		      << K << " numeric arguments\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --covariance option requires 3 numeric arguments\n";
	  _status = call_status::error;
	}
      }
      else {
	try {
	  _rho = (scalar_type)std::stod(argv[i]);
	}
	catch (...) {
	  std::cerr << "\nUnrecognized option: " << arg;
	  _status = call_status::error;
	}
      }
      if (error() || help_invoked()) {
	show_usage();
	break;
      }
    }  // for (int i = 1; i < argc ...
  }  // if (argc < 2) ... else ...
};




template< typename T >
bool dualres::NeighborhoodCommandParser<T>::error() const {
  return _status == call_status::error;
};

template< typename T >
bool dualres::NeighborhoodCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};

template< typename T >
dualres::NeighborhoodCommandParser<T>::operator bool() const {
  return !error();
};

template< typename T >
bool dualres::NeighborhoodCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
typename dualres::NeighborhoodCommandParser<T>::scalar_type
dualres::NeighborhoodCommandParser<T>::parameter(const int which) const {
  // May have to adapt if ever add different kernels
  if (which < 0 || which >= 3) {
    std::cerr << "Warning: parameter index must be between [0, 2]\n";
    return -1;
  }
  return _kernel_params[which];
};



template< typename T >
typename dualres::NeighborhoodCommandParser<T>::scalar_type
dualres::NeighborhoodCommandParser<T>::rho() const {
  return _rho;
};



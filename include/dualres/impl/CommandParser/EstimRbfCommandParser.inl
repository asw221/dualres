
#include <fstream>
#include <iostream>
#include <string>
#include <vector>



template< typename T >
void dualres::EstimRbfCommandParser<T>::show_usage() const {
  std::cerr << "\nUsage:\n"
	    << "\testimate_rbf path/to/img <options>\n\n";
};


template< typename T >
void dualres::EstimRbfCommandParser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "\t--bandwidth f1  fixes correlation bandwidth parameter to f1\n"
	    << "\t--constraint    imposes a bandwidth <= exponent constraint\n"
	    << "\t--exponent  f1  fixes correlation exponent parameter to f1\n"
	    << "\t--mask path/to/mask_img  ~~NOT IMPLEMENTED~~\n"
	    << "\t--output ofile/basename  output file for MCE summary data\n"
	    << "\t--variance  f1  fixes the marginal variance parameter to f1\n"
	    << "\t--xtol      f1  set the numerical tolerance (default 1e-5)\n"
	    << "\n"
	    << "Computes minimum contrast estimation (MCE) data from the input NIfTI\n"
	    << "image file. This data summarizes empirical covariances between pairs\n"
	    << "of voxels at different distances, and will be written in *.csv format\n"
	    << "to the --output (-o) file if requested.\n"
	    << "MCE data is then used to estimate the parameters of a radial basis\n"
	    << "covariance function,\n"
	    << "\t\tC(d; v, b, e) = v * exp(-b |d|^e) >= 0,\n"
	    << "using non-linear least squares (COBYLA algorithm implemented by\n"
	    << "the NLopt library).\n"
	    << "\n"
	    << "Covariance parameters may be 'fixed' (within a small tolerance)\n"
	    << "prior to estimation by specifying their value. The arguments\n"
	    << "--variance (or -var), --bandwidth (or -bw), and --exponent (or -exp)\n"
	    << "are meant for exactly this purpose. In general, the variance and\n"
	    << "bandwidth should be > 0, and the exponent should be within (0, 2).\n"
	    << "We do not typically recommend fixing the non-exponent parameters.\n"
	    << "\n"
	    << "If the --constraint flag is given, the correlation parameters will be\n"
	    << "estimated under the constraint that the bandwidth <= exponent. This\n"
	    << "can sometimes help improve numerical stability when estimating the\n"
	    << "eigen functions of the radial basis.\n"
	    << "\n"
	    << "~~PLACEHOLDER: MASK~~"
	    << "\n\n";
};






template< typename T >
dualres::EstimRbfCommandParser<T>::EstimRbfCommandParser(int argc, char **argv) {
  std::ifstream ifs;
  _status = call_status::success;
  _use_constraint = false;
  _xtol = 1e-5;

  _kernel_params.resize(3);
  for (int i = 0; i < 3; i++)
    _kernel_params[i] = -1;
  if (argc < 2) {
    show_usage();
    _status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      
      if ((arg == "-h") || (arg == "--help")) {
	_status = call_status::help;
      }
      else if (arg == "--constrained" || arg == "--constraint") {
	_use_constraint = true;
      }
      else if (arg == "--variance" || arg == "-var") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _kernel_params[0] = std::stod(argv[i]);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --variance option requires 1 numeric argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --variance option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--bandwidth" || arg == "-bw") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _kernel_params[1] = std::stod(argv[i]);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --bandwidth option requires 1 numeric argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --bandwidth option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--exponent" || arg == "-exp") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _kernel_params[2] = std::stod(argv[i]);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --exponent option requires 1 numeric argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --exponent option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--output" || arg == "-o") {
	if (i + 1 < argc) {
	  i++;
	  _output_file = argv[i];
	}
	else {
	  std::cerr << "\nWarning: --output option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--xtol") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _xtol = std::stod(argv[i]);
	    if (_xtol <= 0) {
	      std::cerr << "\nError: --xtol should be > 0\n";
	      _status = call_status::error;
	    }
	    if (_xtol >= 0.01) {
	      std::cerr << "\nWarning: numeric tolerance is high (default is 1e-5)\n";
	    }
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --xtol option requires 1 numeric argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --xtol option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else {
	_image_file = argv[i];
	ifs.open(_image_file, std::ifstream::in);
	if (!ifs.is_open()) {
	  std::cerr << "\nUnrecognized argument " << _image_file << "\n";
	  _status = call_status::error;
	}
	else {
	  ifs.close();
	}
      }
      if (error() || help_invoked()) {
	break;
      }
    }  // for (int i = 1; i < argc ...
  }  // if (argc < 2) ... else ...
  if (help_invoked()) {
    show_help();
  }
  if (error()) {
    show_usage();
    std::cerr << "\nSee estimate_rbf --help for more information\n";
  }
};



template< typename T >
bool dualres::EstimRbfCommandParser<T>::use_constraint() const {
  return _use_constraint;
};

template< typename T >
bool dualres::EstimRbfCommandParser<T>::error() const {
  return _status == call_status::error;
};

template< typename T >
bool dualres::EstimRbfCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};

template< typename T >
dualres::EstimRbfCommandParser<T>::operator bool() const {
  return !error();
};

template< typename T >
bool dualres::EstimRbfCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
double dualres::EstimRbfCommandParser<T>::xtol_rel() const {
  return _xtol;
};


template< typename T >
std::string dualres::EstimRbfCommandParser<T>::image_file() const {
  return _image_file;
};

template< typename T >
std::string dualres::EstimRbfCommandParser<T>::output_file() const {
  return _output_file;
};

template< typename T >
typename dualres::EstimRbfCommandParser<T>::scalar_type
dualres::EstimRbfCommandParser<T>::parameter(const int which) const {
  if (which < 0 || which >= 3) {
    std::cerr << "Warning: parameter index must be between [0, 2]\n";
    return -1;
  }
  return _kernel_params[which];
};

template< typename T >
bool dualres::EstimRbfCommandParser<T>::parameter_fixed(const int which) const {
  if (which < 0 || which >= 3) {
    std::cerr << "Warning: parameter index must be between [0, 2]\n";
    return false;
  }
  return _kernel_params[which] != -1;
};



template< typename T >
typename std::vector<typename dualres::EstimRbfCommandParser<T>::scalar_type>::const_iterator
dualres::EstimRbfCommandParser<T>::kernel_cbegin() const {
  return _kernel_params.cbegin();
};


template< typename T >
typename std::vector<typename dualres::EstimRbfCommandParser<T>::scalar_type>::const_iterator
dualres::EstimRbfCommandParser<T>::kernel_cend() const {
  return _kernel_params.cend();
};


#include <fstream>
#include <iostream>
#include <string>
#include <vector>



template< typename T >
void dualres::EstimRbfCommandParser<T>::show_usage() const {
  std::cout << "Placeholder help for estimateRbfParameters\n";
};






template< typename T >
dualres::EstimRbfCommandParser<T>::EstimRbfCommandParser(int argc, char **argv) {
  std::ifstream ifs;
  _status = call_status::success;

  _kernel_params.resize(3);
  for (int i = 0; i < 3; i++)
    _kernel_params[i] = 0;
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
	show_usage();
	break;
      }
    }  // for (int i = 1; i < argc ...
  }  // if (argc < 2) ... else ...
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
  return _kernel_params[which] != 0;
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

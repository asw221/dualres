
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>



template< typename T >
void dualres::GPMCommandParser<T>::show_usage() const {
  std::cerr << "\nUsage:\n"
	    << "\tdualresGP <options>\n\n"
	    << "Options:\n"
	    << "\t--highres path/to/img1\n"
	    << "\t--kernel  f1 f2 f3\n"
	    << "\t--neighborhood f1"
	    << "\t--stdres  path/to/img2\n"
	    << "\nWhere img[1-2] are valid NIfTI files and f[1-3] are parameters "
	    << "of an exponential radial kernel function.\n\n"
	    << "See dualresGP --help for more information.\n";
};



template< typename T >
dualres::GPMCommandParser<T>::GPMCommandParser(int argc, char* argv[]) {
  // using typename dualres::CommandParser<T>;
  // typedef typename dualres::CommandParser<T>::call_status call_status;
  // using typename dualres::CommandParser<T>::call_status;
  // dualres::CommandParser<T>::CommandParser();
  
  std::ifstream ifs;
  _status = call_status::success;

  const int K = 3;  // number of kernel parameters
  const auto time = std::chrono::high_resolution_clock::now().time_since_epoch();

  // Default values --------------------------------------------------
  _neighborhood = -1;
  _mcmc_burnin = 1000;
  _mcmc_leapfrog_steps = 10;
  _mcmc_nsave = 1000;
  _mcmc_thin = 1;
  _seed = static_cast<unsigned int>(
    std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
  _seed = std::max(_seed, (unsigned)1);
  _threads = (unsigned)0;
  
  // _kernel_params.resize(3);
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
      else if (arg == "--highres") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _highres_file = argv[i];
	  ifs.open(_highres_file, std::ifstream::in);
	  if (!ifs.is_open()) {  // verify file exists
	    std::cerr << "\nWarning: Could not find high-resolution file: "
		      << _highres_file << std::endl;
	    _status = call_status::error;
	  }
	  else {
	    ifs.close();
	  }
	}
	else {
	  std::cerr << "\nWarning: --highres option requires one argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--stdres") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _stdres_file = argv[i];
	  ifs.open(_stdres_file, std::ifstream::in);
	  if (!ifs.is_open()) {  // verify file exists
	    std::cerr << "\nWarning: Could not find standard-resolution file: "
		      << _stdres_file << std::endl;
	    _status = call_status::error;
	  }
	  else {
	    ifs.close();
	  }
	}
	else {
	  std::cerr << "\nWarning: --stdres option requires one argument\n";
	  _status = call_status::error;
	}
      }
      else if ((arg == "--kernel") || (arg == "--theta")) {
	if (i + K < argc) {
	  _kernel_params.resize(K);
	  for (int j = 0; j < _kernel_params.size(); j++) {
	    i++;
	    try {
	      _kernel_params[j] = (scalar_type)std::stod(argv[i]);
	    }
	    catch (...) {
	      std::cerr << "\nWarning: --kernel arguments must be numeric\n";
	      _status = call_status::error;
	      break;
	    }
	  }
	}
	else {
	  std::cerr << "\nWarning: --kernel option requires 3 numeric arguments\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--neighborhood") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _neighborhood = std::abs((scalar_type)std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --neighborhood option requires 1 numeric argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --neighborhood option requires 1 numeric argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--burnin") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_burnin = (unsigned)std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --burnin option requires 1 positive integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --burnin option requires 1 positive integer argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--leapfrog" || arg == "-L") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_leapfrog_steps = (unsigned)std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --leapfrog option requires 1 positive integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --leapfrog option requires 1 positive integer argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--nsave" || arg == "-n") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_nsave = (unsigned)std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --nsave option requires 1 positive integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --nsave option requires 1 positive integer argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--thin") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_nsave = (unsigned)std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --thin option requires 1 positive integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --thin option requires 1 positive integer argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--seed") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _seed = (unsigned)std::abs(std::stoi(argv[i]));
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --seed option requires 1 integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --seed option requires 1 integer argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--threads") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _threads = (unsigned)std::abs(std::stoi(argv[i]));
	  }
	  catch (...) {
	    std::cerr << "\nWarning: --threads option requires 1 integer argument\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --threads option requires 1 integer argument\n";
	  _status = call_status::error;
	}
      }
      else {
	std::cerr<< "\nWarning: Unrecognized option: " << arg << std::endl;
	_status = call_status::error;
      }
      
      if (error() || help_invoked()) {
	show_usage();
	break;
      }
    }  // for (int i = 1; i < argc; i++)
  }
  if (_highres_file.empty()) {
    std::cerr << "\nError: User must supply the --highres argumentn\n";
    _status = call_status::error;
  }
};







template< typename T >
bool dualres::GPMCommandParser<T>::error() const {
  return _status == call_status::error;
};

template< typename T >
bool dualres::GPMCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};

template< typename T >
dualres::GPMCommandParser<T>::operator bool() const {
  return !error();
};

template< typename T >
bool dualres::GPMCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
typename dualres::GPMCommandParser<T>::scalar_type
dualres::GPMCommandParser<T>::neighborhood() const {
  return _neighborhood;
};

template< typename T >
std::string dualres::GPMCommandParser<T>::highres_file() const {
  return _highres_file;
};

template< typename T >
std::string dualres::GPMCommandParser<T>::stdres_file() const {
  return _stdres_file;
};



template< typename T >
unsigned int dualres::GPMCommandParser<T>::mcmc_burnin() const {
  return _mcmc_burnin;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::mcmc_leapfrog_steps() const {
  return _mcmc_leapfrog_steps;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::mcmc_nsave() const {
  return _mcmc_nsave;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::mcmc_thin() const {
  return _mcmc_thin;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::seed() const {
  return _seed;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::threads() const {
  return _threads;
};



template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>
dualres::GPMCommandParser<T>::kernel_parameters() const {
  std::vector<typename dualres::GPMCommandParser<T>::scalar_type> theta(_kernel_params);
  return theta;
};

template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>::iterator
dualres::GPMCommandParser<T>::kernel_begin() {
  return _kernel_params.begin();
};

template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>::iterator
dualres::GPMCommandParser<T>::kernel_end() {
  return _kernel_params.end();
};



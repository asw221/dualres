
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "dualres/nifti_manipulation.h"



template< typename T >
void dualres::GPMCommandParser<T>::show_usage() const {
  std::cerr << "Fit dual (or single) resolution Gaussian process model to NIfTI data:\n"
	    << "Usage:\n"
	    << "\tdualgpm --highres path/to/img1 <options>\n\n";
};


template< typename T >
void dualres::GPMCommandParser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "\t--burnin       int        number of MCMC burnin iterations\n"
	    << "\t--covariance   f1 f2 f3   RBF covariance parameters\n"
	    << "\t--debug                   run a short debug-length MCMC chain\n"
	    << "\t--leapfrog     int        number of MCMC integrator steps\n"
	    << "\t--mhtarget     float      target metropolis hastings rate\n"
	    << "\t--monitor                 monitor MCMC iterations (debugging)\n"
	    << "\t--neighborhood float      n'hood size (mm) for kriging approx\n"
	    << "\t--nsave        int        MCMC samples to save in output\n"
	    << "\t--output   file/basename  path prefix for output files\n"
	    << "\t--seed         int        RNG seed\n"
	    << "\t--stdres   path/to/img2   \n"
	    << "\t--theta                   alias for --covariance\n"
	    << "\t--thin         int        thinning factor for MCMC samples\n"
	    << "\t--threads      int        number of parallel threads\n"
	    << "\n"
	    << "img[1-2] are valid NIfTI files and f[1-3] are parameters "
	    << "of an exponential radial covariance function."
	    << "\n\n";
};



template< typename T >
dualres::GPMCommandParser<T>::GPMCommandParser(int argc, char* argv[]) {
  // using typename dualres::CommandParser<T>;
  // typedef typename dualres::CommandParser<T>::call_status call_status;
  // using typename dualres::CommandParser<T>::call_status;
  // dualres::CommandParser<T>::CommandParser();

  std::stringstream ss;
  _status = call_status::success;

  const int K = 3;  // number of covariance parameters
  const auto time = std::chrono::high_resolution_clock::now().time_since_epoch();

  // Default values --------------------------------------------------
  _neighborhood = -1;
  _mcmc_burnin = 1000;
  _mcmc_leapfrog_steps = 10;
  _mcmc_nsave = 1000;
  _mcmc_thin = 1;
  _mcmc_mhtarget = 0.65;
  _monitor = false;
  _seed = static_cast<unsigned int>(
    std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
  _seed = std::max(_seed, (unsigned)1);
  _threads = (unsigned)0;

  ss << "dualres_mcmc_" << _seed << "_";
  _output_base = ss.str();


  const std::string _MESSAGE_IMPROPER_BURNIN =
    "\nWarning: --burnin option requires 1 positive integer argument\n";
  const std::string _MESSAGE_IMPROPER_COVARIANCE =
    "\nWarning: --covariance option requires 3 numeric arguments\n";
  const std::string _MESSAGE_IMPROPER_LEAPFROG =
    "\nWarning: --leapfrog option requires 1 positive integer argument\n";
  const std::string _MESSAGE_IMPROPER_MHTARGET =
    "\nWarning: --mhtarget option requires 1 numeric argument on (0, 1)\n";
  const std::string _MESSAGE_IMPROPER_NEIGHBORHOOD =
    "\nWarning: --neighborhood option requires 1 numeric argument\n";
  const std::string _MESSAGE_IMPROPER_NSAVE =
    "\nWarning: --nsave option requires 1 positive integer argument\n";
  const std::string _MESSAGE_IMPROPER_SEED =
    "\nWarning: --seed option requires 1 integer argument\n";
  const std::string _MESSAGE_IMPROPER_THIN =
    "\nWarning: --thin option requires 1 positive integer argument\n";
  const std::string _MESSAGE_IMPROPER_THREADS =
    "\nWarning: --threads option requires 1 integer argument\n";
  const std::string _MESSAGE_NOT_NIFTI_FILE =
    "\nWarning: input file does not appear to conform to the nifti-1 standard\n";
  const std::string _MESSAGE_UNRECOGNIZED_OPTION =
    "\nWarning: Unrecognized option: ";  // fill-in-blank
    
  
  // _covariance_params.resize(3);
  if (argc < 2) {
    _status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if ((arg == "-h") || (arg == "--help")) {
	_status = call_status::help;
	break;
      }
      else if (arg == "--burnin") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_burnin = std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_BURNIN;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_BURNIN;
	  _status = call_status::error;
	}
      }
      else if ((arg == "--covariance") || (arg == "--theta")) {
	if (i + K < argc) {
	  _covariance_params.resize(K);
	  for (int j = 0; j < (int)_covariance_params.size(); j++) {
	    i++;
	    try {
	      _covariance_params[j] = (scalar_type)std::stod(argv[i]);
	    }
	    catch (...) {
	      std::cerr << _MESSAGE_IMPROPER_COVARIANCE;
	      _status = call_status::error;
	      break;
	    }
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_COVARIANCE;
	  _status = call_status::error;
	}
      }
      else if (arg == "--debug") {
	_mcmc_burnin = 50;
	_mcmc_nsave = 10;
	_mcmc_thin = 1;
	_monitor = true;
      }
      else if (arg == "--highres") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _highres_file = argv[i];
	  if (!dualres::is_nifti_file(_highres_file)) {
	    std::cerr << _MESSAGE_NOT_NIFTI_FILE << "\t" << _highres_file
		      << std::endl;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --highres option requires one argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--leapfrog" || arg == "-L") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_leapfrog_steps = std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_LEAPFROG;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_LEAPFROG;
	  _status = call_status::error;
	}
      }
      else if (arg == "--mhtarget") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_mhtarget = std::stod(argv[i]);
	    if (_mcmc_mhtarget <= 0 || _mcmc_mhtarget >= 1)
	      throw std::domain_error("--mhtarget outside bounds");
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_MHTARGET;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_MHTARGET;
	  _status = call_status::error;
	}
      }
      else if (arg == "--monitor") {
	_monitor = true;
      }
      else if (arg == "--neighborhood") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _neighborhood = std::abs((scalar_type)std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_NEIGHBORHOOD;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_NEIGHBORHOOD;
	  _status = call_status::error;
	}
      }
      else if (arg == "--nsave" || arg == "-n") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_nsave = std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_NSAVE;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_NSAVE;
	  _status = call_status::error;
	}
      }
      else if (arg == "--output" || arg == "-o") {
	if (i + 1 < argc) {
	  i++;
	  _output_base = std::string(argv[i]);
	}
	else {
	  std::cerr << "\nWarning: --output option requires 1 string argument\n";
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
	    std::cerr << _MESSAGE_IMPROPER_SEED;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_SEED;
	  _status = call_status::error;
	}
      }
      else if (arg == "--stdres") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _stdres_file = argv[i];
	  if (!dualres::is_nifti_file(_stdres_file)) {
	    std::cerr << _MESSAGE_NOT_NIFTI_FILE << "\t" << _stdres_file
		      << std::endl;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --stdres option requires one argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--thin") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _mcmc_thin = std::max(std::abs(std::stoi(argv[i])), 1);
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_THIN;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_THIN;
	  _status = call_status::error;
	}
      }
      else if (arg == "--threads") {
	if (i + 1 < argc) {
	  i++;
	  try {
	    _threads = std::abs(std::stoi(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_IMPROPER_THREADS;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_THREADS;
	  _status = call_status::error;
	}
      }
      else {
	std::cerr<< _MESSAGE_UNRECOGNIZED_OPTION << arg << std::endl;
	_status = call_status::error;
      }
      
      if (error() || help_invoked()) {
	break;
      }
    }  // for (int i = 1; i < argc; i++)
  }
  if (help_invoked()) {
    show_help();
  }
  else if (_highres_file.empty()) {
    std::cerr << "\nError: User must supply the --highres argument\n";
    _status = call_status::error;
  }
  if (error()) {
    show_usage();
    std::cerr << "\nSee dualgpm --help for more information\n";
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
bool dualres::GPMCommandParser<T>::monitor() const {
  return _monitor;
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
dualres::GPMCommandParser<T>::mcmc_mhtarget() const {
  return _mcmc_mhtarget;
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
std::string dualres::GPMCommandParser<T>::output_file_base() const {
  return _output_base;
};

template< typename T >
std::string dualres::GPMCommandParser<T>::output_file(const std::string &extension) const {
  return (_output_base + extension);
};

template< typename T >
std::string dualres::GPMCommandParser<T>::stdres_file() const {
  return _stdres_file;
};



template< typename T >
int dualres::GPMCommandParser<T>::mcmc_burnin() const {
  return _mcmc_burnin;
};


template< typename T >
int dualres::GPMCommandParser<T>::mcmc_leapfrog_steps() const {
  return _mcmc_leapfrog_steps;
};


template< typename T >
int dualres::GPMCommandParser<T>::mcmc_nsave() const {
  return _mcmc_nsave;
};


template< typename T >
int dualres::GPMCommandParser<T>::mcmc_thin() const {
  return _mcmc_thin;
};


template< typename T >
int dualres::GPMCommandParser<T>::threads() const {
  return _threads;
};


template< typename T >
unsigned int dualres::GPMCommandParser<T>::seed() const {
  return _seed;
};



template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>
dualres::GPMCommandParser<T>::covariance_parameters() const {
  std::vector<typename dualres::GPMCommandParser<T>::scalar_type> theta(_covariance_params);
  return theta;
};

template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>::iterator
dualres::GPMCommandParser<T>::covariance_begin() {
  return _covariance_params.begin();
};

template< typename T >
typename std::vector<typename dualres::GPMCommandParser<T>::scalar_type>::iterator
dualres::GPMCommandParser<T>::covariance_end() {
  return _covariance_params.end();
};





#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "dualres/defines.h"
#include "dualres/nifti_manipulation.h"



template< typename T >
void dualres::GPMCommandParser<T>::show_usage() const {
  std::cerr << "\nFit dual (or single) resolution Gaussian process model to NIfTI data:\n"
	    << "\nUsage:\n"
	    << "\t" << _caller << " --highres path/to/img1 <options>\n\n";
};


template< typename T >
void dualres::GPMCommandParser<T>::show_help() const {
  std::string user_response;
  show_usage();
  std::cerr << "REQUIRED:\n"
	    << "\t--highres  path/to/img1   Primary data source\n"
	    << "\n"
	    << "Optional:\n"
	    << "\t--stdres   path/to/img2   Optional secondary data file\n"
	    << "\t--burnin       int        Number of MCMC burnin iterations\n"
	    << "\t--covariance   f1 f2 f3   RBF covariance parameters\n"
	    << "\t--debug                   Run a short debug-length MCMC chain\n"
	    << "\t--leapfrog     int        Number of MCMC integrator steps\n"
	    << "\t--mhtarget     float      Target metropolis hastings rate\n"
	    << "\t--monitor                 Monitor MCMC iterations (debugging)\n"
	    << "\t--hmask    path/to/mask   Mask image for --highres input\n"
	    << "\t--neighborhood float      N'hood size (mm) for kriging approx\n"
	    << "\t--nsave        int        MCMC samples to save in output\n"
	    << "\t--omask    path/to/mask   Mask image to define output space\n"
	    << "\t--output   file/basename  Path prefix for output files\n"
	    << "\t--samples                 Request full MCMC output\n"
	    << "\t--smask    path/to/mask   Mask image for --stdres input\n"
	    << "\t--seed         int        RNG seed\n"
	    << "\t--theta        f1 f2 f3   Alias for --covariance\n"
	    << "\t--thin         int        Thinning factor for MCMC samples\n"
	    << "\t--threads      int        Number of parallel threads\n"
	    << "\n"
	    << "----------------------------------------------------------------------\n"
	    << "\n"
	    << "All arguments have default values except for image inputs. \n"
	    << "\n"
	    << "Denoises the input --highres image, using additional --stdres data if \n"
	    << "available. Minimum output will include:"
	    << "\n"
	    << "  - *_posterior_activation.nii.gz \n"
	    << "    file mapping [0, 1] estimates of the posterior probability of \n"
	    << "    functional (de)activation \n"
	    << "\n"
	    << "  - *_posterior_mean.nii.gz \n"
	    << "    file containing the estimated denoised --highres image \n"
	    << "\n"
	    << "  - *_posterior_variance.nii.gz and *_residual.nii.gz \n"
	    << "    files mapping the uncertainty in the posterior mean, and the \n"
	    << "    estimated noise, respectively. \n"
	    << "\n"
	    << "Shorthand \n"
	    << "--------- \n"
	    << "img[1-2] should be files in the NIfTI standard and f[1-3] \n"
	    << "denote floating point parameters. \n"
	    << "\n"
	    << "[h,s,o]mask are used to define masks for the high/base resolution \n"
	    << "image file, the standard/secondary resolution image file, and an \n"
	    << "output ROI, respectively. All will default to implicit image masks if \n"
	    << "not specified. If you choose to use image masks, they must be in the \n"
	    << "same orientation as their corresponding data image. This will be \n"
	    << "checked by comparing the qform matrix in the image/mask header files. \n"
	    << std::endl;

  std::cerr << "<Press Enter/Return for more or {any}+Enter to terminate> ";
  std::cin.clear();
  std::getline( std::cin, user_response );
  if ( user_response.empty() ) {
    std::cerr << "\n";
    show_details();
 
    std::cerr << "<Press Enter/Return for more or {any}+Enter to terminate> ";
    std::cin.clear();
    std::getline( std::cin, user_response );
    if ( user_response.empty() ) {
      std::cerr << "\n";
      show_mcmc_control();
    }
  }

  
  std::cerr << "----------------------------------------------------------------------\n"
	    << "\n"
	    << std::endl;
};




template< typename T >
void dualres::GPMCommandParser<T>::show_details() const {
  std::cerr << "\nOther Details \n"
	    << "------------- \n"
	    << "--covariance \n"
	    << "    Takes three floating point parameters as its argument \n"
	    << "    corresponding to the parameters of the Gaussian process covariance \n"
	    << "    function. We use a radial basis and the parameters correspond to \n"
	    << "    the {partial sill variance; bandwidth; exponent}. dualgpm[f] will \n"
	    << "    estimate these from the data if not provided, but for finer user \n"
	    << "    control, please see our other program estimate_rbf. \n"
	    << "\n"
	    << "--neighborhood \n"
	    << "    Default value is 3x the largest --highres voxel dimension. Our \n"
	    << "    algorithm uses a local kriging approximation for the --stdres \n"
	    << "    image, if present. The --neighborhood argument (in mm) controls \n"
	    << "    the radius of that local approximation. \n"
	    << "\n"
	    << "--omask \n"
	    << "    Must match the orientation of the --highres image. For special use \n"
	    << "    cases (e.g. signal loss), if the desired output region is \n"
	    << "    different from the implicit --highres or --hmask image mask, the \n"
	    << "    user can specify the desired output region with this flag and its \n"
	    << "    argument. \n"
	    << "\n"
	    << "--output \n"
	    << "    Include this flag with its argument to direct dualgpm[f] where to \n"
	    << "    write its output. The output basename will be appended with, e.g. \n"
	    << "    *_posterior_mean.nii.gz identifiers. \n"
	    << "\n"
	    << "--seed \n"
	    << "    Default value is set based on the system clock. The output of \n"
	    << "    dualgpm[f] is deterministic given the random seed. \n"
	    << "\n"
	    << "--threads \n"
	    << "    Default value is 80% of the available threads on the current \n"
	    << "    machine. Using more threads will typically result in a faster \n"
	    << "    analysis. \n"
	    << std::endl;
};



template< typename T >
void dualres::GPMCommandParser<T>::show_mcmc_control() const {
  std::cerr << "\nMCMC Control \n"
	    << "------------ \n"
	    << "--burnin \n"
	    << "    Set to 1000 by default. The number of iterations to discard from \n"
	    << "    the beginning of the Hamiltonian Monte Carlo (HMC) chain. \n"
	    << "\n"
	    << "--leapfrog \n"
	    << "    Set to 25 by default. The number of numerical integration steps \n"
	    << "    per HMC iteration. \n"
	    << "\n"
	    << "--mhtarget \n"
	    << "    Set to 0.65 by default. Rate at which the algorithm accepts HMC \n"
	    << "    proposals. Given the --mhtarget and --leapfrog, dualgpm[f] \n"
	    << "    automatically tunes the HMC 'path length' to achieve the desired \n"
	    << "    target acceptance rate. \n"
	    << "\n"
	    << "--monitor \n"
	    << "    Directs dualgpm[f] to print extra verbose information about the \n"
	    << "    HMC chain while it is running. Useful for debugging. \n"
	    << "\n"
	    << "--nsave \n"
	    << "    Set to 1000 by default. The total number of HMC samples is \n"
	    << "    [--burnin] + [--thin] * [--nsave]. Posterior quantities of \n"
	    << "    interest are computed over --nsave samples."
	    << "\n"
	    << "--samples \n"
	    << "    Include this flag in your call to dualgpm[f] to request full MCMC \n"
	    << "    output including samples of the mean parameter from each voxel in \n"
	    << "    the --highres image. Files will be written in plain, tab delimited \n"
	    << "    format and can be large. \n"
	    << "\n"
	    << "--thin \n"
	    << "    Set to 3 by default. The post-burnin HMC sample thinning rate. \n"
	    << "    Per the default, posterior quantities of ininterest are computed \n"
	    << "    using every 3rd post-burnin sample. \n"
	    << std::endl;;
};



template< typename T >
dualres::GPMCommandParser<T>::GPMCommandParser(int argc, char* argv[]) {
  // using typename dualres::CommandParser<T>;
  // typedef typename dualres::CommandParser<T>::call_status call_status;
  // using typename dualres::CommandParser<T>::call_status;
  // dualres::CommandParser<T>::CommandParser();

  std::stringstream ss;
  _status = call_status::success;
  _caller = dualres::path( argv[0] ).stem().string();

  const int K = 3;  // number of covariance parameters
  const auto time = std::chrono::high_resolution_clock::now().time_since_epoch();

  // Default values --------------------------------------------------
  _neighborhood = -1;
  _mcmc_burnin = 1000;
  _mcmc_leapfrog_steps = 25;
  _mcmc_nsave = 1000;
  _mcmc_thin = 3;
  _mcmc_mhtarget = 0.65;
  _monitor = false;
  _output_samples = false;
  _seed = static_cast<unsigned int>(
    std::chrono::duration_cast<std::chrono::milliseconds>(time).count());
  _seed = std::max(_seed, (unsigned)1);
  _threads = (unsigned)0;

  ss << "dualgpm_mcmc_" << _seed << "_";
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
	_output_samples = true;
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
      else if (arg == "--hmask" || arg == "-hm") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _hmask_file = argv[i];
	  if (!dualres::is_nifti_file(_hmask_file)) {
	    std::cerr << _MESSAGE_NOT_NIFTI_FILE << "\t" << _hmask_file
		      << std::endl;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --hmask option requires one argument\n";
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
      else if (arg == "--omask" || arg == "-om") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _omask_file = argv[i];
	  if (!dualres::is_nifti_file(_omask_file)) {
	    std::cerr << _MESSAGE_NOT_NIFTI_FILE << "\t" << _omask_file
		      << std::endl;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --omask option requires one argument\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--samples") {
	_output_samples = true;
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
      else if (arg == "--smask" || arg == "-sm") {
	if (i + 1 < argc) {  // make sure not at end of argv
	  i++;
	  _smask_file = argv[i];
	  if (!dualres::is_nifti_file(_smask_file)) {
	    std::cerr << _MESSAGE_NOT_NIFTI_FILE << "\t" << _smask_file
		      << std::endl;
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << "\nWarning: --smask option requires one argument\n";
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
  if (_hmask_file.empty()) {
    _hmask_file = _highres_file;
  }
  // if (_smask_file.empty() && !_stdres_file.empty()) {
  //   _smask_file = _stdres_file;
  // }
  if (_omask_file.empty()) {
    _omask_file = _highres_file;
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
bool dualres::GPMCommandParser<T>::output_samples() const {
  return _output_samples;
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
std::string dualres::GPMCommandParser<T>::hmask_file() const {
  return _hmask_file;
};


template< typename T >
std::string dualres::GPMCommandParser<T>::omask_file() const {
  return _omask_file;
};


template< typename T >
std::string dualres::GPMCommandParser<T>::smask_file() const {
  return _smask_file;
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




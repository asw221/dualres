
#include <iostream>
#include <string>

#include "dualres/nifti_manipulation.h"
#include "dualres/utilities.h"


template< typename T >
void dualres::FFTWWisdomCommandParser<T>::show_usage() const {
  std::cerr  << "Pre-select optimal Discrete Fourier Transform algorithms:\n"
	     << "Usage:\n"
	     << "\tpreplan_fft path/to/img <options>\n\n";
};


template< typename T >
void dualres::FFTWWisdomCommandParser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "\t--exhaustive    spend lots of time finding the optimal FFT\n"
	    << "\t--patient       (default) spend time finding the optimal FFT\n"
	    << "\t--threads  int  sets the number of threads to run on\n"
	    << "\n"
	    << "Find the optimal 3D DFT algorithm to run on a grid of size related to \n"
	    << "the input image's grid. Information about the optimal algorithm will \n"
	    << "be saved to a small file so that subsequent FFTs are computed \n"
	    << "very efficiently.\n"
	    << "\n"
	    << "The --patient flag usually takes several minutes to select the best \n"
	    << "algorithm, and results in very good performance; the --exhaustive \n"
	    << "flag can take much longer, and will sometimes improve performance \n"
	    << "even further."
	    << "\n\n";
};




template< typename T >
dualres::FFTWWisdomCommandParser<T>::FFTWWisdomCommandParser(int argc, char* argv[]) {
  _status = call_status::success;
  _planner_flag = fftw_flags::patient;

  const std::string _MESSAGE_IMPROPER_THREADS =
    "\nWarning: --threads option requires 1 integer argument\n";
  const std::string _MESSAGE_NOT_NIFTI_FILE =
    "\nWarning: input file does not appear to conform to the nifti-1 standard\n";
  const std::string _MESSAGE_UNRECOGNIZED_OPTION =
    "\nWarning: Unrecognized option: ";  // fill-in-blank
  

  if (argc < 2) {
    _status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if ((arg == "-h") || (arg == "--help")) {
	_status = call_status::help;
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
      else if (dualres::utilities::file_exists(arg)) {
	if (dualres::is_nifti_file(arg)) {
	  _image_file = arg;
	}
	else {
	  std::cerr << _MESSAGE_NOT_NIFTI_FILE
		    << "\t" << arg << std::endl;
	  _status = call_status::error;
	}
      }
      else if (arg == "--patient") {
	_planner_flag = fftw_flags::patient;
      }
      else if (arg == "--exhaustive") {
	_planner_flag = fftw_flags::exhaustive;
      }
      else {
	std::cerr << _MESSAGE_UNRECOGNIZED_OPTION << arg << std::endl;;
	_status = call_status::error;
      }
      
      if (error() || help_invoked()) {
	break;
      }
    }  // for (int i = 1; i < argc; i++)
  }  // if (argc < 2) ... else
  if (help_invoked()) {
    show_help();
  }
  if (error()) {
    show_usage();
    std::cerr << "\nSee preplan_fft --help for more information\n";
  }
};




template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::error() const {
  return _status == call_status::error;
};


template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::flagged_exhaustive() const {
  return _planner_flag == fftw_flags::exhaustive;
};


template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::flagged_patient() const {
  return _planner_flag == fftw_flags::patient;
};


template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};


template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
int dualres::FFTWWisdomCommandParser<T>::threads() const {
  return _threads;
};


template< typename T >
std::string dualres::FFTWWisdomCommandParser<T>::image_file() const {
  return _image_file;
};


template< typename T >
bool dualres::FFTWWisdomCommandParser<T>::native_grid() const {
  return false;
};


template< typename T >
dualres::FFTWWisdomCommandParser<T>::operator bool() const {
  return !error();
};

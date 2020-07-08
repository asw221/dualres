
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>


#include "dualres/nifti_manipulation.h"



template< typename T >
void dualres::KrigingCommandParser<T>::show_usage() const {
  std::cerr << "\nUsage:\n"
	    << "\t" << _caller
	    << " path/to/img --space path/to/img2 <options>\n\n";
};


template< typename T >
void dualres::KrigingCommandParser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "--exponent  float  kernel exponent\n"
	    << "--fwhm      float  kernel full width (mm) at half max\n"
	    << "--radius    float  extent (mm) of the local kriging interpolation\n"
	    << "\nimg is a valid NIfTI file, and float denotes a floating point\n"
	    << "parameter. Computes a local kriging interpolation of the input image\n"
	    << "into a grid specified by the '--space' image. Utilizes a radial basis\n"
	    << "kernel and writes a new NIfTI file to disc, with\n"
	    << "*_<fwhm>mm_fwhm_kriged.nii appended to the original file name.\n"
	    << "Default parameter values are [fwhm = 6] and [exponent = 2]; we allow\n"
	    << "any exponent between (0, 2]. The default value of radius corresponds\n"
	    << "to 3x the largest voxel dimension in the '--space' image. For example,\n"
	    << "if the input image is interpolated into a (2 x 2 x 4) mm resolution,\n"
	    << "the radius will be set to 12 mm by default. Smaller radii will\n"
	    << "result in faster computation times but less accurate (or less\n"
	    << "smooth) results.\n\n"
	    << "NB: the output will be masked by the '--space' image"
	    << "\n\n";
};






template< typename T >
dualres::KrigingCommandParser<T>::KrigingCommandParser() {
  _status = call_status::success;
  _exponent = 1.99999;
  _fwhm = 6;
  _radius = -1;
};



template< typename T >
dualres::KrigingCommandParser<T>::KrigingCommandParser(int argc, char **argv) {
  _status = call_status::success;
  _caller = std::string(argv[0]);
  _exponent = 1.99999;
  _fwhm = 6;
  _radius = -1;

  const std::string _MESSAGE_BAD_PARAMETER_VALUE = _caller +
    ": Warning: bad parameter value ";
  const std::string _MESSAGE_UNRECOGNIZED_OPTION = _caller +
    ": Warning: Unrecognized option: ";  // fill-in-blank
  
  
  if (argc < 2) {
    show_usage();
    _status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
	_status = call_status::help;
      }
      else if (arg == "--exponent" || arg == "-e") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    _exponent = (scalar_type)std::stod(argv[i]);
	    if (_exponent <= 0 || _exponent > 2)
	      throw std::domain_error("exponent");
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(exponent = "
		      << argv[i] << ")\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(exponent = "
		    << argv[i] << ")\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--fwhm" || arg == "-f") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    _fwhm = (scalar_type)std::abs(std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(fwhm = "
		      << argv[i] << ")\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(fwhm = "
		    << argv[i] << ")\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--radius" || arg == "--neighborhood" || arg == "-r") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    _radius = (scalar_type)std::abs(std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(radius = "
		      << argv[i] << ")\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(radius = "
		    << argv[i] << ")\n";
	  _status = call_status::error;
	}
      }
      else if (arg == "--space") {
	if ((i + 1) <= argc) {
	  i++;
	  arg = std::string(argv[i]);
	  if (dualres::is_nifti_file(arg)) {
	    _output_image_file = arg;
	  }
	  else {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE
		      << "(output space image is not a NIfTI file)\n";
	    _status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE
		    << "(output space = "
		    << argv[i] << ")\n";
	  _status = call_status::error;
	}
      }
      else if (dualres::is_nifti_file(arg)) {
	_image_file = arg;
      }
      else {
	std::cerr << _MESSAGE_UNRECOGNIZED_OPTION << arg << "\n";
	_status = call_status::error;
      }

      if (error() || help_invoked()) {
	break;
      }
    }  // for (int i = 1; i < argc; ...
  }  // if (argc >= 2) ...
  if (help_invoked()) {
    show_help();
  }
  else if (_image_file.empty()) {
    std::cerr << "\n" << _caller
	      << ": Error: User must supply a NIfTI file input\n";
    _status = call_status::error;
  }
  else if (_output_image_file.empty()) {
    std::cerr << "\n" << _caller
	      << ": Error: User must supply a --space NIfTI file input\n";
    _status = call_status::error;
  }
  if (error())
    std::cerr << "\nSee " << _caller << " --help for more information\n";
};




template< typename T >
bool dualres::KrigingCommandParser<T>::error() const {
  return _status == call_status::error;
};


template< typename T >
bool dualres::KrigingCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};


template< typename T >
bool dualres::KrigingCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
dualres::KrigingCommandParser<T>::operator bool() const {
  return !error();
};


template< typename T >
typename dualres::KrigingCommandParser<T>::scalar_type
dualres::KrigingCommandParser<T>::exponent() const {
  return _exponent;
};


template< typename T >
typename dualres::KrigingCommandParser<T>::scalar_type
dualres::KrigingCommandParser<T>::fwhm() const {
  return _fwhm;
};


template< typename T >
typename dualres::KrigingCommandParser<T>::scalar_type
dualres::KrigingCommandParser<T>::radius() const {
  return _radius;
};


template< typename T >
std::string dualres::KrigingCommandParser<T>::image_file() const {
  return _image_file;
};

template< typename T >
std::string dualres::KrigingCommandParser<T>::interpolant() const {
  return _image_file;
};

template< typename T >
std::string dualres::KrigingCommandParser<T>::output_image() const {
  return _output_image_file;
};






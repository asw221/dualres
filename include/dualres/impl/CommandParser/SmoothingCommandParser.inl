
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>


#include "dualres/nifti_manipulation.h"



template< typename T >
void dualres::SmoothingCommandParser<T>::show_usage() const {
  std::cerr << "\nUsage:\n"
	    << "\t" << _caller << " path/to/img <options>\n\n";
};


template< typename T >
void dualres::SmoothingCommandParser<T>::show_help() const {
  show_usage();
  std::cerr << "Options:\n"
	    << "--fwhm      f1  kernel full width (mm) at half max\n"
	    << "--exponent  f1  kernel exponent\n"
	    << "\nimg is a valid NIfTI file, and f1 denotes a floating point "
	    << "parameter. Smooths the input image with a radial basis kernel "
	    << "and writes a new NIfTI file to disc, with *_<fwhm>mm_fwhm.nii "
	    << "appended to the original file name.\n"
	    << "Default parameter values are [fwhm = 6] and [exponent = 2]; "
	    << "we allow any exponent between (0, 2] although technically "
	    << "you will only get a 'gaussian' smooth from the default."
	    << "\n\n";
};






template< typename T >
dualres::SmoothingCommandParser<T>::SmoothingCommandParser() {
  _status = call_status::success;
  _exponent = 1.99999;
  _fwhm = 6;
  _radius = -1;
};



template< typename T >
dualres::SmoothingCommandParser<T>::SmoothingCommandParser(int argc, char **argv) {
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
  if (error())
    std::cerr << "\nSee " << _caller << " --help for more information\n";
};




template< typename T >
bool dualres::SmoothingCommandParser<T>::error() const {
  return _status == call_status::error;
};


template< typename T >
bool dualres::SmoothingCommandParser<T>::help_invoked() const {
  return _status == call_status::help;
};


template< typename T >
bool dualres::SmoothingCommandParser<T>::operator!() const {
  return error();
};


template< typename T >
dualres::SmoothingCommandParser<T>::operator bool() const {
  return !error();
};


template< typename T >
typename dualres::SmoothingCommandParser<T>::scalar_type
dualres::SmoothingCommandParser<T>::exponent() const {
  return _exponent;
};


template< typename T >
typename dualres::SmoothingCommandParser<T>::scalar_type
dualres::SmoothingCommandParser<T>::fwhm() const {
  return _fwhm;
};


template< typename T >
typename dualres::SmoothingCommandParser<T>::scalar_type
dualres::SmoothingCommandParser<T>::radius() const {
  return _radius;
};


template< typename T >
std::string dualres::SmoothingCommandParser<T>::image_file() const {
  return _image_file;
};








// -------------------------------------------------------------------




template< typename T >
void dualres::SimulationCommandParser<T>::show_help() const {
  this->show_usage();
  std::cerr << "Options:\n"
	    << "\t--fwhm       float  kernel full width (mm) at half max\n"
	    << "\t--exponent   float  kernel exponent\n"
	    << "\t--mean_image img    mean intensity image\n"
	    << "\t--seed       int    RNG seed\n"
	    << "\n"
	    << "\n~~EDIT ME~~\n"
	    << "img is a valid NIfTI file, and f1 denotes a floating point "
	    << "parameter. Smooths the input image with a radial basis kernel "
	    << "and writes a new NIfTI file to disc, with *_<fwhm>mm_fwhm.nii "
	    << "appended to the original file name.\n"
	    << "Default parameter values are [fwhm = 6] and [exponent = 2]; "
	    << "we allow any exponent between (0, 2] although technically "
	    << "you will only get a 'gaussian' smooth from the default."
	    << "\n\n";
};



template< typename T >
dualres::SimulationCommandParser<T>::SimulationCommandParser(int argc, char **argv) {
  this->_caller = std::string(argv[0]);
  _seed = 42;

  const std::string _MESSAGE_BAD_PARAMETER_VALUE = this->_caller +
    ": Warning: bad parameter value ";
  const std::string _MESSAGE_IMPROPER_SEED =
    "\nWarning: --seed option requires 1 integer argument\n";
  const std::string _MESSAGE_UNRECOGNIZED_OPTION = this->_caller +
    ": Warning: Unrecognized option: ";  // fill-in-blank
  
  
  if (argc < 2) {
    this->show_usage();
    this->_status = call_status::error;
  }
  else {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
	this->_status = call_status::help;
      }
      else if (arg == "--exponent" || arg == "-e") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    this->_exponent = (scalar_type)std::stod(argv[i]);
	    if (this->_exponent <= 0 || this->_exponent > 2)
	      throw std::domain_error("exponent");
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(exponent = "
		      << argv[i] << ")\n";
	    this->_status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(exponent = "
		    << argv[i] << ")\n";
	  this->_status = call_status::error;
	}
      }
      else if (arg == "--fwhm" || arg == "-f") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    this->_fwhm = (scalar_type)std::abs(std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(fwhm = "
		      << argv[i] << ")\n";
	    this->_status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(fwhm = "
		    << argv[i] << ")\n";
	  this->_status = call_status::error;
	}
      }
      else if (arg == "--mean_image") {
	if ((i + 1) <= argc) {
	  i++;
	  _mean_image = std::string(argv[i]);
	  if (!dualres::is_nifti_file(_mean_image)) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(mean_image: '"
		      << _mean_image << "')\n";
	    this->_status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(mean_image: '"
		    << _mean_image << "')\n";
	  this->_status = call_status::error;
	}
      }
      else if (arg == "--radius" || arg == "--neighborhood" || arg == "-r") {
	if ((i + 1) <= argc) {
	  i++;
	  try {
	    this->_radius = (scalar_type)std::abs(std::stod(argv[i]));
	  }
	  catch (...) {
	    std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(radius = "
		      << argv[i] << ")\n";
	    this->_status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_BAD_PARAMETER_VALUE << "(radius = "
		    << argv[i] << ")\n";
	  this->_status = call_status::error;
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
	    this->_status = call_status::error;
	  }
	}
	else {
	  std::cerr << _MESSAGE_IMPROPER_SEED;
	  this->_status = call_status::error;
	}
      }
      else if (dualres::is_nifti_file(arg)) {
	this->_image_file = arg;
      }
      else {
	std::cerr << _MESSAGE_UNRECOGNIZED_OPTION << arg << "\n";
	this->_status = call_status::error;
      }

      if (this->error() || this->help_invoked()) {
	break;
      }
    }  // for (int i = 1; i < argc; ...
  }  // if (argc >= 2) ...
  if (this->help_invoked()) {
    this->show_help();
  }
  else if (this->_image_file.empty()) {
    std::cerr << "\n" << this->_caller
	      << ": Error: User must supply a NIfTI file input\n";
    this->_status = call_status::error;
  }
  if (this->error())
    std::cerr << "\nSee " << this->_caller << " --help for more information\n";
};



template< typename T >
std::string dualres::SimulationCommandParser<T>::mean_image_file() const {
  return _mean_image;
};


template< typename T >
unsigned int dualres::SimulationCommandParser<T>::seed() const {
  return _seed;
};

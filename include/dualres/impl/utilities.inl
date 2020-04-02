
#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "defines.h"


bool dualres::initialize_temporary_directory() {
  boost::filesystem::create_directory(dualres::__internals::_TEMP_DIR_);
  return boost::filesystem::is_directory(dualres::__internals::_TEMP_DIR_);
};


dualres::__internals::path dualres::fftw_wisdom_file() {
  return dualres::__internals::_FFTW_WISDOM_FILE_;
};


dualres::__internals::rng_type& dualres::rng() {
  return dualres::__internals::_RNG_;
};


int dualres::set_number_of_threads(const unsigned int threads) {
  if (threads > 0 && threads <= dualres::__internals::_MAX_THREADS_) {
    dualres::__internals::_N_THREADS_ = (int)threads;
  }
  return dualres::__internals::_N_THREADS_;
};


int dualres::threads() {
  return dualres::__internals::_N_THREADS_;
};
  

void dualres::set_seed(const unsigned int seed) {
  dualres::__internals::_RNG_.seed(seed);
};

  



bool dualres::utilities::file_exists(const std::string &fname) {
  std::ifstream ifs(fname.c_str());
  if (ifs.is_open()) {
    ifs.close();
    return true;
  }
  return false;
};






// progress_bar
// -------------------------------------------------------------------


dualres::utilities::progress_bar::progress_bar(unsigned int max_val) {
  _active = true;
  __ = '=';
  
  _max_val = max_val;
  _print_width = 60;
  _bar_print_width = _print_width - 8;  // 8 additional characters: || xy.z%
  _value = 0;
};
      
void dualres::utilities::progress_bar::finish() {
  _active = false;
std::cout << std::setprecision(4) << std::endl;
};

void dualres::utilities::progress_bar::operator++() {
  _value++;
  _value = (_value > _max_val) ? _max_val : _value;
};

void dualres::utilities::progress_bar::operator++(int) {
  ++(*this);
};

void dualres::utilities::progress_bar::value(unsigned int value) {
  _value = value;
  _value = (_value > _max_val) ? _max_val : _value;
};
      



template< typename OStream >
OStream& dualres::utilities::operator<<(OStream& os, const dualres::utilities::progress_bar& pb) {
  const double prop = (double)pb._value / pb._max_val;
  const unsigned int bars = (unsigned int)(prop * pb._bar_print_width);
  if (pb._active) {
    if (pb._value > 0) {
      for (unsigned int i = 0; i < pb._print_width; i++)  os << "\b";
    }
    os << "|";
    for (unsigned int i = 1; i <= pb._bar_print_width; i++) {
      if (i <= bars)
	os << pb.__;
      else
	os << " ";
    }
    if (prop < 0.095)
      os << "|  ";
    else if (prop < 0.995)
      os << "| ";
    else
      os << "|";
    os << std::setprecision(1) << std::fixed << (prop * 100) << "%" << std::flush;
  }
  return os;
};





#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>

#include "defines.h"


int dualres::utilities::set_number_of_threads(const unsigned int threads) {
  if (threads > 0 && threads <= dualres::internals::_MAX_THREADS_) {
    dualres::internals::_N_THREADS_ = (int)threads;
  }
  return dualres::internals::_N_THREADS_;
};


int dualres::utilities::threads() {
  return dualres::internals::_N_THREADS_;
};
  

void dualres::utilities::set_seed(const unsigned int seed) {
  dualres::internals::_RNG_.seed(seed);
};

  
bool dualres::utilities::initialize_temporary_directory() {
  boost::filesystem::create_directory(dualres::internals::_TEMP_DIR_);
  return boost::filesystem::is_directory(dualres::internals::_TEMP_DIR_);
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




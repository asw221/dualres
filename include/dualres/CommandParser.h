
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#ifndef _DUALRES_COMMAND_PARSER_
#define _DUALRES_COMMAND_PARSER_

// To Do:
//  - Add print precision parameters for functions that print kernel parameters
//  - Add output image names to GPM
//


namespace dualres {


  template< typename T >
  class CommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    CommandParser();
    bool error() const;
    bool help_invoked() const;
    bool operator!() const;
    operator bool() const;

    void show_usage() const;

  protected:
    call_status status;
  };
  


  template< typename T = float >
  class GPMCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };
    
    GPMCommandParser(int argc, char **argv);
    bool error() const;
    bool help_invoked() const;
    operator bool() const;
    bool operator!() const;
    scalar_type neighborhood() const;
    std::string highres_file() const;
    std::string output_file_base() const;
    std::string stdres_file() const;
    unsigned int mcmc_burnin() const;
    unsigned int mcmc_leapfrog_steps() const;
    unsigned int mcmc_nsave() const;
    unsigned int mcmc_thin() const;
    unsigned int seed() const;
    unsigned int threads() const;
    typename std::vector<scalar_type> covariance_parameters() const;
    typename std::vector<scalar_type>::iterator covariance_begin();
    typename std::vector<scalar_type>::iterator covariance_end();

    void show_usage() const;
    
  private:
    call_status _status;
    scalar_type _neighborhood;
    std::string _highres_file;
    std::string _output_base;
    std::string _stdres_file;
    unsigned int _mcmc_burnin;
    unsigned int _mcmc_leapfrog_steps;
    unsigned int _mcmc_nsave;
    unsigned int _mcmc_thin;
    unsigned int _seed;
    unsigned int _threads;
    std::vector<scalar_type> _covariance_params;
  };


  

  template< typename T = double >
  class EstimRbfCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    EstimRbfCommandParser(int argc, char **argv);
    bool error() const;
    bool help_invoked() const;
    bool operator!() const;
    std::string image_file() const;
    std::string output_file() const;
    scalar_type parameter(const int which) const;
    bool parameter_fixed(const int which) const;
    typename std::vector<scalar_type>::const_iterator kernel_cbegin() const;
    typename std::vector<scalar_type>::const_iterator kernel_cend() const;
    operator bool() const;

    void show_usage() const;

  private:
    call_status _status;
    std::string _image_file;
    std::string _output_file;
    std::vector<scalar_type> _kernel_params;
  };



  template< typename T = double >
  class NeighborhoodCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    NeighborhoodCommandParser(int argc, char **argv);
    bool error() const;
    bool help_invoked() const;
    bool operator!() const;
    scalar_type parameter(const int which) const;
    scalar_type rho() const;
    operator bool() const;

    void show_usage() const;

  private:
    call_status _status;
    scalar_type _rho;
    std::vector<scalar_type> _kernel_params;
  };





  template< typename T = float >
  class FFTWWisdomCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };
    enum class fftw_flags { measure, patient, exhaustive };  // measure not used, currently

    FFTWWisdomCommandParser(int argc, char **argv);
    bool error() const;
    bool flagged_exhaustive() const;
    bool flagged_patient() const;
    bool help_invoked() const;
    bool native_grid() const;
    bool operator!() const;
    unsigned int threads() const;
    std::string image_file() const;
    operator bool() const;

    void show_usage() const;

  private:
    call_status _status;
    fftw_flags _planner_flag;
    unsigned int _threads;
    std::string _image_file;
  };
  
}








template< typename T >
dualres::CommandParser<T>::CommandParser() {
  status = call_status::success;
};

template< typename T >
bool dualres::CommandParser<T>::error() const {
  return status == call_status::error;
};

template< typename T >
bool dualres::CommandParser<T>::help_invoked() const {
  return status == call_status::help;
};

template< typename T >
dualres::CommandParser<T>::operator bool() const {
  return !error();
};

template< typename T >
bool dualres::CommandParser<T>::operator!() const {
  return error();
};

template< typename T >
void dualres::CommandParser<T>::show_usage() const {
  std::cerr << "Error: show_usage() should be redefined for each derived class\n";
};



#include "dualres/impl/CommandParser/GPMCommandParser.inl"
#include "dualres/impl/CommandParser/EstimRbfCommandParser.inl"
#include "dualres/impl/CommandParser/NeighborhoodCommandParser.inl"
#include "dualres/impl/CommandParser/FFTWWisdomCommandParser.inl"

#endif  // _DUALRES_COMMAND_PARSER_



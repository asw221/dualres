
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
    bool monitor() const;
    operator bool() const;
    bool operator!() const;
    int mcmc_burnin() const;
    int mcmc_leapfrog_steps() const;
    int mcmc_nsave() const;
    int mcmc_thin() const;
    int threads() const;
    scalar_type mcmc_mhtarget() const;
    scalar_type neighborhood() const;
    std::string highres_file() const;
    std::string output_file_base() const;
    std::string output_file(const std::string &extension) const;
    std::string stdres_file() const;
    unsigned int seed() const;
    typename std::vector<scalar_type> covariance_parameters() const;
    typename std::vector<scalar_type>::iterator covariance_begin();
    typename std::vector<scalar_type>::iterator covariance_end();

    void show_help() const;
    void show_usage() const;
    
  private:
    bool _monitor;
    call_status _status;
    int _mcmc_burnin;
    int _mcmc_leapfrog_steps;
    int _mcmc_nsave;
    int _mcmc_thin;
    int _threads;
    scalar_type _mcmc_mhtarget;
    scalar_type _neighborhood;
    std::string _highres_file;
    std::string _output_base;
    std::string _stdres_file;
    unsigned int _seed;
    std::vector<scalar_type> _covariance_params;
  };


  

  template< typename T = double >
  class EstimRbfCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    EstimRbfCommandParser(int argc, char **argv);
    bool use_constraint() const;
    bool error() const;
    bool help_invoked() const;
    bool operator!() const;
    double xtol_rel() const;
    std::string image_file() const;
    std::string output_file() const;
    scalar_type parameter(const int which) const;
    bool parameter_fixed(const int which) const;
    typename std::vector<scalar_type>::const_iterator kernel_cbegin() const;
    typename std::vector<scalar_type>::const_iterator kernel_cend() const;
    operator bool() const;

    void show_help() const;
    void show_usage() const;

  private:
    call_status _status;
    bool _use_constraint;
    double _xtol;
    std::string _image_file;
    std::string _output_file;
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
    int threads() const;
    std::string image_file() const;
    operator bool() const;

    void show_usage() const;
    void show_help() const;

  private:
    call_status _status;
    fftw_flags _planner_flag;
    int _threads;
    std::string _image_file;
  };




  template< typename T = float >
  class SmoothingCommandParser {
  public:
    typedef T scalar_type;
    enum class call_status { success, error, help };

    SmoothingCommandParser();
    SmoothingCommandParser(int argc, char **argv);
    bool error() const;
    bool help_invoked() const;
    bool operator!() const;
    operator bool() const;
    scalar_type exponent() const;
    scalar_type fwhm() const;
    scalar_type radius() const;
    std::string image_file() const;

    void show_help() const;
    void show_usage() const;

  protected:
    call_status _status;
    scalar_type _exponent;
    scalar_type _fwhm;
    scalar_type _radius;
    std::string _caller;
    std::string _image_file;
  };




  
  template< typename T = float >
  class SimulationCommandParser : public dualres::SmoothingCommandParser<T> {
  public:
    typedef T scalar_type;
    typedef typename SmoothingCommandParser<scalar_type>::call_status call_status;

    SimulationCommandParser(int argc, char **argv);
    std::string mean_image_file() const;
    unsigned int seed() const;

    void show_help() const;

  private:
    std::string _mean_image;
    unsigned int _seed;
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


  
}  // namespace dualres








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
#include "dualres/impl/CommandParser/FFTWWisdomCommandParser.inl"
#include "dualres/impl/CommandParser/NeighborhoodCommandParser.inl"
#include "dualres/impl/CommandParser/SmoothingCommandParser.inl"

#endif  // _DUALRES_COMMAND_PARSER_



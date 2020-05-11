
#include <fstream>
#include <string>
#include <vector>


#include "dualres/defines.h"


#ifndef _DUALRES_UTILITIES_
#define _DUALRES_UTILITIES_




namespace dualres {
  /*! @addtogroup Dualres
   * @{
   */

  bool initialize_temporary_directory();
  
  dualres::path fftw_wisdom_file();
  dualres::__internals::rng_type& rng();
  
  int set_number_of_threads(const int threads);
  int threads();
  
  void set_seed(const unsigned int seed);

  void set_monitor_simulations(const bool monitor) {
    dualres::__internals::_MONITOR_ = monitor;
  };

  bool monitor_simulations() {
    return dualres::__internals::_MONITOR_;
  };
  

  /*! @} */

  

  namespace utilities {
    /*! @addtogroup Dualres
     * @{
     */

    
    bool file_exists(const std::string &fname);
    


    class progress_bar {
    public:
      progress_bar(unsigned int max_val);
      
      void finish();
      void operator++();
      void operator++(int);
      void value(unsigned int value);

      template< typename OStream >
      friend OStream& operator<<(OStream& os, const progress_bar& pb);
      
    private:
      bool _active;
      char __;
      unsigned int _max_val;
      unsigned int _print_width;
      unsigned int _bar_print_width;
      unsigned int _value;
    };

    
    /*! @} */    
  }  // namespace utilities ------------------------------------------



}  // namespace dualres ----------------------------------------------


#include "dualres/impl/utilities.inl"

#endif  // _DUALRES_UTILITIES_


#include <algorithm>
#include <boost/filesystem.hpp>
// #include <Eigen/Core>
#include <random>
#include <string>
#include <thread>
// #include <type_traits>


#ifndef _DUALRES_DEFINES_
#define _DUALRES_DEFINES_


namespace dualres {

  
  enum class use_lambda_method { PROFILE, NATIVE, EXTENDED };

  enum class nifti_data_type {
    OTHER = 0,  BINARY = 1,   CHAR = 2,    SHORT = 4, INT = 8,
    FLOAT = 16, COMPLEX = 32, DOUBLE = 64, RGB = 128, ALL = 255
  };
  // See:
  // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html


  

  
  namespace __internals {

    typedef std::mt19937 rng_type;
    typedef boost::filesystem::path path;
    

    const int _MAX_THREADS_ = std::thread::hardware_concurrency();
    int _N_THREADS_ = std::max(_MAX_THREADS_ * 4 / 5, 1);
    
    rng_type _RNG_(42);

    const path _TEMP_DIR_(boost::filesystem::temp_directory_path().string() +
			  path::preferred_separator +
			  "dualresTemp" +
			  path::preferred_separator);
    const path _FFTW_WISDOM_FILE_(_TEMP_DIR_.string() + "__fftw_wisdom");
    
  }
  


  
}


// #include "data_types.inl"

#endif  // _DUALRES_DEFINES_








/*
  template< typename T >
  class data_types {
  public:
    typedef T value_type;

    static constexpr ::af_dtype af_dtype = ::af_dtype::f32;
    static constexpr ::af_dtype af_ctype = ::af_dtype::c32;
    static constexpr dualres::nifti_data nifti_data = dualres::nifti_data::OTHER;
    // constexpr ::af_dtype af_dtype() const;    
    // constexpr ::af_dtype af_ctype() const;
    // constexpr dualres::nifti_data nifti_data() const;
  };


    template<>
    class data_types<float> {
    public:
      typedef float value_type;

      static constexpr ::af_dtype af_dtype = ::af_dtype::f32;
      static constexpr ::af_dtype af_ctype = ::af_dtype::c32;
      static constexpr dualres::nifti_data nifti_data = dualres::nifti_data::FLOAT;
    };

  
    template<>
    class data_types<double> {
    public:
      typedef double value_type;

      static constexpr ::af_dtype af_dtype = ::af_dtype::f64;
      static constexpr ::af_dtype af_ctype = ::af_dtype::c64;
      static constexpr dualres::nifti_data nifti_data = dualres::nifti_data::DOUBLE;
    };
*/

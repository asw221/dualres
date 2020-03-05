
#include <arrayfire.h>
#include <Eigen/Core>
#include <random>
#include <type_traits>


#ifndef _DUALRES_DEFINES_
#define _DUALRES_DEFINES_


namespace dualres {
  
  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> qform_type;

  
  enum class use_lambda_method { PROFILE, NATIVE, EXTENDED };

  enum class nifti_data {
    OTHER = 0,  BINARY = 1,   CHAR = 2,    SHORT = 4, INT = 8,
    FLOAT = 16, COMPLEX = 32, DOUBLE = 64, RGB = 128, ALL = 255
  };
  // See:
  // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html



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



  
  namespace internals {

    std::mt19937 _RNG_(42);
    af::randomEngine _AF_RNG_(af::randomEngineType::AF_RANDOM_ENGINE_DEFAULT, 21);
    
  }


  void set_seed(const unsigned int seed) {
    dualres::internals::_RNG_.seed(seed);
    dualres::internals::_AF_RNG_.setSeed(seed / 2);
  };

  
}


// #include "data_types.inl"

#endif  // _DUALRES_DEFINES_

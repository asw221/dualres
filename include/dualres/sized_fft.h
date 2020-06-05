
#include <complex>
#include <fftw3.h>
#include <iostream>
// #include <memory>
#include <stdexcept>
// #include <utility>
#include <vector>

// #include "dualres/utilities.h"


#ifndef _DUALRES_SIZED_FFT_
#define _DUALRES_SIZED_FFT_


namespace dualres {


  template< typename RealType >
  struct fftw_types {};
  
  template<>
  struct fftw_types<float> {
    typedef fftwf_complex fftw_complex_type;
    typedef fftwf_plan_s  fftw_plan_s_type;
    typedef fftwf_plan    fftw_plan_type;
  };

  template<>
  struct fftw_types<double> {
    typedef fftw_complex  fftw_complex_type;
    typedef fftw_plan_s   fftw_plan_s_type;
    typedef fftw_plan     fftw_plan_type;
  };
  



  
  // Unused, yet
  enum class fftw_order { RowMajor, ColMajor };





  template< typename RealType >
  class sized_fft {
  public:
    typedef RealType scalar_type;
    typedef typename std::complex<RealType> complex_type;
    typedef typename dualres::fftw_types<RealType>::fftw_complex_type
      fftw_complex_type;
    typedef typename dualres::fftw_types<RealType>::fftw_plan_s_type
      fftw_plan_s_type;
    typedef typename dualres::fftw_types<RealType>::fftw_plan_type
      fftw_plan_type;


    sized_fft();
    sized_fft(  // define 3D DFT constructor only
      const int d0,
      const int d1,
      const int d2,
      const unsigned int flags = FFTW_PATIENT
    );
    ~sized_fft();


    int dimension() const;
    int grid_length() const;
    int save_plans() const;
    
    void forward(complex_type* input) const;
    void forward(complex_type* input, complex_type* output) const;

    void inverse(complex_type* input) const;
    void inverse(complex_type* input, complex_type* output) const;

    
    void operator= (const dualres::sized_fft<RealType> &rhs);


    std::vector<int> grid_dimensions() const;
  

  private:
    bool _init;
    int _N;  // grid length = prod(_grid_dimensions)
    fftw_plan_type __forward_plan;
    fftw_plan_type __inverse_plan;
    std::vector<int> _grid_dimensions;
    unsigned int _flags;
  };



}  // namespace dualres



template< typename RealType >
dualres::sized_fft<RealType>::sized_fft() {
  _init = false;
};


template<>
dualres::sized_fft<double>::sized_fft(
  const int d0,
  const int d1,
  const int d2,
  const unsigned int flags
) {
  if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
    throw std::domain_error(
      "sized_fft: all grid dimensions must be >= 1");
  }
  _grid_dimensions.resize(3);
  _grid_dimensions[0] = d0;
  _grid_dimensions[1] = d1;
  _grid_dimensions[2] = d2;
  _N = d0 * d1 * d2;

  std::vector<complex_type> pseudo_data(_N, complex_type(0, 0));

  ::fftw_plan_with_nthreads(dualres::threads());
  // ::fftw_plan_with_nthreads(6);
  
  if (dualres::utilities::file_exists(dualres::fftw_wisdom_file().string()))
    ::fftw_import_wisdom_from_filename(dualres::fftw_wisdom_file().c_str());

  // Column-Major default
  __forward_plan = ::fftw_plan_dft_3d(
    d2, d1, d0,
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_FORWARD, flags);
  __inverse_plan = ::fftw_plan_dft_3d(
    d2, d1, d0,
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_BACKWARD, flags);

  _flags = flags;
  _init = true;
};



template<>
dualres::sized_fft<float>::sized_fft(
  const int d0,
  const int d1,
  const int d2,
  const unsigned int flags
) {
  if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
    throw std::domain_error(
      "sized_fft: all grid dimensions must be >= 1");
  }
  _grid_dimensions.resize(3);
  _grid_dimensions[0] = d0;
  _grid_dimensions[1] = d1;
  _grid_dimensions[2] = d2;
  _N = d0 * d1 * d2;

  std::vector<complex_type> pseudo_data(_N, complex_type(0, 0));

  ::fftwf_plan_with_nthreads(dualres::threads());
  // ::fftwf_plan_with_nthreads(6);
  
  if (dualres::utilities::file_exists(dualres::fftw_wisdom_file().string()))
    ::fftwf_import_wisdom_from_filename(dualres::fftw_wisdom_file().c_str());

  // Column-Major default
  __forward_plan = ::fftwf_plan_dft_3d(
    d2, d1, d0,
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_FORWARD, flags);
  __inverse_plan = ::fftwf_plan_dft_3d(
    d2, d1, d0,
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_BACKWARD, flags);

  _flags = flags;
  _init = true;
};



template<>
dualres::sized_fft<double>::~sized_fft() {
  if (_init) {
    ::fftw_destroy_plan(__forward_plan);
    ::fftw_destroy_plan(__inverse_plan);
  }
};

template<>
dualres::sized_fft<float>::~sized_fft() {
  if (_init) {
    ::fftwf_destroy_plan(__forward_plan);
    ::fftwf_destroy_plan(__inverse_plan);
  }
};



template< typename RealType >
int dualres::sized_fft<RealType>::dimension() const {
  return _grid_dimensions.size();
};

template< typename RealType >
int dualres::sized_fft<RealType>::grid_length() const {
  return _N;
};




template<>
int dualres::sized_fft<double>::save_plans() const {
  return ::fftw_export_wisdom_to_filename(dualres::fftw_wisdom_file().c_str());
};

template<>
int dualres::sized_fft<float>::save_plans() const {
  return ::fftwf_export_wisdom_to_filename(dualres::fftw_wisdom_file().c_str());
};




template< typename RealType >
void dualres::sized_fft<RealType>::forward(
  typename dualres::sized_fft<RealType>::complex_type* input
) const {
  forward(input, input);
};


template<>
void dualres::sized_fft<double>::forward(
  typename dualres::sized_fft<double>::complex_type* input,
  typename dualres::sized_fft<double>::complex_type* output
) const {
  ::fftw_execute_dft(
    __forward_plan,
    reinterpret_cast<fftw_complex_type*>(input),
    reinterpret_cast<fftw_complex_type*>(output)
  );
};

template<>
void dualres::sized_fft<float>::forward(
  typename dualres::sized_fft<float>::complex_type* input,
  typename dualres::sized_fft<float>::complex_type* output
) const {
  ::fftwf_execute_dft(
    __forward_plan,
    reinterpret_cast<fftw_complex_type*>(input),
    reinterpret_cast<fftw_complex_type*>(output)
  );
};


template< typename RealType >
void dualres::sized_fft<RealType>::inverse(
  typename dualres::sized_fft<RealType>::complex_type* input
) const {
  inverse(input, input);
};


template<>
void dualres::sized_fft<double>::inverse(
  typename dualres::sized_fft<double>::complex_type* input,
  typename dualres::sized_fft<double>::complex_type* output
) const {
  ::fftw_execute_dft(
    __inverse_plan,
    reinterpret_cast<fftw_complex_type*>(input),
    reinterpret_cast<fftw_complex_type*>(output)
  );  
};

template<>
void dualres::sized_fft<float>::inverse(
  typename dualres::sized_fft<float>::complex_type* input,
  typename dualres::sized_fft<float>::complex_type* output
) const {
  ::fftwf_execute_dft(
    __inverse_plan,
    reinterpret_cast<fftw_complex_type*>(input),
    reinterpret_cast<fftw_complex_type*>(output)
  );  
};




template<>
void dualres::sized_fft<double>::operator= (
  const dualres::sized_fft<double> &rhs
) {
  _N = rhs._N;
  _grid_dimensions = std::vector<int>(
    rhs._grid_dimensions.cbegin(),
    rhs._grid_dimensions.cend()
  );

  ::fftw_plan_with_nthreads(dualres::threads());

  std::vector<complex_type> pseudo_data(_N, complex_type(0, 0));
  
  __forward_plan = ::fftw_plan_dft_3d(
    _grid_dimensions[2], _grid_dimensions[1], _grid_dimensions[0],
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_FORWARD, rhs._flags);
  
  __inverse_plan = ::fftw_plan_dft_3d(
    _grid_dimensions[2], _grid_dimensions[1], _grid_dimensions[0],
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_BACKWARD, rhs._flags);
  
  _init = true;
};


template<>
void dualres::sized_fft<float>::operator= (
  const dualres::sized_fft<float> &rhs
) {
  _N = rhs._N;
  _grid_dimensions = std::vector<int>(
    rhs._grid_dimensions.cbegin(),
    rhs._grid_dimensions.cend()
  );

  ::fftwf_plan_with_nthreads(dualres::threads());

  std::vector<complex_type> pseudo_data(_N, complex_type(0, 0));
  
  __forward_plan = ::fftwf_plan_dft_3d(
    _grid_dimensions[2], _grid_dimensions[1], _grid_dimensions[0],
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_FORWARD, rhs._flags);
  
  __inverse_plan = ::fftwf_plan_dft_3d(
    _grid_dimensions[2], _grid_dimensions[1], _grid_dimensions[0],
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    reinterpret_cast<fftw_complex_type*>(pseudo_data.data()),
    FFTW_BACKWARD, rhs._flags);
  
  _init = true;
};



template< typename RealType >
std::vector<int>
dualres::sized_fft<RealType>::grid_dimensions() const {
  return std::vector<int>(_grid_dimensions.cbegin(),
			  _grid_dimensions.cend());
};

#endif  // _DUALRES_SIZED_FFT_


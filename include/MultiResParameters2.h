
#include <algorithm>
#include <complex>
#include <cmath>
#include <Eigen/Core>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <omp.h>
#include <random>
#include <vector>

#include "circulant_base.h"
#include "defines.h"
#include "MultiResData2.h"
#include "utilities.h"



#ifndef _DUALRES_MULTI_RES_PARAMETERS_2_
#define _DUALRES_MULTI_RES_PARAMETERS_2_


namespace dualres {

    

  template< typename T >
  class MultiResParameters {
  public:
    typedef T scalar_type;
    typedef typename std::complex<T> complex_type;
    typedef typename Eigen::Array<scalar_type, Eigen::Dynamic, 1> ArrayType;
    typedef typename Eigen::Vector<scalar_type, Eigen::Dynamic, 1> VectorType;
    typedef typename Eigen::Vector<complex_type, Eigen::Dynamic, 1> ComplexArrayType;
    typedef typename Eigen::Vector<complex_type, Eigen::Dynamic, 1> ComplexVectorType;
    

    MultiResParameters(
      const int n_datasets,
      const typename std::vector<scalar_type> &kernel_parameters,
      const Eigen::MatrixXi &ijk_y0,
      const dualres::qform_type &Qform_y0,
      const dualres::use_lambda_method select = dualres::use_lambda_method::EXTENDED
    );

    ~MultiResParameters();


    
    void print_test();
    
    scalar_type update(
      const typename dualres::MultiResData<scalar_type> &data,
      const scalar_type eps = 0.1,
      const int L = 10
    );
    
    const typename ComplexVectorType& gradient(
      const typename dualres::MultiResData<scalar_type> &data
    );
    typename VectorType mu() const;
    
    // const af::array& lambda() const;
    const typename ComplexArrayType& lambda() const;
    const typename ComplexArrayType& lambda_inv() const;
    const std::vector<int>& indices() const;
    
    scalar_type log_prior() const;  // returns array of dim = 1
    scalar_type log_likelihood(const typename dualres::MultiResData<scalar_type> &data);
    // ^^ returns array of dim = 1
    scalar_type log_posterior(const typename dualres::MultiResData<scalar_type> &data);
    scalar_type sigma(const int which) const;


    
  private:
    fftwf_plan __forward_fft_plan;
    fftwf_plan __backward_fft_plan;
    

    ComplexArrayType _grad;
    ComplexArrayType _lambda;
    ComplexArrayType _lambda_mass;
    ComplexArrayType _momentum;
    ComplexArrayType _mu;
    ComplexArrayType _mu_star;
    ComplexArrayType __temp_product;

    VectorType _real_mu;
    VectorType _real_sub_grad;
    VectorType _Y_star;


    Eigen::VectorXi __lambda_grid_indices;
    
    std::vector<int> _negative_eigen_values;

    scalar_type _total_energy;
    scalar_type _initial_energy;
    
    int _n_datasets;

    Eigen::Vector3i __image_grid_dims;  // dimensions of bounded image
    Eigen::Vector3i __lambda_grid_dims;

    typename std::vector<scalar_type> _theta;
    typename std::vector<scalar_type> _sigma_sq_inv;

    typename std::normal_distribution<scalar_type> __Standard_Gaussian;

    const ComplexArrayType& _compute_gradient(
      const typename dualres::MultiResData<scalar_type> &data,
      ComplexArrayType &mu_star
    );

    void _compute_mass_matrix_eigen_values();
    void _initialize_mu(const scalar_type k = 10);
    void _initialize_sigma(const scalar_type shape = 1, const scalar_type rate = 1);
    void _sample_momentum();
    void _update_sigma_sq_inv(const typename dualres::MultiResData<scalar_type> &data);

    void _low_rank_adjust(ComplexArrayType &A) const;
    
    scalar_type _log_prior(ComplexArrayType &mu_star) const;
    scalar_type _potential_energy() const;
    
    scalar_type _log_likelihood(
      const typename dualres::MultiResData<scalar_type> &data,
      ComplexArrayType &mu_star
    );
    scalar_type _log_posterior(
      const typename dualres::MultiResData<scalar_type> &data,
      ComplexArrayType &mu_star
    );
  };

  
}





template< typename T >
dualres::MultiResParameters<T>::~MultiResParameters() {
  fftwf_destroy_plan(__forward_fft_plan);
  fftwf_destroy_plan(__backward_fft_plan);
};




template< typename T >
dualres::MultiResParameters<T>::MultiResParameters(
  const int n_datasets,
  const typename std::vector<
    typename dualres::MultiResParameters<T>::scalar_type> &kernel_parameters,
  const Eigen::MatrixXi &ijk_y0,
  const dualres::qform_type &Qform_y0,
  const dualres::use_lambda_method lambda_method
) {
  _n_datasets = n_datasets;
  _theta = std::vector<scalar_type>(kernel_parameters.cbegin(), kernel_parameters.cend());
  __Standard_Gaussian = std::normal_distribution<scalar_type>(0.0, 1.0);

  bool compute_lambda_on_extended_grid = true;
  
  // Compute diagonal of eigen vector matrix
  // lambda_grid_dims: "dimensions" of (extended?) circulant matrix base
  //   - (_lambda stored as a flat array)
  //   - lambda_grid_dims = 2 * (__image_grid_dims - 1);
  __image_grid_dims = ijk_y0.colwise().maxCoeff().head<3>();
  __lambda_grid_dims = 2 * __image_grid_dims;
  __image_grid_dims += Eigen::Vector3i::Ones();
  
  if (lambda_method != dualres::use_lambda_method::EXTENDED) {
    throw std::domain_error("Only extended-grid eigen vector computation implemented");
    // want to bother profiling?
    compute_lambda_on_extended_grid = false;  // Possibly?
  }
  else {  // Extended grid computation
    // lambda_grid_dims = pow(2, ceil(log2( lambda_grid_dims )))
    //  - Eigen does not implement a log2() method
    __lambda_grid_dims = Eigen::pow(2.0,
      (__lambda_grid_dims.cast<double>().array().log() / std::log(2.0)).ceil())
      .matrix().cast<int>().eval();
  }

  
    
  // Comptue fftw plans before filling _lambda:
  //  - _lambda will be computed in column-major order: reverse order of dims in plan
  // _lambda = ComplexArrayType::Constant(
  //   __lambda_grid_dims.prod(), complex_type(0.0, 0.0));
  _lambda = ComplexArrayType::Zero(__lambda_grid_dims.prod());
  _grad = _lambda;
  _lambda_mass = _lambda;
  _momentum = _lambda;
  __temp_product = _lambda;

  // In-place transformations:
  if (dualres::utilities::file_exists(dualres::internals::_FFTW_WISDOM_FILE_.string()))
    fftwf_import_wisdom_from_filename(dualres::internals::_FFTW_WISDOM_FILE_.c_str());
  
  std::cout << "Selecting forward DFT algorithm... " << std::flush;
  __forward_fft_plan = fftwf_plan_dft_3d(
    __image_grid_dims[2], __image_grid_dims[1], __image_grid_dims[0],
    reinterpret_cast<fftwf_complex*>(_lambda.data()),
    reinterpret_cast<fftwf_complex*>(_lambda.data()),
    FFTW_FORWARD, FFTW_PATIENT
  );
  std::cout << "Done!" << std::endl;
    
  std::cout << "Selecting inverse DFT algorithm... " << std::flush;
  __backward_fft_plan = fftwf_plan_dft_3d(
    __image_grid_dims[2], __image_grid_dims[1], __image_grid_dims[0],
    reinterpret_cast<fftwf_complex*>(_lambda.data()),
    reinterpret_cast<fftwf_complex*>(_lambda.data()),
    FFTW_BACKWARD, FFTW_PATIENT
  );
  std::cout << "Done!" << std::endl;

  fftwf_export_wisdom_to_filename(dualres::internals::_FFTW_WISDOM_FILE_.c_str());
  

  // Now compute _lambda
  _lambda = dualres::circulant_base_3d<scalar_type>(__image_grid_dims, Qform_y0,
    compute_lambda_on_extended_grid).cast<complex_type>();
  for (int i = 0; i < _lambda.size(); i++) {
    _lambda[i][0] = dualres::kernels::rbf(_lambda[i], _theta[1], _theta[2], _theta[0]);
  }
  fftwf_execute(__forward_fft_plan);

  // Get indices where real parts of eigen values are < 0
  for (int i = 0; i < _lambda.size(); i++) {
    if (_lambda[i].real() < 0)
      __negative_eigen_values.push_back(i);
  }
    

  std::cout << (_lambda.size() - __negative_eigen-values.size())
	    << " eigen values have real-part > 0  ("
	    << std::setprecision(2) << std::fixed
	    << ( (1.0 - (double)__negative_eigen-values.size() / _lambda.size())
		 * 100 ) << "%)"
	    << std::endl;
  
  _initialize_mu();
  _initialize_sigma();

  __lambda_grid_indices = ijk_y0.col(2) * __lambda_grid_dims[0] * __lambda_grid_dims[1] +
    ijk_y0.col(1) * __lambda_grid_dims[0] + ijk_y0.col(0);
};








template< typename T >
void dualres::MultiResParameters<T>::_low_rank_adjust(
  typename dualres::MultiResParameters<T>::ComplexArrayType &A
) const {
  for (std::vector<int>::const_iterator it = __negative_eigen_values.cbegin();
       it != __negative_eigen_values.cend(); ++it) {
    A[*it] = complex_type(0, 0);
  }
};






template< typename T >
const typename dualres::MultiResParameters<T>::ComplexArrayType&
dualres::MultiResParameters<T>::_compute_gradient(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data,
  typename dualres::MultiResParameters<T>::ComplexArrayType& mu_star
) {
#ifndef NDEBUG
  if (data.n_datasets() < _n_datasets)
    throw std::logic_error("Insufficient data for parameters");
#endif
  fftwf_execute_dft(__backward_fft_plan,
    reinterpret_cast<fftwf_complex*>(mu_star.data()),
    reinterpret_cast<fftwf_complex*>(_grad.data())
  );
  _grad /= -_lambda.size() * _lambda;
  _low_rank_adjust(_grad);
  fftwf_execute_dft(__forward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_grad.data()),
    reinterpret_cast<fftwf_complex*>(_grad.data())
  );
  _real_mu = dualres::nullary_index(mu_star.matrix().real(), __lambda_grid_indices);
  // _grad = -af::dft(af::idft(mu_star) / _lambda * _positive_eigen_values / mu_star.elements());
  if (_n_datasets == 1) {
    _real_sub_grad = _sigma_sq_inv[0] * (data.Y(0) - _real_mu);
  }
  else if (_n_datasets == 2) {
    _real_sub_grad = _sigma_sq_inv[0] * (data.Y(0) - _real_mu) +
      _sigma_sq_inv[1] * ((data.Y(1) - (data.W(0) * _real_mu)).transpose() *
			  data.W(0)).transpose();
  }
  else {
    throw std::logic_error("Gradient only implemented for single or dual resolution");
  }
  for (int i = 0; i < __lambda_grid_indices.size(); i++)
    _grad[__lambda_grid_indices[i]] += complex_type(_real_sub_grad[i], 0);
  return _grad;
};





template< typename T >
const typename dualres::MultiResParameters<T>::ComplexArrayType&
dualres::MultiResParameters<T>::gradient(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  return _compute_gradient(data, _mu);
};



template< typename T >
const typename dualres::MultiResParameters<T>::VectorType&
dualres::MultiResParameters<T>::mu() const {
  return _real_mu;
};




template< typename T >
const const typename dualres::MultiResParameters<T>::ComplexArrayType&
dualres::MultiResParameters<T>::lambda() const {
  return _lambda;
};


template< typename T >
const typename dualres::MultiResParameters<T>::ComplexArrayType&
dualres::MultiResParameters<T>::lambda_inv() const {
  return 1 / _lambda;
};


template< typename T >
const Eigen::VectorXi& dualres::MultiResParameters<T>::data_indices() const {
  return __lambda_grid_indices;
};





template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::_log_prior(
  const typename dualres::MultiResParameters<T>::ComplexArrayType& &mu_star
) const {
  // __Fh_x = af::idft(mu_star) / std::sqrt((scalar_type)mu_star.elements());
  // scalar_type __lp = -0.5 *
  //   af::sum<scalar_type>(af::real(af::conjg(__Fh_x) / _lambda
  // 				  * _positive_eigen_values * __Fh_x));
  complex_type __lp(0, 0);
  fftwf_execute_dft(__backward_fft_plan,
    reinterpret_cast<fftwf_complex*>(mu_star.data()),
    reinterpret_cast<fftwf_complex*>(__temp_product.data())
  );
  // __temp_product /= _lambda.size() * _lambda;
  _low_rank_adjust(__temp_product);
#pragma omp parallel for shared(__temp_product, _lambda) private(i) reduction(+ : __lp)
  for (int i = 0; i < __temp_product.size(); i++)
    __lp += std::conj(__temp_product[i]) * __temp_product[i] / (_lambda[i] * _lambda.size());
  return __lp.real();
};



template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::log_prior() const {
  return _log_prior(_mu);
};




template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::_log_likelihood(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data,
  typename dualres::MultiResParameters<T>::ComplexArrayType &mu_star
) {
#ifndef NDEBUG
  if (data.n_datasets() < _n_datasets)
    throw std::logic_error("Insufficient data for parameters");
#endif
  scalar_type __ll(0);
  _real_mu = dualres::nullary_index(mu_star.matrix().real(), __lambda_grid_indices);
  __ll = -0.5 * _sigma_sq_inv[0] * (data.Y(0) - _real_mu).squaredNorm();
  if (_n_datasets == 2) {
    __ll += -0.5 * sigma_sq_inv[1] * (data.Y(1) - (W * _real_mu)).squaredNorm();
  }
  else if (_n_datasets > 2) {
    throw std::logic_error("Likelihood only implemented for single or dual resolution");
  }
  return __ll;
};


template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::log_likelihood(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  return _log_likelihood(data, _mu);
};




template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::_log_posterior(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data,
  typename dualres::MultiResParameters<T>::ComplexArrayType &mu_star
) {
  return _log_prior(mu_star) + _log_likelihood(data, mu_star);
};


template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::log_posterior(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  return _log_posterior(data, _mu);
};






template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::_potential_energy() const {
  // __Fh_x = af::idft(_momentum) / std::sqrt((scalar_type)_momentum.elements());
  // return -0.5 * af::sum<scalar_type>(af::real(af::conjg(__Fh_x) * _lambda_mass * __Fh_x));
  complex_type __pe(0, 0);
  fftwf_execute_dft(__backward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_momentum.data()),
    reinterpret_cast<fftwf_complex*>(__temp_product.data())
  );
#pragma omp parallel for shared(__temp_product, _lambda_mass) private(i) reduction(+ : __pe)
  for (int i = 0; i < __temp_product.size(); i++)
    __pe += std::conj(__temp_product[i]) * _lambda_mass[i] * __temp_product[i];
  return -0.5 * __pe.real();
};







template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::sigma(const int which) const {
#ifndef NDEBUG
  if (which < 0 || which >= _n_datasets)
    throw std::logic_error("Parameters: bad indexing");
#endif
  return std::sqrt(1 / _sigma_sq_inv[which]);
};







// template< af_dtype T >
// void dualres::MultiResParameters<T>::print_test() {
//   std::cout << _n_datasets << " datasets measured on a grid of dim ("
// 	      << (_mu.dims(0) / 2 + 1) << " x " << (_mu.dims(1) / 2 + 1)
// 	      << " x " << (_mu.dims(2) / 2 + 1) << ")\n"
// 	      << "  theta = [" << _theta[0] << ", " << _theta[1]
// 	      << ", " << _theta[2] << "]^T\n"
// 	      << "  sigma^-2 = [";
//   for (int i = 0; i < _n_datasets; i++) {
//     std::cout << _sigma_sq_inv[i] << ", ";
//   }
//   std::cout << "\b\b]^T\n"
// 	      << "  lambda =\n";
//   print_array_summary(af::real(_lambda));
//   std::cout << "  mu =\n";
//   print_array_summary(af::real(_mu(__lambda_grid_indices)));
//   std::cout << "\n";
// };






template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::update(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data,
  const typename dualres::MultiResParameters<T>::scalar_type eps,
  const int L
) {
  static std::uniform_real_distribution<scalar_type> Uniform(0, 1);
  scalar_type k = 0.5, proposed_energy, R;
  _mu_star = _mu;
  _compute_mass_matrix_eigen_values();
  _sample_momentum();
  _low_rank_adjust(_lambda_mass);
  _initial_energy += _log_posterior(data, _mu);  // _sample_momentum() initilizes energy
  _momentum += k * eps * _compute_gradient(data, _mu_star);
  for (int step = 0; step < L; step++) {
    k = (step == (L - 1)) ? 0.5 : 1;
    fftwf_execute_dft(__backward_fft_plan,
      reinterpret_cast<fftwf_complex*>(_momentum.data()),
      reinterpret_cast<fftwf_complex*>(_mu_star.data())
    );
    _mu_star *= eps / _lambda_mass.size() * _lambda_mass;
    fftwf_execute_dft(__forward_fft_plan,
      reinterpret_cast<fftwf_complex*>(_mu_star.data()),
      reinterpret_cast<fftwf_complex*>(_mu_star.data())
    );
    // _mu_star += eps * af::dft(_lambda_mass * af::idft(_momentum) /
    // 			       _momentum.elements());
    _momentum += k * eps * _compute_gradient(data, _mu_star);
  }
  // Technically: _momentum *= -1;
  proposed_energy = _log_posterior(data, _mu_star) + _potential_energy();
  R = std::exp(proposed_energy - _initial_energy);
  if (isnan(R))  R = 0;
  if (Uniform(dualres::internals::_RNG_) < R) {
    _mu = _mu_star;
    _total_energy = proposed_energy;
  }
  // __real_mu = af::moddims(af::real(_mu), _lDim);
  _real_mu = dualres::nullary_index(_mu.matrix().real(), __lambda_grid_indices);
  // ^^ _real_mu must be set before _update_sigma_sq_inv()
  _update_sigma_sq_inv(data);
  return std::min((scalar_type)1.0, R);
};






template< typename T >
void dualres::MultiResParameters<T>::_compute_mass_matrix_eigen_values() {
  // _lambda_mass = (-1 / (_lambda * _sigma_sq_inv[0] + 1) + 1) /
  //   _sigma_sq_inv[0];
  _lambda_mass = 1 / (_lambda + 1 / _sigma_sq_inv[0]);
};




template< typename T >
void dualres::MultiResParameters<T>::_initialize_mu(
  const typename dualres::MultiResParameters<T>::scalar_type k
) {
  // Temporary debug version
  // _mu = af::constant(0, _lambda.dims(), _ctype);
  _mu = ComplexArrayType(_lambda.size());
  for (int i = 0; i < _mu.size(); i++) {
    _mu[i] = complex_type(__Standard_Gaussian(dualres::internals::_RNG_),
			  __Standard_Gaussian(dualres::internals::_RNG_));
  }
  fftwf_execute_dft(__backward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_mu.data()),
    reinterpret_cast<fftwf_complex*>(_mu.data())
  );
  _mu *= _lambda.sqrt() / (k * _mu.size());
  _low_rank_adjust(_mu);
  fftwf_execute_dft(__forward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_mu.data()),
    reinterpret_cast<fftwf_complex*>(_mu.data())
  );
  _real_mu = dualres::nullary_index(_mu.matrix().real(), __lambda_grid_indices);
};




template< typename T >
void dualres::MultiResParameters<T>::_initialize_sigma(
  const typename dualres::MultiResParameters<T>::scalar_type shape,
  const typename dualres::MultiResParameters<T>::scalar_type rate
) {
  if (_n_datasets <= 0)
    throw std::logic_error("MultiResParameters: n_datasets not set");
  // boost::math::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  std::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  std::uniform_real_distribution<scalar_type> Uniform(0, 1);
  _sigma_sq_inv.resize(_n_datasets);
  _sigma_sq_inv[0] = Gamma(dualres::internals::_RNG_);
  for (int i = 1; i < _n_datasets; i++)
    _sigma_sq_inv[i] = _sigma_sq_inv[0] / Uniform(dualres::internals::_RNG_);
};




template< typename T >
void dualres::MultiResParameters<T>::_sample_momentum() {
  _initial_energy = 0;
  for (int i = 0; i < _momentum.size(); i++) {
    _momentum[i] = complex_type(__Standard_Gaussian(dualres::internals::_RNG_),
				__Standard_Gaussian(dualres::internals::_RNG_));
    _initial_energy += (std::conj(_momentum[i]) * _momentum[i]).real();
  }
  _initial_energy *= -0.5;
  fftwf_execute_dft(__backward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_momentum.data()),
    reinterpret_cast<fftwf_complex*>(_momentum.data())
  );
  _momentum /= _lambda_mass.sqrt() * _momentum.size();
  _low_rank_adjust(_momentum);
  fftwf_execute_dft(__forward_fft_plan,
    reinterpret_cast<fftwf_complex*>(_momentum.data()),
    reinterpret_cast<fftwf_complex*>(_momentum.data())
  );
};




template< typename T >
void dualres::MultiResParameters<T>::_update_sigma_sq_inv(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  scalar_type shape = 0.5 * data.Y(0).size() + 1;
  scalar_type rate = 0.5 * (data.Y(0) - __real_mu).squaredNorm();
  // New update scheme: just sample _sigma_sq_inv's from gamma's w/out worrying
  // about truncation
  std::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  _sigma_sq_inv[0] = Gamma(dualres::internals::_RNG_);

  if (_n_datasets == 2) {
    shape = 0.5 * data.Y(1).size() + 1;
    rate = 0.5 * (data.Y(1) - data.W(0) * __real_mu).squaredNorm();
    Gamma = std::gamma_distribution<scalar_type>(shape, 1 / rate);
    _sigma_sq_inv[1] = Gamma(dualres::internals::_RNG_);
  }
  else if (_n_datasets != 1) {
    throw std::logic_error("Parameter updates not implemented for > 2 datasets");
  }

  // Old sampling scheme (truncated gamma's):
  // if (_n_datasets == 1) {
  //   std::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  //   _sigma_sq_inv[0] = Gamma(dualres::_RNG_);
  // }
  // else if (_n_datasets == 2) {
  //   boost::math::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  //   _sigma_sq_inv[0] = sample_truncated_gamma(Gamma, 0.0, _sigma_sq_inv[1]);
  //   shape = 0.5 * data.Y(1).elements() + 1;
  //   rate = 0.5 * af::sum<scalar_type>(af::pow(data.Y(1) - af::matmul(data.W(0), mu()), 2));
  //   Gamma = boost::math::gamma_distribution<scalar_type>(shape, 1 / rate);
  //   _sigma_sq_inv[1] = sample_truncated_gamma(Gamma, _sigma_sq_inv[0]);
  // }
  // else {
  //   throw std::logic_error("Parameter updates not implemented for > 2 datasets");
  // }
};









#endif  // _DUALRES_MULTI_RES_PARAMETERS_2_

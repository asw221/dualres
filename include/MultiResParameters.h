
#include <algorithm>
#include <arrayfire.h>
#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <math.h>
#include <random>
#include <vector>

#include "circulant_base.h"
#include "defines.h"
#include "MultiResData.h"



#ifndef _DUALRES_MULTI_RES_PARAMETERS_
#define _DUALRES_MULTI_RES_PARAMETERS_


namespace dualres {

    

  template< typename T >
  class MultiResParameters {
  public:
    typedef T scalar_type;
    

    MultiResParameters(
      const int n_datasets,
      const typename std::vector<scalar_type> &kernel_parameters,
      const Eigen::MatrixXi &ijk_y0,
      const dualres::qform_type &Qform_y0,
      const dualres::use_lambda_method select = dualres::use_lambda_method::EXTENDED
   );


    ::af_dtype type() const;
    ::af_dtype complex_type() const;
    
    void print_test();
    
    scalar_type update(
      const typename dualres::MultiResData<scalar_type> &data,
      const scalar_type eps = 0.1,
      const int L = 10
    );
    
    const af::array& gradient(const typename dualres::MultiResData<scalar_type> &data);
    af::array mu() const;
    
    // const af::array& lambda() const;
    const af::array& lambda() const;
    const af::array& lambda_inv() const;
    const af::array& indices() const;
    
    scalar_type log_prior() const;  // returns array of dim = 1
    scalar_type log_likelihood(const typename dualres::MultiResData<scalar_type> &data);
    // ^^ returns array of dim = 1
    scalar_type log_posterior(const typename dualres::MultiResData<scalar_type> &data);
    scalar_type sigma(const int which) const;


    
  private:
    ::af_dtype _dtype;
    ::af_dtype _ctype;

    af::array __grad;
    af::array __real_mu;
    af::array __real_mu_temp;
    af::array __mu_star;
    // af::array __Fh_x;
    
    af::array _binary_data_positions;
    af::array _lambda;
    // af::array _lambda_inv;
    af::array _lambda_mass;
    af::array _momentum;
    af::array _mu;
    af::array _positive_eigen_values;
    af::array _y0_indices;

    af::dim4 _lDim;

    scalar_type _total_energy;
    scalar_type _initial_energy;
    
    int _n_datasets;

    typename std::vector<scalar_type> _theta;
    typename std::vector<scalar_type> _sigma_sq_inv;

    const af::array& _gradient(
      const typename dualres::MultiResData<scalar_type> &data,
      const af::array &mu_star
    );

    void _compute_mass_matrix_eigen_values();
    void _initialize_mu(const scalar_type k = 10);
    void _initialize_sigma(const scalar_type shape = 1, const scalar_type rate = 1);
    void _sample_momentum();
    void _update_sigma_sq_inv(const typename dualres::MultiResData<scalar_type> &data);

    scalar_type _log_prior(const af::array &mu_star) const;
    scalar_type _potential_energy() const;
    
    scalar_type _log_likelihood(
      const typename dualres::MultiResData<scalar_type> &data,
      const af::array &mu_star
    );
    scalar_type _log_posterior(
      const typename dualres::MultiResData<scalar_type> &data,
      const af::array &mu_star
    );
  };

  
}








template< typename T >
const af::array& dualres::MultiResParameters<T>::_gradient(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data,
  const af::array &mu_star
) {
#ifndef NDEBUG
  if (data.n_datasets() < _n_datasets)
    throw std::logic_error("Insufficient data for parameters");
#endif
  __grad = -af::dft(af::idft(mu_star) / _lambda * _positive_eigen_values / mu_star.elements());
  __real_mu_temp = af::moddims(af::real(mu_star), _lDim);
  if (_n_datasets == 1) {
    __grad += _sigma_sq_inv[0] * (data.Y(0) - __real_mu_temp) * _binary_data_positions;
  }
  else if (_n_datasets == 2) {
    __grad += _sigma_sq_inv[0] * (data.Y(0) - __real_mu_temp) * _binary_data_positions +
      _sigma_sq_inv[1] *
      af::matmul(data.W(0), (data.Y(1) - af::matmul(data.W(0), __real_mu_temp)),
		 af_mat_prop::AF_MAT_TRANS);
  }
  else {
    throw std::logic_error("Gradient only implemented for single or dual resolution");
  }
  return __grad;
};





template< typename T >
const af::array& dualres::MultiResParameters<T>::gradient(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  return _gradient(data, _mu);
};



template< typename T >
af::array dualres::MultiResParameters<T>::mu() const {
  return __real_mu(_y0_indices);
};




template< typename T >
const af::array& dualres::MultiResParameters<T>::lambda() const {
  return _lambda;
};


template< typename T >
const af::array& dualres::MultiResParameters<T>::lambda_inv() const {
  return 1 / _lambda;
};


template< typename T >
const af::array& dualres::MultiResParameters<T>::indices() const {
  return _y0_indices;
};





template< typename T >
typename dualres::MultiResParameters<T>::scalar_type
dualres::MultiResParameters<T>::_log_prior(const af::array &mu_star) const {
  // __Fh_x = af::idft(mu_star) / std::sqrt((scalar_type)mu_star.elements());
  // scalar_type __lp = -0.5 *
  //   af::sum<scalar_type>(af::real(af::conjg(__Fh_x) / _lambda
  // 				  * _positive_eigen_values * __Fh_x));
  const scalar_type __lp = -0.5 * af::sum<scalar_type>(af::real(
    af::conjg(af::idft(mu_star)) / _lambda * _positive_eigen_values * af::idft(mu_star)
    )) / mu_star.elements();
  return __lp;
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
  const af::array &mu_star
) {
#ifndef NDEBUG
  if (data.n_datasets() < _n_datasets)
    throw std::logic_error("Insufficient data for parameters");
#endif
  __real_mu_temp = af::moddims(af::real(mu_star), _lDim);
  scalar_type __ll = -0.5 * _sigma_sq_inv[0] *
    af::sum<scalar_type>((data.Y(0) - __real_mu_temp) * (data.Y(0) - __real_mu_temp) *
			 _binary_data_positions);
  if (_n_datasets == 2)
    __ll += -0.5 * _sigma_sq_inv[1] *
      af::sum<scalar_type>(af::pow(data.Y(1) - af::matmul(data.W(0), __real_mu_temp), 2.0));
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
  const af::array &mu_star
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
  const scalar_type __pe = -0.5 * af::sum<scalar_type>(af::real(
    af::conjg(af::idft(_momentum)) * _lambda_mass * _positive_eigen_values * af::idft(_momentum)
    )) / _momentum.elements();
  return __pe;
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





template< typename T >
::af_dtype dualres::MultiResParameters<T>::type() const {
  return _dtype;
};

template< typename T >
::af_dtype dualres::MultiResParameters<T>::complex_type() const {
  return _ctype;
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
//   print_array_summary(af::real(_mu(_y0_indices)));
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
  __mu_star = _mu.copy();
  _compute_mass_matrix_eigen_values();
  _sample_momentum();
  _initial_energy += _log_posterior(data, _mu);  // _sample_momentum() initilizes energy
  _momentum += k * eps * _gradient(data, __mu_star);
  for (int step = 0; step < L; step++) {
    k = (step == (L - 1)) ? 0.5 : 1;
    __mu_star += eps * af::dft(_lambda_mass * af::idft(_momentum) /
			       _momentum.elements());
    _momentum += k * eps * _gradient(data, __mu_star);
  }
  proposed_energy = _log_posterior(data, __mu_star) + _potential_energy();
  R = std::exp(proposed_energy - _initial_energy);
  if (isnan(R))  R = 0;
  if (Uniform(dualres::internals::_RNG_) < R) {
    _mu = __mu_star.copy();
    _total_energy = proposed_energy;
  }
  __real_mu = af::moddims(af::real(_mu), _lDim);
  // ^^ __real_mu must be set before _update_sigma_sq_inv()
  _update_sigma_sq_inv(data);
  return std::min((scalar_type)1.0, R);
};






template< typename T >
void dualres::MultiResParameters<T>::_compute_mass_matrix_eigen_values() {
  _lambda_mass = (-1 / (_lambda * _sigma_sq_inv[0] + 1) + 1) /
    _sigma_sq_inv[0];
};




template< typename T >
void dualres::MultiResParameters<T>::_initialize_mu(
  const typename dualres::MultiResParameters<T>::scalar_type k
) {
  // Temporary debug version
  // _mu = af::constant(0, _lambda.dims(), _ctype);
  _mu = af::dft(af::sqrt(_lambda) * _positive_eigen_values *
  		af::idft(af::randn(_lambda.dims(), _ctype, dualres::internals::_AF_RNG_)) /
  		(k * _lambda.elements()));
  __real_mu = af::moddims(af::real(_mu), _lDim);
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
  // _sigma_sq_inv[i] = sample_truncated_gamma(Gamma, 0.0, _sigma_sq_inv[0]);

  // // Temporary debug version
  // _sigma_sq_inv[0] = 1;
  // for (int i = 1; i < _n_datasets; i++)
  //   _sigma_sq_inv[i] = _sigma_sq_inv[i-1] * 2;
};




template< typename T >
void dualres::MultiResParameters<T>::_sample_momentum() {
  _momentum = af::randn(_lambda.dims(), _ctype, dualres::internals::_AF_RNG_);
  _initial_energy = -0.5 * af::sum<scalar_type>(af::real(af::conjg(_momentum) * _momentum));
  _momentum = af::dft(af::idft(_momentum) / af::sqrt(_lambda_mass) *
		      _positive_eigen_values / _momentum.elements());
};




template< typename T >
void dualres::MultiResParameters<T>::_update_sigma_sq_inv(
  const typename dualres::MultiResData<
    typename dualres::MultiResParameters<T>::scalar_type> &data
) {
  scalar_type shape = 0.5 * _y0_indices.elements() + 1;
  scalar_type rate = 0.5 * af::sum<scalar_type>(
    (data.Y(0) - __real_mu) * (data.Y(0) - __real_mu) * _binary_data_positions);
  // New update scheme: just sample _sigma_sq_inv's from gamma's w/out worrying
  // about truncation
  std::gamma_distribution<scalar_type> Gamma(shape, 1 / rate);
  _sigma_sq_inv[0] = Gamma(dualres::internals::_RNG_);

  if (_n_datasets == 2) {
    shape = 0.5 * data.Y(1).elements() + 1;
    rate = 0.5 * af::sum<scalar_type>(af::pow(data.Y(1) - af::matmul(data.W(0), __real_mu), 2));
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







template< typename T >
dualres::MultiResParameters<T>::MultiResParameters(
  const int n_datasets,
  const typename std::vector<
    typename dualres::MultiResParameters<T>::scalar_type> &kernel_parameters,
  const Eigen::MatrixXi &ijk_y0,
  const dualres::qform_type &Qform_y0,
  const dualres::use_lambda_method lambda_method
) {
  _dtype = dualres::data_types<scalar_type>::af_dtype;
  _ctype = dualres::data_types<scalar_type>::af_ctype;
  
  _n_datasets = n_datasets;
  _theta = std::vector<scalar_type>(kernel_parameters.cbegin(), kernel_parameters.cend());

  // Compute diagonal of eigen vector matrix
  Eigen::Vector3i grid_dims = ijk_y0.colwise().maxCoeff().head<3>();
  grid_dims += Eigen::Vector3i::Ones();
  // std::cout << grid_dims << std::endl;
  
  if (lambda_method == dualres::use_lambda_method::PROFILE) {
    _lambda = dualres::profile_and_compute_lambda_3d<scalar_type>(
      grid_dims, Qform_y0, _theta[1], _theta[2], true);
  }
  else {
    _lambda = af::dft(dualres::kernels::rbf(dualres::circulant_base_3d<scalar_type>(
      grid_dims, Qform_y0, (lambda_method == dualres::use_lambda_method::EXTENDED)),
      _theta[1], _theta[2]));
  }


  _lDim = af::dim4(_lambda.elements());  // must be set before _initialize_mu()
  _positive_eigen_values = 1 - af::sign(af::real(_lambda));
  _lambda *= _theta[0];
  // _lambda_inv = 1 / _lambda;
  // af::replace(_lambda_inv, af::real(_lambda) != 0, 0);
  _initialize_mu();
  _initialize_sigma();
  
  std::vector<int> indices(ijk_y0.rows());              //
  std::vector<scalar_type> bdp(_lambda.elements(), 0);  // _binary_data_positions
  for (int i = 0; i < ijk_y0.rows(); i++) {
    indices[i] = (_lambda.dims(1) * _lambda.dims(0)) * ijk_y0(i, 2) +
      _lambda.dims(0) * ijk_y0(i, 1) + ijk_y0(i, 0);
    bdp[indices[i]] = 1;
  }
  _binary_data_positions = af::array(bdp.size(), bdp.data());
  _y0_indices = af::array(indices.size(), indices.data());
};





#endif  // _DUALRES_MULTI_RES_PARAMETERS_

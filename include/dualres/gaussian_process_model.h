
#include <chrono>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>


#include "dualres/defines.h"
#include "dualres/HMCParameters.h"
#include "dualres/MultiResData.h"
#include "dualres/MultiResParameters.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/utilities.h"



#ifndef _DUALRES_GAUSSIAN_PROCESS_MODEL_
#define _DUALRES_GAUSSIAN_PROCESS_MODEL_



namespace dualres {



  namespace gaussian_process {


    namespace sor_approx {

      
      template< typename T >
      class mcmc_mode {
      public:
	typedef std::true_type is_mcmc_type;
	typedef T scalar_type;
	typedef std::chrono::milliseconds milliseconds;
	typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> VectorType;

	mcmc_mode(const int size_mu, const int size_sigma);
	void update(
          const VectorType &mu,
	  const std::vector<scalar_type> &sigma,
	  const scalar_type &log_posterior
        );

	template< typename DurationType >
	void sampling_time(const DurationType &duration);

	int samples() const;
	std::vector<scalar_type> mode_sigma() const;
	VectorType mode_mu() const;
	VectorType var_mu() const;
	scalar_type sampling_time() const;

      private:
	int _samples;
	scalar_type _log_posterior;
	scalar_type _sampling_time;
	VectorType _first_moment_mu;
	VectorType _second_moment_mu;
	std::vector<scalar_type> _sigma;
      };

  

  

      template< typename T, typename OStream >
      dualres::gaussian_process::sor_approx::mcmc_mode<T> fit_model(
        const nifti_image* const _high_res_,
        dualres::MultiResData<T> &_data_,
	dualres::HMCParameters<T> &_hmc_,
	OStream &_output_stream_
      ) {
	typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> VectorType;
	
	dualres::MultiResParameters<T> _theta_(
          _data_.n_datasets(), _data_.kernel_parameters(),
	  dualres::get_nonzero_indices_bounded(_high_res_),
	  dualres::qform_matrix(_high_res_),
	  dualres::use_lambda_method::EXTENDED
        );

	T mh_rate, log_posterior;
	int save_count = 0;
	dualres::gaussian_process::sor_approx::mcmc_mode<T> posterior_summary(
          _theta_.mu().size(), _data_.n_datasets());
	VectorType mu(_theta_.mu().size());
	std::vector<T> sigma(_data_.n_datasets());
	

	std::cout << "\nFitting model with rmHMC and subset of regressors approximation:"
		  << std::endl;
	dualres::utilities::progress_bar pb(_hmc_.max_iterations());
	
	auto start = std::chrono::high_resolution_clock::now();
	while (save_count < _hmc_.n_save() && _hmc_.iteration() <= _hmc_.max_iterations()) {
	  mh_rate = _theta_.update(_data_, _hmc_.eps(), _hmc_.integrator_steps());
	  _hmc_.update(mh_rate);
	  if (_hmc_.save_iteration()) {
	    if (save_count == 0)
	      start = std::chrono::high_resolution_clock::now();
	    // save stuff ...
	    mu = _theta_.mu();
	    _output_stream_ << mu.transpose() << "  ";
	    for (int i = 0; i < _data_.n_datasets(); i++) {
	      sigma[i] = _theta_.sigma(i);
	      _output_stream_ << sigma[i] << "  ";
	    }
	    log_posterior = _theta_.log_posterior(_data_);
	    _output_stream_ << log_posterior;
	    if (save_count < (_hmc_.n_save() - 1))
	      _output_stream_ << std::endl;

	    posterior_summary.update(mu, sigma, log_posterior);
	    save_count++;
	  }
	  pb++;
	  std::cout << pb;
	}
	pb.finish();
	const auto stop = std::chrono::high_resolution_clock::now();
	posterior_summary.sampling_time(stop - start);
	std::cout << "Sampling took " << posterior_summary.sampling_time() << " (sec)"
		  << std::endl;
	std::cout << "Metropolis-Hastings rate was "
		  << (_hmc_.metropolis_hastings_rate() * 100)
		  << "%" << std::endl;
	return posterior_summary;
      };


      
    }  // namespace sor_approx
  }  // namespace gaussian_process
}  // namespace dualres







template< typename T >
dualres::gaussian_process::sor_approx::mcmc_mode<T>::mcmc_mode(
  const int size_mu,
  const int size_sigma
) {
  _samples = 0;
  _log_posterior = 0;
  _sampling_time = 0;
  _first_moment_mu = VectorType::Zero(size_mu);
  _second_moment_mu = VectorType::Zero(size_mu);
  _sigma = std::vector<scalar_type>(size_sigma, 0);	
};




template< typename T >
void dualres::gaussian_process::sor_approx::mcmc_mode<T>::update(
  const dualres::gaussian_process::sor_approx::mcmc_mode<T>::VectorType& mu,
  const std::vector<
    typename dualres::gaussian_process::sor_approx::mcmc_mode<T>::scalar_type>& sigma,
  const dualres::gaussian_process::sor_approx::mcmc_mode<T>::scalar_type& log_posterior
) {
  if (_first_moment_mu.size() != mu.size())
    throw std::domain_error("mu dimension mismatch");
  if (_sigma.size() != sigma.size())
    throw std::domain_error("sigma dimension mismatch");
  _first_moment_mu += mu;
  _second_moment_mu += (mu.array() * mu.array()).matrix();
  for (int i = 0; i < _sigma.size(); i++) {
    _sigma[i] += sigma[i];
  }
  _log_posterior += log_posterior;
  _samples++;
};





template< typename T >
template< typename DurationType >
void dualres::gaussian_process::sor_approx::mcmc_mode<T>::sampling_time(
  const DurationType &duration
) {
  _sampling_time = (scalar_type)std::chrono::duration_cast<milliseconds>
    (duration).count() / 1e3;
};



template< typename T >
int dualres::gaussian_process::sor_approx::mcmc_mode<T>::samples() const {
  return (_samples <= 0) ? 1 : _samples;
};




template< typename T >
std::vector<typename dualres::gaussian_process::sor_approx::mcmc_mode<T>::scalar_type>
dualres::gaussian_process::sor_approx::mcmc_mode<T>::mode_sigma() const {
  std::vector<scalar_type> __mode(_sigma.size());
  for (int i = 0; i < _sigma.size(); i++)
    __mode[i] = _sigma[i] / samples();
  return __mode;
};



template< typename T >
typename dualres::gaussian_process::sor_approx::mcmc_mode<T>::VectorType
dualres::gaussian_process::sor_approx::mcmc_mode<T>::mode_mu() const {
  return _first_moment_mu / samples();
};



template< typename T >
typename dualres::gaussian_process::sor_approx::mcmc_mode<T>::VectorType
dualres::gaussian_process::sor_approx::mcmc_mode<T>::var_mu() const {
  return ( _second_moment_mu -
	  (_first_moment_mu.array() * _first_moment_mu.array()).matrix() /
	  samples() ) / samples();
};


template< typename T >
typename dualres::gaussian_process::sor_approx::mcmc_mode<T>::scalar_type
dualres::gaussian_process::sor_approx::mcmc_mode<T>::sampling_time() const {
  return _sampling_time;
};



#endif  // _DUALRES_GAUSSIAN_PROCESS_MODEL_


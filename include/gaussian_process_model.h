
#include <chrono>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <vector>


#include "defines.h"
#include "HMCParameters.h"
#include "MultiResData.h"
#include "MultiResParameters.h"
#include "utilities.h"



#ifndef _DUALRES_GAUSSIAN_PROCESS_MODEL_
#define _DUALRES_GAUSSIAN_PROCESS_MODEL_



namespace dualres {



  template< typename T >
  class mcmc_output {
  public:
    typedef T value_type;
    typedef typename Eigen::Vector<value_type, Eigen::Dynamic> VectorType;
    
    std::vector<VectorType> mu;
    std::vector<std::vector<T> > sigma;
    std::vector<T> log_posterior;
    T sampling_time;
    
    mcmc_output(const int n) {
      if (n <= 0)
	throw std::domain_error("Must reserve space for > 0 MCMC samples");
      mu.reserve(n);
      sigma.reserve(n);
      log_posterior.reserve(n);
    };
  };

  

  

  template< typename T >
  void fit_gpm_with_sor_approximation(
    dualres::MultiResData<T> &_data_,
    dualres::MultiResParameters<T> &_theta_,
    dualres::HMCParameters<T> &_hmc_
  ) {
    T mh_rate;
    int save_count = 0;
    std::vector<T> _log_posterior_(_hmc_.n_save());

    //
    std::ofstream csv("test/hmc_test.csv");
    if (!csv)
      throw std::logic_error("Not run from directory with 'test' sub-dir");
    //
    
      // std::vector<T> _sigma_(_data_.n_datasets());
    dualres::utilities::progress_bar pb(_hmc_.max_iterations());
    // std::cout << "Fitting model with HMC" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while (save_count < _hmc_.n_save() && _hmc_.iteration() <= _hmc_.max_iterations()) {
      mh_rate = _theta_.update(_data_, _hmc_.eps(), _hmc_.integrator_steps());
      _hmc_.update(mh_rate);
      if (_hmc_.save_iteration()) {
	if (save_count == 0)
	  start = std::chrono::high_resolution_clock::now();
	// save stuff ...
	//  (i)   (semi-)random voxels
	//  (ii)  log-posterior
	//  (iii) estimate n-th quantile of max|mu|
	//  (iv)  estimate of E mu
	//  (v)   estimate of var mu
	//  (vi)  confidence band given (iii)
	_mu_ += _theta_.mu();
	_log_posterior_[save_count] = _theta_.log_posterior(_data_);
	// for (int ii = 0; ii < _data_.n_datasets(); ii++)
	//   _sigma_[ii] = _theta_.sigma(ii);
	save_count++;
      }
      pb++;
      std::cout << pb;
    }
    pb.finish();
    _mu_ /= save_count;
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Sampling took " << ((double)duration.count() / 1000) << " (sec)"
	      << std::endl;
    std::cout << "Metropolis-Hastings rate was "
	      << (_hmc_.metropolis_hastings_rate() * 100)
	      << "%" << std::endl;
    _mu_.host(_mu_host_.data());

    //
    csv << "mu";
    for (int i = 0; i < _mu_host_.size(); i++)
      csv << "\n" <<_mu_host_[i];
    csv.close();
    // return _mu_host_;
  };
  
}


#endif  // _DUALRES_GAUSSIAN_PROCESS_MODEL_


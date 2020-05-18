
#include <algorithm>
// #include <limits>
#include <math.h>
#include <random>
#include <stdexcept>

#include "dualres/defines.h"


#ifndef _DUALRES_HMC_PARAMETERS_
#define _DUALRES_HMC_PARAMETERS_



namespace dualres {


  /*! Parameters for Hamiltonian Monte Carlo
   * 
   * Implements the dual-averaging method of
   * <a href="http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf">Hoffman and Gelman, 2014</a>
   * to automatically tune the step size, \c eps(). The value returned 
   * by \c eps() has an enforced lower bound set to the value of 
   * \c eps_min().
   */
  // -----------------------------------------------------------------

  template< typename RealType = float >
  class HMCParameters {
  public:
    typedef RealType value_type;

    HMCParameters(
      const int burnin = 1000,
      const int n_save = 500,
      const int thin = 1,
      const int integrator_steps = 10,
      const value_type starting_eps = 0.1,
      const value_type mh_target = 0.65,
      const value_type gamma = 0.05,
      const value_type t0 = 10,
      const value_type kappa = 0.75
    );


    bool save_iteration() const;
    
    int burnin_iterations() const;
    int integrator_steps() const;
    int iteration() const;
    int max_iterations() const;
    int n_save() const;
    int thin_iterations() const;
    
    value_type eps() const;        /*!< Return step size sampled around \c eps_value() */
    value_type eps_min() const;    /*!< Return enforced minimum value of \c eps() */
    value_type eps_value() const;  /*!< Return non-random step size value */
    value_type metropolis_hastings_rate() const;
    value_type path_length() const;

    void update(const value_type &mh_rate);
    void reset();
    
    
  private:
    bool _warmup;
    bool _eps_start_found;
    
    int _burnin;
    int _integrator_steps;
    int _iteration;
    int _n_save;
    int _thin;

    double _A;
    double _eps;
    double _eps0;            /*!< Minimum value of HMC step size. Set to 1e-3 */
    double _eps_bar;
    double _eps_target;
    double _gamma;
    double _h_bar;
    double _mh_running_sum;  /**/
    double _mh_target;
    double _t0;
    double _kappa;
    
  };

  
}





template< typename RealType >
dualres::HMCParameters<RealType>::HMCParameters(
  const int burnin,
  const int n_save,
  const int thin,
  const int integrator_steps,
  const dualres::HMCParameters<RealType>::value_type starting_eps,
  const dualres::HMCParameters<RealType>::value_type mh_target,
  const dualres::HMCParameters<RealType>::value_type gamma,
  const dualres::HMCParameters<RealType>::value_type t0,
  const dualres::HMCParameters<RealType>::value_type kappa
) {
  if (burnin < 0)
    throw std::domain_error("HMCParameters: burnin must be >= 0");
  if (n_save < 0)
    throw std::domain_error("HMCParameters: n_save must be >= 0");
  if (thin < 0)
    throw std::domain_error("HMCParameters: thin must be >= 0");
  if (integrator_steps <= 0)
    throw std::domain_error("HMCParameters: integrator_steps must be > 0");
  
  if (starting_eps <= 0)
    throw std::domain_error("HMCParameters: starting_eps must be > 0");
  if (mh_target <= 0 || mh_target >= 1)
    throw std::domain_error("HMCParameters: mh_target should be within (0, 1)");
  if (gamma <= 0 || gamma >= 1)
    throw std::domain_error("HMCParameters: gamma should be within (0, 1)");
  if (t0 < 0)
    throw std::domain_error("HMCParameters: t0 should be >= 0");
  if (kappa <= 0 || kappa >= 1)
    throw std::domain_error("HMCParameters: kappa should be within (0, 1)");
  
  _warmup = burnin > 0;
  _eps_start_found = !_warmup;
  
  _burnin = burnin;
  _integrator_steps = integrator_steps;
  _iteration = 0;
  _n_save = n_save;
  _thin = (thin == 0) ? 1 : thin;

  _A = 1;
  // _eps0 = (double)std::numeric_limits<value_type>::epsilon();
  _eps0 = 1e-3;
  _eps = std::max((double)starting_eps, _eps0);
  _eps_bar = 1;
  _gamma = gamma;
  _h_bar = 0;
  _mh_running_sum = 0;
  _mh_target = mh_target;
  _t0 = t0;
  _kappa = kappa;
};








template< typename RealType > 
bool dualres::HMCParameters<RealType>::save_iteration() const {
  return (_iteration > _burnin) && ((_iteration - _burnin) % _thin == 0);
};



template< typename RealType > 
int dualres::HMCParameters<RealType>::burnin_iterations() const {
  return _burnin;
};



template< typename RealType > 
int dualres::HMCParameters<RealType>::integrator_steps() const {
  // if (!_eps_start_found)
  //   return 1;
  // else
  //   return _integrator_steps;
  return _eps_start_found ? _integrator_steps : 1;
};



template< typename RealType > 
int dualres::HMCParameters<RealType>::iteration() const {
  return _iteration;
};




template< typename RealType > 
int dualres::HMCParameters<RealType>::max_iterations() const {
  return _burnin + _thin * _n_save;
};



template< typename RealType > 
int dualres::HMCParameters<RealType>::n_save() const {
  return _n_save;
};



template< typename RealType > 
int dualres::HMCParameters<RealType>::thin_iterations() const {
  return _thin;
};



template< typename RealType > 
typename dualres::HMCParameters<RealType>::value_type
dualres::HMCParameters<RealType>::eps() const {
  static std::uniform_real_distribution<> Uniform(0.9, 1.1);
  if (_warmup)
    return static_cast<value_type>(_eps);
  else
    return static_cast<value_type>(
      std::max(_eps * Uniform(dualres::__internals::_RNG_), _eps0));
};



template< typename RealType > 
typename dualres::HMCParameters<RealType>::value_type
dualres::HMCParameters<RealType>::eps_min() const {
  return static_cast<value_type>(_eps0);
};



template< typename RealType > 
typename dualres::HMCParameters<RealType>::value_type
dualres::HMCParameters<RealType>::eps_value() const {
  return static_cast<value_type>(_eps);
};



    
template< typename RealType > 
typename dualres::HMCParameters<RealType>::value_type
dualres::HMCParameters<RealType>::metropolis_hastings_rate() const {
  return static_cast<value_type>(_mh_running_sum /
				 std::max(int(_iteration - _burnin), 1));
};



template< typename RealType > 
typename dualres::HMCParameters<RealType>::value_type
dualres::HMCParameters<RealType>::path_length() const {
  return static_cast<value_type>(_eps * _integrator_steps);
};



template< typename RealType > 
void dualres::HMCParameters<RealType>::update(
  const dualres::HMCParameters<RealType>::value_type &mh_rate
) {
  if (_iteration == _burnin) {
    _warmup = false;
    _eps_start_found = true;
    if (_eps_bar > _eps0)
      _eps = _eps_bar;
  }
  if (_warmup) {
    if (!_eps_start_found) {
      if (_iteration == 0)
	_A = 2 * int(mh_rate > 0.5) - 1;
      if (std::pow(mh_rate, _A) > std::pow(2.0, -_A)) {
	_eps *= std::pow(2.0, _A);
      }
      else {
	_eps_start_found = true;
	_eps_target = std::log(10.0 * _eps);
      }
    }
    else {
      _h_bar = (1 - 1 / (_iteration + _t0)) * _h_bar +
	(_mh_target - mh_rate) / (_iteration + _t0);
      _eps = std::exp(_eps_target - std::sqrt(static_cast<value_type>(_iteration)) *
		      _h_bar / _gamma);
      _eps = std::max(_eps, _eps0);
      _eps_bar = std::exp(std::pow(static_cast<value_type>(_iteration), -_kappa) *
			  std::log(_eps) +
			  (1 - std::pow(static_cast<value_type>(_iteration), -_kappa)) *
			   std::log(_eps_bar));
    }
  }
  else {
    _mh_running_sum += mh_rate;
  }
  _iteration++;
};




template< typename RealType >
void dualres::HMCParameters<RealType>::reset() {
  _iteration = 0;
  _mh_running_sum = 0;
};



#endif  // _DUALRES_HMC_PARAMETERS_

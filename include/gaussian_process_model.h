
#include <arrayfire.h>
#include <iostream>


#include "HMCParameters.h"
#include "MultiResData.h"
#include "MultiResParameters.h"
#include "utilities.h"



#ifndef _DUALRES_GAUSSIAN_PROCESS_MODEL_
#define _DUALRES_GAUSSIAN_PROCESS_MODEL_



namespace dualres {

  template< typename T >
  void fit_dualres_gaussian_process_model(
    dualres::MultiResData<T> &_data_,
    dualres::MultiResParameters<T> &_theta_,
    dualres::HMCParameters<T> &_hmc_
  ) {
    T mh_rate;
    int save_count = 0;
    dualres::utilities::progress_bar pb(_hmc_.max_iterations());
    std::cout << "Fitting model with HMC" << std::endl;
    while (save_count < _hmc_.n_save() && _hmc_.iteration() <= _hmc_.max_iterations()) {
      mh_rate = _theta_.update(_data_, _hmc_.eps(), _hmc_.integrator_steps());
      _hmc_.update(mh_rate);
      if (_hmc_.save_iteration()) {
	// save stuff ...
	save_count++;
      }
      pb++;
      std::cout << pb;
    }
    pb.finish();
  };
  
}


#endif  // _DUALRES_GAUSSIAN_PROCESS_MODEL_


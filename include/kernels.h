
#include <arrayfire.h>
#include <cmath>


#ifndef _DUALRES_KERNELS_
#define _DUALRES_KERNELS_


namespace dualres {

  

  template< typename T >
  T inverse_logistic_function(const T &x, const T &max_val = 1.0);

  
  template< typename T >
  T inverse_logistic_gradient(const T &x, const T &max_val = 1.0);
    
  
  template< typename T >
  T logistic_function(const T &x);

  
  template< typename T >
  T logistic_gradient(const T &x);


  

  namespace kernels {

    // scalar version
    template< typename T >
    T rbf(
      const T &distance,
      const T &bandwidth,
      const T &exponent = 1.9999,
      const T &variance = 1.0
    ) {
      return variance * std::exp(-bandwidth * std::pow(std::abs(distance), exponent));
    };



    template< typename T >
    T rbf_inverse(
      const T &rho,
      const T &bandwidth,
      const T &exponent = 1.9999,
      const T &variance = 1.0
    ) {
      if (rho <= (T)0 || rho >= (T)1)
	throw std::logic_error("rbf_inverse: inverse kernel only for rho between (0, 1)");
      return std::pow(-std::log(rho / variance) / bandwidth, 1 / exponent);
    };

    

    template< typename T >
    af::array rbf(
      const af::array &distance,
      const T &bandwidth,
      const T &exponent = 1.9999,
      const T &marginal_variance = 1.0
    );


    af::array rbf(const af::array &distance, const af::array &theta);
    

    template< typename T >
    af::array rbf_transformed(
      const af::array &distance,
      const af::array &trans,
      const T &tau_max
    );

    
    template< typename T >
    af::array rbf_inverse_transform_parameters(
      const af::array &trans,
      const T &tau_max
    );

    
    template< typename T >
    af::array rbf_transform_parameters(const af::array &theta, const T &tau_max);
    
    
  };
  
}



#include "kernels.inl"

#endif  // _DUALRES_KERNELS_


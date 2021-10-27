

#ifndef _DUALRES_KERNELS_
#define _DUALRES_KERNELS_


namespace dualres {
  

  namespace kernels {
    /*! @addtogroup GaussianProcessModels 
     * @{
     */

    // scalar version
    template< typename T >
    T rbf(
      const T distance,
      const T bandwidth,
      const T exponent = 1.9999,
      const T variance = 1.0
    );


    template< typename T >
    T rbf_inverse(
      const T rho,
      const T bandwidth,
      const T exponent = 1.9999,
      const T variance = 1.0
    );


    template< typename T >
    T rbf_bandwidth_to_fwhm(const T bandwidth, const T exponent = 1.9999);
    
    template< typename T >
    T rbf_fwhm_to_bandwidth(const T fwhm, const T exponent = 1.9999);


    template< typename T >
    T rational_quadratic(
      const T val,
      const T psi,
      const T nu = 1,
      const T variance = 1
    );

    /*! @} */
  }  // namespace kernels 
  
}



#include "dualres/impl/kernels.inl"

#endif  // _DUALRES_KERNELS_




// Recycling

  // template< typename T >
  // T inverse_logistic_function(const T &x, const T &max_val = 1.0);

  
  // template< typename T >
  // T inverse_logistic_gradient(const T &x, const T &max_val = 1.0);
    
  
  // template< typename T >
  // T logistic_function(const T &x);

  
  // template< typename T >
  // T logistic_gradient(const T &x);



    // template< typename T >
    // af::array rbf(
    //   const af::array &distance,
    //   const T &bandwidth,
    //   const T &exponent = 1.9999,
    //   const T &marginal_variance = 1.0
    // );


    // af::array rbf(const af::array &distance, const af::array &theta);
    

    // template< typename T >
    // af::array rbf_transformed(
    //   const af::array &distance,
    //   const af::array &trans,
    //   const T &tau_max
    // );

    
    // template< typename T >
    // af::array rbf_inverse_transform_parameters(
    //   const af::array &trans,
    //   const T &tau_max
    // );

    
    // template< typename T >
    // af::array rbf_transform_parameters(const af::array &theta, const T &tau_max);
    

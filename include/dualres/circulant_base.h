
#include <Eigen/Core>

#include "dualres/nifti_manipulation.h"


#ifndef _DUALRES_CIRCULANT_BASE_
#define _DUALRES_CIRCULANT_BASE_



namespace dualres {
  /*! @addtogroup GaussianProcessModels
   * @{
   */


  struct lambda_profile {
    double compute_time;
    int negative_values;
    int size;
  };

  template< typename OStream >
  OStream& operator<<(OStream& os, const lambda_profile& lp);
  


  template< typename scalar_type >
  Eigen::Array<scalar_type, Eigen::Dynamic, 1> circulant_base_3d(
    const Eigen::Vector3i &data_grid_dims,
    const dualres::qform_type &Qform,  // 
    const bool use_nearest_power_2 = true
  );


  /*! @} */
}


#include "dualres/impl/circulant_base.inl"

#endif  // _DUALRES_CIRCULANT_BASE_












// Recycling



  // template< typename scalar_type >
  // dualres::lambda_profile profile_lambda_computation_3d(  // return elapsed time per dft
  //   const Eigen::Vector3i &data_grid_dims,
  //   const dualres::qform_type &Qform,
  //   const bool use_nearest_power_2 = true,
  //   const scalar_type kernel_bandwidth = 1.0,
  //   const scalar_type kernel_exponent = 1.999
  // );

  
  // template< typename scalar_type >
  // af::array profile_and_compute_lambda_3d(  // return elapsed time per dft
  //   const Eigen::Vector3i &data_grid_dims,
  //   const dualres::qform_type &Qform,
  //   const scalar_type kernel_bandwidth = 1.0,
  //   const scalar_type kernel_exponent = 1.999,
  //   const bool verbose = false
  // );





  

  

  // template< typename scalar_type >
  // af::array circulant_base_3d_af(
  //   const Eigen::Vector3i &data_grid_dims,
  //   const dualres::qform_type &Qform,  // 
  //   const bool use_nearest_power_2 = true
  // ) {
  //   const int D = 3;  // 3d grid
  //   std::vector<int> grid_dims(D);  // dimensions of (extended) data grid
  //   std::vector<int> base_dims(D);  // dimensions of circulant matrix base
  //   int i = 0, j = 0, k = 0, ind = 0, base_len = 1;
  //   //
  //   // Eigen::Vector4d pos0;
  //   // pos0 << 1, 1, 1, 0;
  //   Eigen::Vector4f pos = Eigen::Vector4f::Zero();
  //   //
  //   // Allocate grid_dims and base_dims
  //   for (int ll = 0; ll < D; ll++) {
  //     if (data_grid_dims[ll] <= 1)
  // 	throw std::logic_error("circulant_base_3d: Cannot have dimensions <= 1");
  //     base_dims[ll] = 2 * (data_grid_dims[ll] - 1);
  //     if (use_nearest_power_2) {
  // 	base_dims[ll] = (int)std::pow(2.0,
  // 				      std::ceil(std::log2((double)base_dims[ll])));
  // 	grid_dims[ll] = base_dims[ll] / 2 + 1;
  //     }
  //     else {
  // 	grid_dims[ll] = data_grid_dims[ll];
  //     }
  //     base_len *= base_dims[ll];
  //   }
  //   // Allocate and fill circulant base: Row-Major order
  //   std::vector<scalar_type> base(base_len);
  //   // Eigen::Array<scalar_type, Eigen::Dynamic, 1> base(base_len);
  //   for (int ll = 0; ll < base_dims[0]; ll++) {
  //     j = 0;
  //     if (ll < grid_dims[0]) ++i; else --i;
  //     pos(0) = static_cast<float>(i - 1);
  //     for (int mm = 0; mm < base_dims[1]; mm++) {
  // 	k = 0;
  // 	if (mm < grid_dims[1]) ++j; else --j;
  // 	pos(1) = static_cast<float>(j - 1);
  // 	for (int nn = 0; nn < base_dims[2]; nn++) {
  // 	  if (nn < grid_dims[2]) ++k; else --k;
  // 	  pos(2) = static_cast<float>(k - 1);
  // 	  // base[ind] = std::sqrt((affine * (pos - pos0)).head(d).squaredNorm());
  // 	  base[ind] = static_cast<scalar_type>((Qform * pos).head<3>().norm());
  // 	  ind++;
  // 	}
  //     }
  //   }
  //   return af::array(base_dims[0], base_dims[1], base_dims[2], base.data());
  // };






#include <cmath>
#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <math.h>
#include <vector>

#include "dualres/defines.h"
#include "dualres/nifti_manipulation.h"




template< typename OStream >
OStream& dualres::operator<<(OStream& os, const dualres::lambda_profile& lp) {
  os << "Circulant matrix with base of size " << lp.size << " elements:\n"
     << "  Decomposition took " << lp.compute_time << "(sec)\n"
     << "  And returned " << lp.negative_values << " eigen values < 0 ("
     << ((double)lp.negative_values / lp.size * 100) << "%)\n";
  return os;
};





template< typename scalar_type >
Eigen::Array<scalar_type, Eigen::Dynamic, 1> dualres::circulant_base_3d(
  const Eigen::Vector3i &data_grid_dims,
  const dualres::qform_type &Qform,  // 
  const bool use_nearest_power_2
) {
  typedef typename Eigen::Array<scalar_type, Eigen::Dynamic, 1> ArrayType;
  const int D = 3;  // 3d grid
  Eigen::Vector4f ijk0 = Eigen::Vector4f::Zero();
  float j, k;
  int base_len = 1;
  int ll, mm, nn;
  std::vector<int> grid_dims(D);  // dimensions of (extended) data grid
  std::vector<int> base_dims(D);  // dimensions of circulant matrix base
  // Allocate grid_dims and base_dims
  for (int ll = 0; ll < D; ll++) {
    if (data_grid_dims[ll] <= 1)
      throw std::logic_error("circulant_base_3d: Cannot have dimensions <= 1");
    base_dims[ll] = 2 * (data_grid_dims[ll] - 1);
    if (use_nearest_power_2) {
      base_dims[ll] = (int)std::pow(2.0,
        std::ceil(std::log2((double)base_dims[ll])));
      grid_dims[ll] = base_dims[ll] / 2 + 1;
    }
    else {
      grid_dims[ll] = data_grid_dims[ll];
    }
    base_len *= base_dims[ll];
  }

  // Allocate and fill circulant base: Column Major order
  float i = 0;
  std::vector<float> dim0_seq(base_dims[0]);
  for (int ll = 0; ll < base_dims[0]; ll++) {
    if (ll < grid_dims[0]) ++i; else --i;
    dim0_seq[ll] = i - 1;
  }
    
  // Eigen::Array<scalar_type, Eigen::Dynamic, 1> base(base_len);
  ArrayType base = ArrayType::Zero(base_len);

#pragma omp parallel for shared(base, base_dims, dim0_seq, grid_dims, Qform) private(ijk0, j, k, ll, mm, nn) schedule(static, base_dims[0] / dualres::__internals::_N_THREADS_)
  for (ll = 0; ll < base_dims[0]; ll++) {
    j = 0;
    ijk0 = Eigen::Vector4f::Zero();
    ijk0[0] = dim0_seq[ll];
    for (mm = 0; mm < base_dims[1]; mm++) {
      k = 0;
      if (mm < grid_dims[1]) ++j; else --j;
      ijk0[1] = j - 1;
      for (nn = 0; nn < base_dims[2]; nn++) {
	if (nn < grid_dims[2]) ++k; else --k;
	ijk0[2] = k - 1;
	// base[ nn + mm * base_dims[2] + ll * base_dims[1] * base_dims[2] ] =
	base[ nn * base_dims[0] * base_dims[1] + mm * base_dims[0] + ll ] =
	  static_cast<scalar_type>((Qform * ijk0).head<3>().norm());
      }
    }
  }
  return base;
};















/*


template< typename scalar_type >
dualres::lambda_profile dualres::profile_lambda_computation_3d(  // return elapsed time per dft
  const Eigen::Vector3i &data_grid_dims,
  const dualres::qform_type &Qform,
  const bool use_nearest_power_2,
  const scalar_type kernel_bandwidth,
  const scalar_type kernel_exponent
) {
  dualres::lambda_profile prof;
  af::array base = kernels::rbf(dualres::circulant_base_3d<scalar_type>(
      data_grid_dims, Qform, use_nearest_power_2),
    kernel_bandwidth, kernel_exponent);
  af::timer t0 = af::timer::start();
  af::array lambda = af::dft(base);
  prof.compute_time = af::timer::stop(t0);
  prof.size = lambda.elements();
  prof.negative_values = af::sum<int>(af::count(af::sign(af::real(lambda))));
  return prof;
};



template< typename scalar_type >
af::array dualres::profile_and_compute_lambda_3d(  // return elapsed time per dft
  const Eigen::Vector3i &data_grid_dims,
  const dualres::qform_type &Qform,
  const scalar_type kernel_bandwidth,
  const scalar_type kernel_exponent,
  const bool verbose
) {
  dualres::lambda_profile prof_extended = profile_lambda_computation_3d<scalar_type>(
    data_grid_dims, Qform, true, kernel_bandwidth, kernel_exponent);
  dualres::lambda_profile prof_short = profile_lambda_computation_3d<scalar_type>(
    data_grid_dims, Qform, false, kernel_bandwidth, kernel_exponent);
  const bool use_extended =
    (double)prof_extended.negative_values / prof_extended.size <
    (double)prof_short.negative_values / prof_short.size;
  if (verbose) {
    std::cout << "Profiling eigen decomposition:\n\n"
	      << "Extended grid " << prof_extended << "\n"
	      << "Non-extended grid (" << prof_short.size
	      << " elements) circulant matrix decomposition\n"
	      << "  took " << prof_short.compute_time
	      << " (sec) and returned\n"
	      << "  " << prof_short.negative_values
	      << " eigen values < 0 ("
	      << std::setprecision(2) << std::fixed
	      << ((double)prof_short.negative_values / prof_short.size * 100)
	      << "%)\n" << std::endl;
    if (use_extended)
      std::cout << "<selecting extended grid space>\n\n";
    else
      std::cout << "<selecting native grid space>\n\n";
  }
  af::array lambda = af::dft(kernels::rbf(circulant_base_3d<scalar_type>(
      data_grid_dims, Qform, use_extended),
    kernel_bandwidth, kernel_exponent));
  // replace negative eigen values with 0 if necessary
  if ((use_extended && prof_extended.negative_values > 0) ||
      (!use_extended && prof_short.negative_values > 0)) {
      // lambda *= (1 - af::sign(af::real(lambda)));
    af::replace(lambda, af::sign(af::real(lambda)) != 1, 0);
  }
  return lambda;
};

*/

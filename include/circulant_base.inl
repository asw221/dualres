
#include <arrayfire.h>
#include <Eigen/Core>
#include <iomanip>
#include <iostream>

#include "defines.h"
#include "kernels.h"














template< typename OStream >
OStream& dualres::operator<<(OStream& os, const dualres::lambda_profile& lp) {
  os << "Circulant matrix with base of size " << lp.size << " elements:\n"
     << "  Decomposition took " << lp.compute_time << "(sec)\n"
     << "  And returned " << lp.negative_values << " eigen values < 0 ("
     << ((double)lp.negative_values / lp.size * 100) << "%)\n";
  return os;
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


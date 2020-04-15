
#include <cmath>
#include <stdexcept>


// scalar version
template< typename T >
T dualres::kernels::rbf(
  const T &distance,
  const T &bandwidth,
  const T &exponent,
  const T &variance
) {
  return variance * std::exp(-bandwidth * std::pow(std::abs(distance), exponent));
};



template< typename T >
T dualres::kernels::rbf_inverse(
  const T &rho,
  const T &bandwidth,
  const T &exponent,
  const T &variance
) {
  if (rho <= (T)0 || rho >= (T)1)
    throw std::domain_error("rbf_inverse: inverse kernel only for rho between (0, 1)");
  return std::pow(-std::log(rho / variance) / bandwidth, 1 / exponent);
};




template< typename T >
T dualres::kernels::rbf_bandwidth_to_fwhm(
  const T &bandwidth,
  const T &exponent
) {
  return 2.0 * std::pow(std::log((T)2) / bandwidth, 1 / exponent);
};



template< typename T >
T dualres::kernels::rbf_fwhm_to_bandwidth(
  const T &fwhm,
  const T &exponent
) {
  return std::log((T)2) / std::pow(fwhm / 2, exponent);
};











// From DUALRES namespace
// -------------------------------------------------------------------

// template< typename T >
// T dualres::inverse_logistic_function(const T &x, const T &max_val) {
//   return std::log(x / (max_val - x));
// };


// template< typename T >
// T dualres::inverse_logistic_gradient(const T &x, const T &max_val) {
//   return max_val / (x * (max_val - x));
// };


// template< typename T >
// T dualres::logistic_function(const T &x) {
//   return 1 / (1 + std::exp(-x));
// };


// template< typename T >
// T dualres::logistic_gradient(const T &x) {
//   return logistic_function(x) * (1 - logistic_function(x));
// };

  



// // From DUALRES::KERNELS namespace
// // -------------------------------------------------------------------


// template< typename T >
// af::array dualres::kernels::rbf(
//   const af::array &distance,
//   const T &bandwidth,
//   const T &exponent,
//   const T &marginal_variance
// ) {
//   return marginal_variance *
//     af::exp(-bandwidth * af::pow(af::abs(distance), exponent));
// };



// af::array dualres::kernels::rbf(const af::array &distance, const af::array &theta) {
// #ifndef NDEBUG
//   if (theta.dims(0) != 3)
//     throw std::logic_error("RBF kernel must have 3 parameters");
// #endif
//   return dualres::kernels::rbf(distance,
//     theta(1).scalar<float>(), theta(2).scalar<float>(), theta(0).scalar<float>());
// };



// template< typename T >
// af::array dualres::kernels::rbf_inverse_transform_parameters(
//   const af::array &trans,
//   const T &tau_max
// ) {
// #ifndef NDEBUG
//   if (trans.dims(0) != 3)
//     throw std::logic_error("RBF kernel must have 3 parameters");
// #endif
//   af::array theta = trans;
//   theta(0) = dualres::logistic_function(trans(0).scalar<T>()) * tau_max;
//   theta(1) = std::exp(trans(1).scalar<T>());
//   // theta(1) = dualres::logistic_function(trans(1).scalar<T>()) * 10;
//   theta(2) = dualres::logistic_function(trans(2).scalar<T>()) * 2;
//   return theta;
// };



// template< typename T >
// af::array dualres::kernels::rbf_transformed(
//   const af::array &distance,
//   const af::array &trans,
//   const T &tau_max
// ) {
//   return dualres::kernels::rbf(
//     distance, dualres::kernels::rbf_inverse_transform_parameters(trans, tau_max));
// };
    


// template< typename T >
// af::array dualres::kernels::rbf_transform_parameters(
//   const af::array &theta,
//   const T &tau_max
// ) {
// #ifndef NDEBUG
//   if (theta.dims(0) != 3)
//     throw std::logic_error("RBF kernel must have 3 parameters");
// #endif
//   af::array trans = theta;
//   trans(0) = dualres::inverse_logistic_function(theta(0).scalar<T>(), tau_max);
//   // trans(1) = dualres::inverse_logistic_function(theta(1).scalar<T>(), (T)10.0);
//   trans(1) = std::log(theta(1).scalar<T>());
//   trans(2) = dualres::inverse_logistic_function(theta(2).scalar<T>(), (T)2.0);
//   return trans;
// };

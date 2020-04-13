
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <nifti1_io.h>
#include <nlopt.hpp>
#include <stdexcept>
#include <vector>

#include "dualres/defines.h"
#include "dualres/nifti_manipulation.h"
#include "dualres/perturbation_matrix.h"


#ifndef _DUALRES_ESTIMATE_KERNEL_PARAMETERS_
#define _DUALRES_ESTIMATE_KERNEL_PARAMETERS_


namespace dualres {



  struct mce_data {  // Summary statistic data for Minimum Contrast Estimation
    Eigen::VectorXf covariance;
    Eigen::VectorXf distance;
    Eigen::VectorXi npairs;
  };

  
  struct kernel_data {  // Simplification of above with std vectors
    std::vector<double> distance;
    std::vector<double> covariance;
    // double marginal_variance;
  };
  
  

  template< typename Scalar = float >
  dualres::mce_data compute_mce_summary_data_impl(const nifti_image* const img) {
    const int nx = img->nx, ny = img->ny, nvox = (int)img->nvox;
    Scalar* dataPtr = (Scalar*)img->data;
    double voxel_value_x, voxel_value_y;
    dualres::qform_type Q = dualres::qform_matrix(img);
    Eigen::MatrixXi P = dualres::perturbation_matrix_3d();
    Eigen::VectorXi stride = P.col(2) * nx * ny + P.col(1) * nx + P.col(0);
    P.conservativeResize(P.rows(), P.cols() + 1);
    P.col(P.cols() - 1) = Eigen::VectorXi::Zero(P.rows());
    Eigen::VectorXd sum_xy = Eigen::VectorXd::Zero(P.rows());
    Eigen::VectorXd sum_x  = Eigen::VectorXd::Zero(P.rows());
    Eigen::VectorXd sum_y  = Eigen::VectorXd::Zero(P.rows());
    dualres::mce_data summ;
    summ.covariance = Eigen::VectorXf::Zero(P.rows());
    summ.distance = (P.cast<float>() * Q.transpose()).rowwise().norm();
    summ.npairs = Eigen::VectorXi::Zero(P.rows());
    for (int voxel_index = 0; voxel_index < nvox; voxel_index++) {
      voxel_value_x = (double)(*dataPtr);
      if (!(isnan(voxel_value_x) || voxel_value_x == 0)) {
	for (int i = 0; i < P.rows(); i++) {
	  if (voxel_index + stride[i] >= 0 && voxel_index + stride[i] < nvox) {
	    voxel_value_y = (double)(*(dataPtr + stride[i]));
	    if (!(isnan(voxel_value_y) || voxel_value_y == 0)) {
	      sum_xy[i] += voxel_value_x * voxel_value_y;
	      sum_x[i] += voxel_value_x;
	      sum_y[i] += voxel_value_y;
	      summ.npairs[i]++;
	    }
	  }
	}  // for (int i; ...
      }
      dataPtr++;
    }  // for (int voxel_index = 0; ...
    for (int i = 0; i < P.rows(); i++) {
      if (summ.npairs[i] > 1) {
	summ.covariance[i] = (sum_xy[i] - sum_x[i] * sum_y[i] / summ.npairs[i]) /
	  (summ.npairs[i] - 1);
      }
    }
    return summ;
  };



  dualres::mce_data compute_mce_summary_data(const nifti_image* const img) {
    const dualres::nifti_data_type dtype = dualres::nii_data_type(img);
    if (dtype == dualres::nifti_data_type::FLOAT)
      return dualres::compute_mce_summary_data_impl<float>(img);
    else if (dtype == dualres::nifti_data_type::DOUBLE)
      return dualres::compute_mce_summary_data_impl<double>(img);
    else
      throw std::logic_error("compute_mce_summary_data: unimplemented image type");

    // Not reached:
    dualres::mce_data not_reached;
    return not_reached;
  };

  


  double _rbf_least_squares(
    const std::vector<double> &x,
    std::vector<double> &grad,
    void *data
  ) {
    kernel_data *_dat = (kernel_data*)data;
    const int N = _dat->distance.size();
    double objective = 0.0, resid;
    for (int i = 0; i < N; i++) {
      resid = _dat->covariance[i] - x[0] *
	std::exp(-x[1] * std::pow(_dat->distance[i], x[2]));
      objective += resid * resid / N;
    }
    return objective;
  };


  double _rbf_constraint(
    const std::vector<double> &x,
    std::vector<double> &grad,
    void* data
  ) {
    // Constrain exponent > bandwidth
    return x[1] - x[2];
  };


  

  // Estimate RBF parameters from MCE_DATA summaries
  // Modify theta with optimized values. Uses derivative-free COBYLA 
  // optimizer from NLOPT
  int compute_rbf_parameters(
    std::vector<double> &theta,
    const dualres::mce_data &data
  ) {
    const int K = 3;  // 3 parameters, rbf model
    const double tau_max = (double)data.covariance[0], eps0 = 1e-5;
    double min_obj;
    int retval = 0;
    std::vector<double> _x, _y;
    kernel_data objective_data;
    std::vector<double> lb{eps0, eps0, eps0};
    std::vector<double> ub{tau_max, HUGE_VAL, 2.0 - eps0};
    nlopt::opt optimizer(nlopt::LN_COBYLA, K);
    // Prepare data
    _x.reserve(data.npairs.size());
    _y.reserve(data.npairs.size());
    for (int i = 1; i < data.npairs.size(); i++) {
      if (data.npairs[i] != 0) {
	_x.push_back(std::abs((double)data.distance[i]));
	_y.push_back((double)data.covariance[i]);
      }
    }
    _x.shrink_to_fit();
    _y.shrink_to_fit();
    objective_data.distance = std::move(_x);
    objective_data.covariance = std::move(_y);
    optimizer.set_lower_bounds(lb);
    optimizer.set_upper_bounds(ub);
    optimizer.set_min_objective(dualres::_rbf_least_squares, &objective_data);
    optimizer.add_inequality_constraint(dualres::_rbf_constraint, NULL, eps0 / 100);
    optimizer.set_xtol_rel(1e-5);
    try {
      optimizer.optimize(theta, min_obj);
    }
    catch (std::exception &_e) {
      retval = 1;
      // std::cout << "NLOPT failed\n";
    }
    return retval;
  };
  

  
}


#endif  // _DUALRES_ESTIMATE_KERNEL_PARAMETERS_










// Recycling
// -------------------------------------------------------------------



  // af::array compute_rbf_parameters_orig(const dualres::mce_data &data) {
  //   // Opitmization with Adam algorithm
  //   const float eta = 0.1, b = 0.9, g = 0.99, tol = 1e-5, eps0 = 1e-7;
  //   // const float max_psi = 10.0;  // can set this as log(1e-4) / min_dist^0.01 e.g.
  //   const int max_steps = 1000;
  //   std::vector<int> _use(data.npairs.size());
  //   for (int i = 0; i < _use.size(); i++) {
  //     if (i == 0 || data.npairs[i] == 0)
  // 	_use[i] = 0;
  //     else
  // 	_use[i] = 1;
  //   }
  //   const af::array y(data.covariance.rows(), data.covariance.data());
  //   const af::array x = af::abs(af::array(data.distance.rows(), data.distance.data()));
  //   const af::array use_data(_use.size(), _use.data());
  //   af::array update = af::constant(1, 3, af_dtype::f32);
  //   update(0) = 0;
  //   update(2) = 0;
  //   af::array theta = _rbf_find_starting_values(y, x, use_data);
  //   theta(0) = 0.9 * data.covariance[0];
  //   af_print(theta.T());
  //   af::array transformed_theta = dualres::kernels::rbf_transform_parameters(
  //     theta, data.covariance[0]);
  //   af::array grad = dualres::rbf_transformed_gradient_ssq(
  //     y, x, use_data, transformed_theta, data.covariance[0]) * update;
  //   af::array mt = grad * 0;
  //   af::array vt = grad * 0;
  //   af::array dt = grad * 0;
  //   af::array residuals = y - dualres::kernels::rbf_transformed(
  //     x, transformed_theta, data.covariance[0]);
  //   bool converged = false;
  //   float objective_val, last_objective_val;
  //   int step = 0;
  //   last_objective_val = af::sum<float>(residuals * residuals * use_data);
  //   while (!converged && step < max_steps) {
  //     // af_print(grad.T());
  //     mt = b * mt + (1 - b) / (1 - std::pow(1 - b, step + 1)) * grad;
  //     vt = g * vt + (1 - g) * std::sqrt(1 - std::pow(1 - g, step + 1)) * grad * grad;
  //     dt = eta * mt / af::sqrt(vt + eps0);
  //     transformed_theta -= dt;
  //     residuals = y - dualres::kernels::rbf_transformed(
  //       x, transformed_theta, data.covariance[0]);
  //     objective_val = af::sum<float>(residuals * residuals * use_data);
  //     converged = (objective_val <= last_objective_val) &&
  // 	(std::abs(objective_val - last_objective_val) < tol ||
  // 	 af::norm(dt) < tol);
  //     grad = dualres::rbf_transformed_gradient_ssq(
  //       y, x, use_data, transformed_theta, data.covariance[0]) * update;
  //     last_objective_val = objective_val;
  //     // std::cout << "Step " << step << ": objective = " << objective_val << "\n";
  //     step++;
  //   }
  //   if (!converged)
  //     std::cout << "WARNING: did not converge after " << max_steps << " iterations\n";
  //   else
  //     std::cout << "Converged after " << step << " iterations\n"
  // 		<< "Objective = " << objective_val << "\n";
  //   theta = dualres::kernels::rbf_inverse_transform_parameters(
  //     transformed_theta, data.covariance[0]);
  //   return theta;
  // };
  





  
  //   template< typename T >
  //   af::array rbf_transformed_gradient_ssq(
  //     const af::array &y,
  //     const af::array &x,
  //     const af::array &use_data,
  //     const af::array &trans,
  //     const T &tau_max
  //   ) {
  //     std::vector<float> _grad(trans.dims(0));
  //     af::array theta = dualres::kernels::rbf_inverse_transform_parameters(trans, tau_max);
  //     af::array mean = dualres::kernels::rbf(x, theta);
  //     af::array scaled_residuals = -2 / theta(0).scalar<T>() * mean * (y - mean) * use_data;
  //     _grad[0] = tau_max * dualres::logistic_gradient(trans(0).scalar<T>()) *
  // 	af::sum<T>(scaled_residuals);
  //     // _grad[1] = 10.0 * dualres::logistic_gradient(trans(1).scalar<T>()) *
  //     // 	theta(0).scalar<T>() *
  //     // 	af::sum<T>(af::pow(af::abs(x), theta(2).scalar<T>()) * scaled_residuals);
  //     _grad[1] = (theta(1) * theta(0)).scalar<T>() *
  //     	af::sum<T>(af::pow(af::abs(x), theta(2).scalar<T>()) * scaled_residuals);
  //     _grad[2] = 2.0 * dualres::logistic_gradient(trans(2).scalar<T>()) *
  // 	(theta(0) * theta(1)).scalar<T>() *
  // 	af::sum<T>(af::log(af::abs(x) + 1e-5) *
  // 		   af::pow(af::abs(x), theta(2).scalar<T>()) * scaled_residuals);
  //     return af::array(_grad.size(), _grad.data());
  //   };






  // af::array _rbf_find_starting_values(
  //   const af::array &y,
  //   const af::array &x,
  //   const af::array &use_data,
  //   const float delta = 1.5
  // ) {
  //   // Find approximate starting values through a series of simple
  //   // linear regressions
  //   const float eps0 = 1e-5, N = af::sum<float>(use_data);
  //   float beta0, beta1, gamma0, gamma1;
  //   std::vector<float> _theta(3);
  //   af::array A, z;
  //   // With Y = Covariance, and X = | Distance |, first solve,
  //   // Y_i ~ b0 + b1 X_i + e_i
  //   beta1 = af::sum<float>(use_data * y * x) / af::sum<float>(use_data * x * x);
  //   beta0 = af::sum<float>(use_data * y - beta1 * x * use_data) / N;
  //   A = beta0 + beta1 * x;
  //   af::replace(A, A > eps0, eps0);
  //   A = af::log(A);
  //   z = af::pow(x, delta);
  //   // Then with the RBF exponent fixed, Y = log(b0 + b1 X), and Z = X^delta,
  //   // solve,
  //   // A_i = g0 + g1 Z_i + d_i
  //   gamma1 = af::sum<float>(use_data * A * z) / af::sum<float>(use_data * z * z);
  //   gamma0 = af::sum<float>(use_data * A - gamma1 * z * use_data) / N;
  //   _theta[0] = std::exp(gamma0);
  //   _theta[1] = -gamma1;
  //   _theta[2] = delta;
  //   return af::array(_theta.size(), _theta.data());
  // };



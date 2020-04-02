
#include <algorithm>
#include <boost/math/distributions.hpp>
#include <limits>
#include <random>


/*
                          *** Currently Unused ***
 */


#ifndef _DUALRES_EXTRA_DISTRIBUTIONS_
#define _DUALRES_EXTRA_DISTRIBUTIONS_



namespace dualres {
    

  template< typename RealType >
  RealType sample_truncated_gamma(
    const boost::math::gamma_distribution<RealType> &Gamma,
    const RealType &lower_limit = 0,
    const RealType &upper_limit = std::numeric_limits<RealType>::max()
  ) {
#ifndef NDEBUG
    if (lower_limit < 0 || upper_limit < 0)
      throw std::logic_error("Gamma distribution support must be > 0");
#endif
    static const RealType _EPS_ = 0.0000001;
    static const RealType _PMAX_ = 0.9999999;
    const RealType unif_upper = std::min(boost::math::cdf(Gamma, upper_limit), _PMAX_);
    const RealType unif_lower = boost::math::cdf(Gamma, std::max(lower_limit, _EPS_));
    std::uniform_real_distribution<RealType> Uniform(unif_lower, unif_upper);
    const RealType p = Uniform(dualres::_RNG_);
    return boost::math::quantile(Gamma, p);
  };
  
  
}


#endif  // _DUALRES_EXTRA_DISTRIBUTIONS_


#include <arrayfire.h>


#ifndef _DUALRES_EXTRA_MATH_
#define _DUALRES_EXTRA_MATH_


namespace dualres {


  af::array csqrt(const af::array& in) {
    af::array r = af::real(in);
    af::array i = af::imag(in);

    af::array phi = af::atan2(i, r) / 2.0f;
    af::array a = af::sqrt(af::hypot(r, i));

    return af::complex(a * af::cos(phi), a * af::sin(phi));
  };
  
  
}


#endif  // _DUALRES_EXTRA_MATH_


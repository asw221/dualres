
#include <Eigen/Core>
#include <vector>


#ifndef _DUALRES_UTILITIES_
#define _DUALRES_UTILITIES_


namespace dualres {


  namespace utilities {



    class progress_bar {
    public:
      progress_bar(unsigned int max_val);
      
      void finish();
      void operator++();
      void operator++(int);
      void value(unsigned int value);

      template< typename OStream >
      friend OStream& operator<<(OStream& os, const progress_bar& pb);
      
    private:
      bool _active;
      char __;
      unsigned int _max_val;
      unsigned int _print_width;
      unsigned int _bar_print_width;
      unsigned int _value;
    };


    

    template< typename MatrixType >
    MatrixType eigen_select(
      const MatrixType &M,
      const std::vector<int> &row_indices,
      const std::vector<int> &col_indices
    ) {
      MatrixType Sub(row_indices.size(), col_indices.size());
      int i, j = 0;
      for (std::vector<int>::const_iterator jt = col_indices.cbegin();
	   jt != col_indices.cend(); ++jt, ++j) {
	i = 0;
	for (std::vector<int>::const_iterator it = row_indices.cbegin();
	     it != row_indices.cend(); ++it, ++i) {
	  Sub(i, j) = M(*it, *jt);
	}
      }
      return Sub;
    };



    template< typename MatrixType >
    MatrixType eigen_select_symmetric(const MatrixType &M, const std::vector<int> &indices) {
      MatrixType Sub(indices.size(), indices.size());
      int i, j = 0;
      // Loop only over lower triangle + diagonal
      for (std::vector<int>::const_iterator jt = indices.cbegin();
	   jt != indices.end(); ++jt, ++j) {
	i = j;
	for (std::vector<int>::const_iterator it = jt;
	     it != indices.cend(); ++it, ++i) {
	  Sub(i, j) = M(*it, *jt);
	  Sub(j, i) = M(*it, *jt);  // redundant assignment for diagonal
	}
      }
      return Sub;
    };

    
  }
  
}


#include "utilities.inl"

#endif  // _DUALRES_UTILITIES_

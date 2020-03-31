
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nifti1_io.h>
#include <vector>

#include "defines.h"
#include "kriging_matrix.h"
#include "nifti_manipulation.h"


#ifndef _DUALRES_MULTI_RES_DATA_2_
#define _DUALRES_MULTI_RES_DATA_2_


namespace dualres {
  

  template< typename T >
  class MultiResData {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Vector<scalar_type, Eigen::Dynamic> VectorType;
    typedef typename Eigen::SparseMatrix<scalar_type, Eigen::RowMajor> SparseMatrixType;
    
    
    void push_back_data(const scalar_type* data, const int size);
    void push_back_weight(const dualres::kriging_matrix_data<scalar_type> &kmd);

    const VectorType& Y(const int which) const;
    const SparseMatrixType& W(const int which) const;

    int n_datasets() const;
    
  private:
    std::vector< Eigen::Map<VectorType> > _Y;  // data - Y[0] high res; Y[1] std res
    std::vector< Eigen::Map<SparseMatrixType> > _W;
    // ^^ Mappings of Y[0] space -> Y[1], ... space
  };


  
}





// -----------------------------------------------------------------------------




template< typename T >
int dualres::MultiResData<T>::n_datasets() const {
  return _Y.size();
};





template< typename T >
void dualres::MultiResData<T>::push_back_data(
  const typename dualres::MultiResData<T>::scalar_type* data,
  const int size
) {
  _Y.push_back(Eigen::Map<VectorType>(data, size));
};



template< typename T >
void dualres::MultiResData<T>::push_back_weight(
  const dualres::kriging_matrix_data<typename dualres::MultiResData<T>::scalar_type> &kmd
) {
  if (_Y.size() >= 1) {
    if (_Y[0].size() != kmd.ncol)
      throw std::logic_error("Weight matrix ncol should equal dim Y(0)");
  }
  _W.push_back(Eigen::Map<SparseMatrixType>(
    kmd.nrow, kmd.ncol, kmd._Data.size(),
    kmd.cum_row_counts.data(), kmd.column_indices.data(),
    kmd._Data.data())
  );
};



template< typename T >
const typename dualres::MultiResData<T>::VectorType&
dualres::MultiResData<T>::Y(const int which) const {
#ifndef NDEBUG
  if (which < 0 || which >= _Y.size())
    throw std::logic_error("MultiResData: bad data index");
#endif
  return _Y[which];
};


template< typename T >
const typename dualres::MultiResData<T>::SparseMatrixType&
dualres::MultiResData<T>::W(const int which) const {
#ifndef NDEBUG
  if (which < 0 || which >= _W.size())
    throw std::logic_error("MultiResData: bad weight matrix index");
#endif
  return _W[which];
};



#endif  // _DUALRES_MULTI_RES_DATA_2_



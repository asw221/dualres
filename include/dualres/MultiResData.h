
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nifti1_io.h>
#include <vector>

#include "dualres/defines.h"
#include "dualres/kriging_matrix.h"
#include "dualres/nifti_manipulation.h"


#ifndef _DUALRES_MULTI_RES_DATA_2_
#define _DUALRES_MULTI_RES_DATA_2_


namespace dualres {
  

  template< typename T >
  class MultiResData {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1> VectorType;
    typedef typename Eigen::SparseMatrix<scalar_type, Eigen::RowMajor> SparseMatrixType;

    MultiResData();
    MultiResData(const nifti_image* const h_res);
    MultiResData(
      const nifti_image* const h_res,
      const nifti_image* const s_res,
      const std::vector<scalar_type> &kernel_parameters,
      const scalar_type &neighborhood_radius
    );

    int n_datasets() const;
    
    const SparseMatrixType& W() const;
    const VectorType& Yh() const;
    const VectorType& Ys() const;
    
  private:
    int _n_datasets;
    VectorType _Yh;
    VectorType _Ys;
    SparseMatrixType _W;
    // ^^ Mappings of _Yh space -> _Ys, ... space
  };


  
} // namespace dualres





// -----------------------------------------------------------------------------



template< typename T >
dualres::MultiResData<T>::MultiResData() {
  _n_datasets = 0;
  _Yh = VectorType::Zero(1);
  _Ys = VectorType::Zero(1);
  _W  = SparseMatrixType(1, 1);
};


template< typename T >
dualres::MultiResData<T>::MultiResData(const nifti_image* const h_res) {
  std::vector<scalar_type> v_Yh = dualres::get_nonzero_data<scalar_type>(h_res);
  _n_datasets = 1;
  _Yh = Eigen::Map<VectorType>(v_Yh.data(), v_Yh.size());
  _Ys = VectorType::Zero(1);
  _W  = SparseMatrixType(1, 1);
};


template< typename T >
dualres::MultiResData<T>::MultiResData(
  const nifti_image* const h_res,
  const nifti_image* const s_res,
  const std::vector<typename dualres::MultiResData<T>::scalar_type> &kernel_parameters,
  const typename dualres::MultiResData<T>::scalar_type &neighborhood_radius
) {
  dualres::kriging_matrix_data<scalar_type> kmd =
    dualres::get_sparse_kriging_matrix_data<scalar_type>(
      h_res, s_res, kernel_parameters, neighborhood_radius);
  std::vector<scalar_type> v_Yh = dualres::get_nonzero_data<scalar_type>(h_res);
  std::vector<scalar_type> v_Ys = dualres::get_nonzero_data<scalar_type>(s_res);
  
  _n_datasets = 1;
  _Yh = Eigen::Map<VectorType>(v_Yh.data(), v_Yh.size());
  _Ys = Eigen::Map<VectorType>(v_Ys.data(), v_Ys.size());
  _W  = Eigen::Map<SparseMatrixType>(
    kmd.nrow, kmd.ncol, kmd._Data.size(),
    kmd.cum_row_counts.data(), kmd.column_indices.data(),
    kmd._Data.data());
};




template< typename T >
int dualres::MultiResData<T>::n_datasets() const {
  return _n_datasets;
};



template< typename T >
const typename dualres::MultiResData<T>::VectorType&
dualres::MultiResData<T>::Yh() const {
  return _Yh;
};


template< typename T >
const typename dualres::MultiResData<T>::VectorType&
dualres::MultiResData<T>::Ys() const {
  return _Ys;
};


template< typename T >
const typename dualres::MultiResData<T>::SparseMatrixType&
dualres::MultiResData<T>::W() const {
  return _W;
};



#endif  // _DUALRES_MULTI_RES_DATA_2_



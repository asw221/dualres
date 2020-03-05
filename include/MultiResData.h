
#include <algorithm>
#include <arrayfire.h>
#include <Eigen/Core>
#include <nifti1_io.h>
#include <vector>

#include "defines.h"
#include "krigging_matrix.h"
#include "nifti_manipulation.h"


#ifndef _DUALRES_MULTI_RES_DATA_
#define _DUALRES_MULTI_RES_DATA_


namespace dualres {
  

  template< typename T >
  class MultiResData {
  public:
    MultiResData() : _dtype(dualres::data_types<T>::af_dtype)  { ; }
    void push_back_data(const af::array &Y);
    void push_back_weight(const int nrow, const int ncol, const int nnz,
			  const void * const values, const int * const rowIndices,
			  const int * const colIndices);

    const af::array& Y(const int which) const;
    const af::array& W(const int which) const;

    ::af_dtype type() const;
    int n_datasets() const;
    
  private:
    ::af_dtype _dtype;
    std::vector< af::array > _Y;  // data - Y[0] high res; Y[1] std res
    std::vector< af::array > _W;  // Mappings of Y[0] space -> Y[1], ... space
  };



    
  template< typename DataType = float >
  void construct_and_store_krigging_array(
    dualres::MultiResData<DataType> &_data,
    const nifti_image* const high_res,
    const nifti_image* const std_res,
    const std::vector<DataType> &rbf_params,
    const DataType &neighborhood_radius,
    const af::dim4 &extended_grid_dims
  ) {
    const dualres::nifti_bounding_box bb_hr = dualres::get_bounding_box(high_res);
    const Eigen::Vector3i bhr_dims = bb_hr.ijk_max - bb_hr.ijk_min +
      Eigen::Vector3i::Ones();
    const af::dim4 bhr_grid(bhr_dims[0], bhr_dims[1], bhr_dims[2]);
    for (int i = 0; i < 3; i++) {
      if (bhr_grid[i] > extended_grid_dims[i])
	throw std::logic_error(
          "sparse_krigging_array: old grid should be <= new grid");
    }
    dualres::krigging_matrix_data<float> kmd =
      dualres::get_sparse_krigging_array_data<DataType>(
        high_res, std_res, rbf_params, neighborhood_radius);
    dualres::re_grid_krigging_matrix_data<DataType>(kmd, bhr_grid, extended_grid_dims);
    _data.push_back_weight(kmd.nrow, kmd.ncol, kmd._Data.size(),
			   kmd._Data.data(), kmd.cum_row_counts.data(),
			   kmd.column_indices.data());
  };


  
}





// -----------------------------------------------------------------------------




template< typename T >
::af_dtype dualres::MultiResData<T>::type() const {
  return _dtype;
};


template< typename T >
int dualres::MultiResData<T>::n_datasets() const {
  return _Y.size();
};





template< typename T >
void dualres::MultiResData<T>::push_back_data(const af::array &Y) {
  _Y.push_back(Y);
};



template< typename T >
void dualres::MultiResData<T>::push_back_weight(
  const int nrow, const int ncol, const int nnz,
  const void * const values, const int * const rowIndices,
  const int * const colIndices
) {
  if (_Y.size() >= 1) {
    if (_Y[0].elements() != ncol)
      throw std::logic_error("Weight matrix should be (n_l x n_h)");
  }
  _W.push_back(af::sparse(nrow, ncol, nnz, values, rowIndices, colIndices, _dtype));
  // AF_STORAGE_COO));
};



template< typename T >
const af::array& dualres::MultiResData<T>::Y(const int which) const {
#ifndef NDEBUG
  if (which < 0 || which >= _Y.size())
    throw std::logic_error("MultiResData: bad data index");
#endif
  return _Y[which];
};


template< typename T >
const af::array& dualres::MultiResData<T>::W(const int which) const {
#ifndef NDEBUG
  if (which < 0 || which >= _W.size())
    throw std::logic_error("MultiResData: bad weight matrix index");
#endif
  return _W[which];
};



#endif  // _DUALRES_MULTI_RES_DATA_



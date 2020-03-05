


// Float specific ----------------------------------------------------


template<>
constexpr ::af_dtype dualres::data_types<float>::af_dtype() const {
  return ::af_dtype::f32;
};

template<>
constexpr ::af_dtype dualres::data_types<float>::af_ctype() const {
  return ::af_dtype::c32;
};

template<>
constexpr dualres::nifti_data dualres::data_types<float>::nifti_data() const {
  return dualres::nifti_data::FLOAT;
};



// Double specific ---------------------------------------------------

template<>
constexpr ::af_dtype dualres::data_types<double>::af_dtype() const {
  return ::af_dtype::f64;
};

template<>
constexpr ::af_dtype dualres::data_types<double>::af_ctype() const {
  return ::af_dtype::c64;
};

template<>
constexpr dualres::nifti_data dualres::data_types<double>::nifti_data() const {
  return dualres::nifti_data::DOUBLE;
};







// Base template -----------------------------------------------------

template< typename T >
constexpr ::af_dtype dualres::data_types<T>::af_dtype() const {
  return ::af_dtype::f32;
};

template< typename T >
constexpr ::af_dtype dualres::data_types<T>::af_ctype() const {
  return ::af_dtype::c32;
};

template< typename T >
constexpr dualres::nifti_data dualres::data_types<T>::nifti_data() const {
  return dualres::nifti_data::OTHER;
};

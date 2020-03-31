
#include <Eigen/Core>
#include <fstream>
#include <string>
#include <vector>


#include "defines.h"


#ifndef _DUALRES_UTILITIES_
#define _DUALRES_UTILITIES_




// *_indexing_functor's from:
// http://eigen.tuxfamily.org/dox-devel/TopicCustomizing_NullaryExpr.html#title1



namespace dualres {


  namespace utilities {

    bool initialize_temporary_directory();
    
    int set_number_of_threads(const unsigned int threads);
    int threads();
  
    void set_seed(const unsigned int seed);

  



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
    );


    template< typename MatrixType >
    MatrixType eigen_select_symmetric(const MatrixType &M, const std::vector<int> &indices);







    template< class ArgType, class RowIndexType, class ColIndexType >
    class matrix_indexing_functor {
    private:
      const ArgType &m_arg;
      const RowIndexType &m_rowIndices;
      const ColIndexType &m_colIndices;
  
    public:
      typedef Eigen::Matrix<typename ArgType::Scalar,
        RowIndexType::SizeAtCompileTime,
        ColIndexType::SizeAtCompileTime,
        ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
        RowIndexType::MaxSizeAtCompileTime,
        ColIndexType::MaxSizeAtCompileTime> MatrixType;
      
      matrix_indexing_functor(
        const ArgType &arg,
	const RowIndexType &row_indices,
	const ColIndexType& col_indices
      ) : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
      { ; }
  
      const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
	return m_arg(m_rowIndices[row], m_colIndices[col]);
      };
    };



    template< class ArgType, class IndexType >
    class vector_indexing_functor {
    private:
      const ArgType &m_arg;
      const IndexType &m_Indices;
  
    public:
      typedef Eigen::Matrix<typename ArgType::Scalar,
        IndexType::SizeAtCompileTime,
        1,
        ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
        IndexType::MaxSizeAtCompileTime,
        1> VectorType;
      
      vector_indexing_functor(
        const ArgType &arg,
	const IndexType &row_indices
      ) : m_arg(arg), m_Indices(row_indices)
      { ; }
  
      const typename ArgType::Scalar& operator() (Eigen::Index ind) const {
	return m_arg(m_Indices[ind]);
      };
    };




    bool file_exists(const std::string &fname) {
      std::ifstream ifs(fname.c_str());
      if (ifs.is_open()) {
	ifs.close();
	return true;
      }
      return false;
    };


    
    
  }  // namespace utilities ------------------------------------------




  



    
  template< class ArgType, class RowIndexType, class ColIndexType >
  Eigen::CwiseNullaryOp<
    dualres::utilities::matrix_indexing_functor<ArgType, RowIndexType, ColIndexType>,
    typename dualres::utilities::matrix_indexing_functor<
      ArgType, RowIndexType, ColIndexType>::MatrixType>
  nullary_index(
    const Eigen::MatrixBase<ArgType> &arg,
    const RowIndexType &row_indices,
    const ColIndexType& col_indices
  );

    

  template< class ArgType, class IndexType >
  Eigen::CwiseNullaryOp<
    dualres::utilities::vector_indexing_functor<ArgType, IndexType>,
    typename dualres::utilities::vector_indexing_functor<ArgType, IndexType>::VectorType >
  nullary_index(
    const Eigen::MatrixBase<ArgType> &arg,
    const IndexType &row_indices
  );




  
}  // namespace dualres ----------------------------------------------


#include "utilities.inl"
#include "eigen_indexing.inl"

#endif  // _DUALRES_UTILITIES_

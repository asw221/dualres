
#include <arrayfire.h>
#include <Eigen/Core>
#include <nifti1_io.h>
#include <vector>


namespace dualres {


  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> affine_matrix(const nifti_image* nii) {
    return Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(&nii->qto_xyz.m[0][0]);
  };


  
  Eigen::MatrixXi get_nonzero_indices(const nifti_image* nii) {
    if (nii->ndim < 3)
      throw std::logic_error("NIfTI image has dim < 3");
    const int nx = nii->nx, ny = nii->ny, nz = nii->nz, nvox = (int)nii->nvox;
    float* dataPtr = (float*)nii->data;
    std::vector<int> indices;
    for (int i = 0; i < nvox; i++) {
      if (!(isnan(*dataPtr) || *dataPtr == 0)) {
	indices.push_back(i);
      }
      dataPtr++;
    }
    Eigen::MatrixXi ijk(indices.size(), 3);
    for (int i = 0; i < indices.size(); i++) {
      ijk(i, 0) = indices[i] % nx;
      ijk(i, 1) = (indices[i] / nx) % ny;
      ijk(i, 2) = (indices[i] / (nx * ny)) % nz;
    }
    return ijk;
  };

  
  
  
  interp_mat compute_interpolation_matrix(
    const Eigen::MatrixXi &ijk_highres,
    const Eigen::MatrixXi &ijk_stdres,
    const Eigen::Matrix4f &affine_highres,
    const Eigen::Matrix4f &affine_stdres,
    const Eigen::Vector3f &kernel_parameters,
    const float &neighborhood_cuttoff = 0.05
  ) {
    
  };
  
}

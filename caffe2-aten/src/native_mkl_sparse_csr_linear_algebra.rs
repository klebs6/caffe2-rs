crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkl/SparseCsrLinearAlgebra.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkl/SparseCsrLinearAlgebra.cpp]

/**
  | Don't compile with MKL for MSVC/macos since
  | linking the sparse MKL routines needs some
  | build fixes.
  |
  | https://github.com/pytorch/pytorch/pull/50937#issuecomment-778732740
  |
  | Macros source:
  |
  | https://web.archive.org/web/20191012035921/http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
  */
#[cfg(any(not(feature = "mkl"),MSC_VER,APPLE,MACH))]
pub mod sparse_csr {

    use super::*;

    pub fn sparse_mm_mkl(
            self_:  &mut Tensor,
            sparse: &SparseCsrTensor,
            dense:  &Tensor,
            t:      &Tensor,
            alpha:  &Scalar,
            beta:   &Scalar) -> &mut Tensor {
        
        todo!();
            /*
                #if _MSC_VER
          AT_ERROR("sparse_mm_mkl: MKL support is disabled on Windows");
        #elif __APPLE__ || __MACH__
          AT_ERROR("sparse_mm_mkl: MKL support is disabled on macos/iOS.");
        #else
          AT_ERROR("sparse_mm_mkl: ATen not compiled with MKL support");
        #endif
          return self; // for stopping compiler warnings.
            */
    }
}

#[cfg(not(any(not(feature = "mkl"),MSC_VER,APPLE,MACH)))]
pub mod sparse_csr {

    use super::*;

    #[cfg(MKL_ILP64)]
    pub const TORCH_INT_TYPE: ScalarType = at::kLong;

    #[cfg(not(MKL_ILP64))]
    pub const TORCH_INT_TYPE: ScalarType = at::kInt;

    //--------------------
    pub struct SparseCsrMKLInterface {
        a:    SparseMatrix, // default = 0
        desc: MatrixDescr,
    }

    impl Drop for SparseCsrMKLInterface {
        fn drop(&mut self) {
            todo!();
            /*
                mkl_sparse_destroy(A);
            */
        }
    }

    impl SparseCsrMKLInterface {
        
        pub fn new(
            col_indices:  *mut MKL_INT,
            crow_indices: *mut MKL_INT,
            values:       *mut f64,
            nrows:        MKL_INT,
            ncols:        MKL_INT) -> Self {
        
            todo!();
            /*


                desc.type = SPARSE_MATRIX_TYPE_GENERAL;
                int retval = mkl_sparse_d_create_csr(
                    &A,
                    SPARSE_INDEX_BASE_ZERO,
                    nrows,
                    ncols,
                    crow_indices,
                    crow_indices + 1,
                    col_indices,
                    values);
                TORCH_CHECK(
                    retval == 0,
                    "mkl_sparse_d_create_csr failed with error code: ",
                    retval);
            */
        }
        
        pub fn new(
            col_indices:  *mut MKL_INT,
            crow_indices: *mut MKL_INT,
            values:       *mut f32,
            nrows:        MKL_INT,
            ncols:        MKL_INT) -> Self {
        
            todo!();
            /*


                desc.type = SPARSE_MATRIX_TYPE_GENERAL;
                int retval = mkl_sparse_s_create_csr(
                    &A,
                    SPARSE_INDEX_BASE_ZERO,
                    nrows,
                    ncols,
                    crow_indices,
                    crow_indices + 1,
                    col_indices,
                    values);
                TORCH_CHECK(
                    retval == 0,
                    "mkl_sparse_s_create_csr failed with error code: ",
                    retval);
            */
        }

        /**
          | res(nrows, dense_ncols) = (sparse(nrows
          | * ncols) @ dense(ncols x dense_ncols))
          |
          */
        #[inline] pub fn sparse_mm(&mut self, 
            res:         *mut f32,
            dense:       *mut f32,
            alpha:       f32,
            beta:        f32,
            nrows:       MKL_INT,
            ncols:       MKL_INT,
            dense_ncols: MKL_INT)  {
            
            todo!();
            /*
                int stat;
                if (dense_ncols == 1) {
                  stat = mkl_sparse_s_mv(
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A,
                    desc,
                    dense,
                    beta,
                    res);
                  TORCH_CHECK(stat == 0, "mkl_sparse_s_mv failed with error code: ", stat);
                } else {
                  stat = mkl_sparse_s_mm(
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A,
                    desc,
                    SPARSE_LAYOUT_ROW_MAJOR,
                    dense,
                    nrows,
                    ncols,
                    beta,
                    res,
                    dense_ncols);
                  TORCH_CHECK(stat == 0, "mkl_sparse_s_mm failed with error code: ", stat);
                }
            */
        }
        
        #[inline] pub fn sparse_mm(&mut self, 
            res:         *mut f64,
            dense:       *mut f64,
            alpha:       f64,
            beta:        f64,
            nrows:       MKL_INT,
            ncols:       MKL_INT,
            dense_ncols: MKL_INT)  {
            
            todo!();
            /*
                int stat;
                if (dense_ncols == 1) {
                  stat = mkl_sparse_d_mv(
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A,
                    desc,
                    dense,
                    beta,
                    res);
                  TORCH_CHECK(stat == 0, "mkl_sparse_d_mv failed with error code: ", stat);
                }
                else {
                  stat = mkl_sparse_d_mm(
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A,
                    desc,
                    SPARSE_LAYOUT_ROW_MAJOR,
                    dense,
                    nrows,
                    ncols,
                    beta,
                    res,
                    dense_ncols);
                  TORCH_CHECK(stat == 0, "mkl_sparse_d_mm failed with error code: ", stat);
                }
            */
        }
    }

    //--------------------
    #[inline] pub fn sparse_mm_mkl_template<Scalar>(
            res:          &mut Tensor,
            col_indices:  &Tensor,
            crow_indices: &Tensor,
            values:       &Tensor,
            dense:        &Tensor,
            t:            &Tensor,
            alpha:        &Scalar,
            beta:         &Scalar,
            size:         &[i32],
            dense_size:   &[i32])  {

        todo!();
            /*
                SparseCsrMKLInterface mkl_impl(
                  col_indices.data_ptr<MKL_INT>(),
                  crow_indices.data_ptr<MKL_INT>(),
                  values.data_ptr<Scalar>(),
                  size[0],
                  size[1]);
              mkl_impl.sparse_mm(
                  res.data_ptr<Scalar>(),
                  dense.data_ptr<Scalar>(),
                  alpha.to<Scalar>(),
                  beta.to<Scalar>(),
                  size[0],
                  size[1],
                  dense_size[1]);
            */
    }

    #[inline] pub fn is_mkl_int32_index() -> bool {
        
        todo!();
            /*
                #ifdef MKL_ILP64
              return false;
            #else
              return true;
            #endif
            */
    }

    pub fn sparse_mm_mkl(
            self_:  &mut Tensor,
            sparse: &SparseCsrTensor,
            dense:  &Tensor,
            t:      &Tensor,
            alpha:  &Scalar,
            beta:   &Scalar) -> &mut Tensor {
        
        todo!();
            /*
                if (is_mkl_int32_index()) {
                if (sparse_.crow_indices().scalar_type() != kInt) {
                  TORCH_WARN(
                      "Pytorch is compiled with MKL LP64 and will convert crow_indices to int32.");
                }
                if (sparse_.col_indices().scalar_type() != kInt) {
                  TORCH_WARN(
                      "Pytorch is compiled with MKL LP64 and will convert col_indices to int32.");
                }
              } else { // This is for future proofing if we ever change to using MKL ILP64.
                if (sparse_.crow_indices().scalar_type() != kLong) {
                  TORCH_WARN(
                      "Pytorch is compiled with MKL ILP64 and will convert crow_indices dtype to int64.");
                }
                if (sparse_.col_indices().scalar_type() != kLong) {
                  TORCH_WARN(
                      "Pytorch is compiled with MKL ILP64 and will convert col_indices dtype to int64.");
                }
              }
              AT_DISPATCH_FLOATING_TYPES(
                  dense.scalar_type(), "addmm_sparse_csr_dense", [&] {
                    sparse_mm_mkl_template<Scalar>(
                        self,
                        sparse_.col_indices().to(TORCH_INT_TYPE),
                        sparse_.crow_indices().to(TORCH_INT_TYPE),
                        sparse_.values(),
                        dense,
                        t,
                        alpha,
                        beta,
                        sparse_.sizes(),
                        dense.sizes());
                  });
              return self;
            */
    }
}

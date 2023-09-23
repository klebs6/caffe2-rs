crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkl/LinearAlgebra.cpp]

#[cfg(not(feature = "mkl"))]
pub mod mkl_disabled {

    use super::*;

    pub fn baddbmm_mkl<'a>(
            self_:  &mut Tensor,
            batch1: &Tensor,
            batch2: &Tensor,
            beta:   &Scalar,
            alpha:  &Scalar) -> &'a mut Tensor {
        
        todo!();
            /*
                AT_ERROR("bmm: ATen not compiled with MKL support");
            */
    }
}

#[cfg(feature = "mkl")]
pub mod mkl_enabled {

    use super::*;

    #[inline] pub fn gemm(
            trans_a: cblas_sys::CBLAS_TRANSPOSE,
            trans_b: cblas_sys::CBLAS_TRANSPOSE,
            m:       i32,
            n:       i32,
            k:       i32,
            alpha:   f32,
            a:       *const f32,
            lda:     i32,
            b:       *const f32,
            ldb:     i32,
            beta:    f32,
            c:       *mut f32,
            ldc:     i32)  {
        
        todo!();
            /*
                cblas_sgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            */
    }

    #[inline] pub fn gemm(
            trans_a: cblas_sys::CBLAS_TRANSPOSE,
            trans_b: cblas_sys::CBLAS_TRANSPOSE,
            m:       i32,
            n:       i32,
            k:       i32,
            alpha:   f64,
            a:       *const f64,
            lda:     i32,
            b:       *const f64,
            ldb:     i32,
            beta:    f64,
            c:       *mut f64,
            ldc:     i32)  {
        
        todo!();
            /*
                cblas_dgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            */
    }

    #[inline] pub fn gemm(
            trans_a: cblas_sys::CBLAS_TRANSPOSE,
            trans_b: cblas_sys::CBLAS_TRANSPOSE,
            m:       i32,
            n:       i32,
            k:       i32,
            alpha:   Complex<f32>,
            a:       *const Complex<f32>,
            lda:     i32,
            b:       *const Complex<f32>,
            ldb:     i32,
            beta:    Complex<f32>,
            c:       *mut Complex<f32>,
            ldc:     i32)  {
        
        todo!();
            /*
                cblas_cgemm(CblasRowMajor, trans_A, trans_B, M, N, K, reinterpret_cast<const void *>(&alpha),
            reinterpret_cast<const void*>(A), lda, reinterpret_cast<const void*>(B), ldb,
            reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(C), ldc);
            */
    }

    #[inline] pub fn gemm(
            trans_a: cblas_sys::CBLAS_TRANSPOSE,
            trans_b: cblas_sys::CBLAS_TRANSPOSE,
            m:       i32,
            n:       i32,
            k:       i32,
            alpha:   Complex<f64>,
            a:       *const Complex<f64>,
            lda:     i32,
            b:       *const Complex<f64>,
            ldb:     i32,
            beta:    Complex<f64>,
            c:       *mut Complex<f64>,
            ldc:     i32)  {
        
        todo!();
            /*
                cblas_zgemm(CblasRowMajor, trans_A, trans_B, M, N, K, reinterpret_cast<const void *>(&alpha),
            reinterpret_cast<const void*>(A), lda, reinterpret_cast<const void*>(B), ldb,
            reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(C), ldc);
            */
    }

    #[inline] pub fn gemm_batched(
            trans_a:    cblas_sys::CBLAS_TRANSPOSE,
            trans_b:    cblas_sys::CBLAS_TRANSPOSE,
            batch_size: i32,
            m:          i32,
            n:          i32,
            k:          i32,
            alpha:      f32,
            a:          *const *const f32,
            lda:        i32,
            b:          *const *const f32,
            ldb:        i32,
            beta:       f32,
            c:          *mut *mut f32,
            ldc:        i32)  {
        
        todo!();
            /*
                cblas_sgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
            A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
            */
    }

    #[inline] pub fn gemm_batched(
            trans_a:    cblas_sys::CBLAS_TRANSPOSE,
            trans_b:    cblas_sys::CBLAS_TRANSPOSE,
            batch_size: i32,
            m:          i32,
            n:          i32,
            k:          i32,
            alpha:      f64,
            a:          *const *const f64,
            lda:        i32,
            b:          *const *const f64,
            ldb:        i32,
            beta:       f64,
            c:          *mut *mut f64,
            ldc:        i32)  {
        
        todo!();
            /*
                cblas_dgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
            A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
            */
    }

    #[inline] pub fn gemm_batched(
            trans_a:    cblas_sys::CBLAS_TRANSPOSE,
            trans_b:    cblas_sys::CBLAS_TRANSPOSE,
            batch_size: i32,
            m:          i32,
            n:          i32,
            k:          i32,
            alpha:      Complex<f32>,
            a:          *const *const Complex<f32>,
            lda:        i32,
            b:          *const *const Complex<f32>,
            ldb:        i32,
            beta:       Complex<f32>,
            c:          *mut *mut Complex<f32>,
            ldc:        i32)  {
        
        todo!();
            /*
                cblas_cgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, reinterpret_cast<const void*>(&alpha),
            reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
            reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
            */
    }

    #[inline] pub fn gemm_batched(
            trans_a:    cblas_sys::CBLAS_TRANSPOSE,
            trans_b:    cblas_sys::CBLAS_TRANSPOSE,
            batch_size: i32,
            m:          i32,
            n:          i32,
            k:          i32,
            alpha:      Complex<f64>,
            a:          *const *const Complex<f64>,
            lda:        i32,
            b:          *const *const Complex<f64>,
            ldb:        i32,
            beta:       Complex<f64>,
            c:          *mut *mut Complex<f64>,
            ldc:        i32)  {
        
        todo!();
            /*
                cblas_zgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, reinterpret_cast<const void*>(&alpha),
            reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
            reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
            */
    }

    #[inline] pub fn baddbmm_mkl_template<Scalar>(
            res:   &Tensor,
            mat1:  &Tensor,
            mat2:  &Tensor,
            beta:  &Scalar,
            alpha: &Scalar)  {

        todo!();
            /*
                const auto mat1_strides = mat1.strides();
          const auto mat2_strides = mat2.strides();
          const auto mat1_sizes = mat1.sizes();
          const auto mat2_sizes = mat2.sizes();

          auto is_transposed = [](const IntArrayRef& strides, const IntArrayRef& sizes) {
            return strides[1] == 1 && strides[2] >= sizes[1];
          };

          const CBLAS_TRANSPOSE trans_A =
              is_transposed(mat1_strides, mat1_sizes) ? CblasTrans : CblasNoTrans;
          const CBLAS_TRANSPOSE trans_B =
              is_transposed(mat2_strides, mat2_sizes) ? CblasTrans : CblasNoTrans;


          // mat1: batch_size * M * K
          const int batch_size = mat1_sizes[0];
          const int M = mat1_sizes[1];
          // mat2: batch_size * K * N
          const int N = mat2_sizes[2];
          const int K = mat1_sizes[2];

          Scalar alpha = alpha_.to<Scalar>();
          Scalar beta = beta_.to<Scalar>();

          const int lda = trans_A == CblasTrans ? mat1_strides[2] : mat1_strides[1];
          const int ldb = trans_B == CblasTrans ? mat2_strides[2] : mat2_strides[1];
          const int ldc = res.strides()[1];

          // avoid using tensor accessor in the case of mat1/mat2 not being transposed
          // or only transposed in the last two axes
          const bool canAvoidTensorAccessor = mat1_strides[0] == mat1_sizes[1] * mat1_sizes[2] &&
            mat2_strides[0] == mat2_sizes[1] * mat2_sizes[2];

          Scalar* const res_data = res.data_ptr<Scalar>();

          if (batch_size == 1) {
            const Scalar* A;
            const Scalar* B;
            if (canAvoidTensorAccessor) {
              Scalar* mat1_data = mat1.data_ptr<Scalar>();
              Scalar* mat2_data = mat2.data_ptr<Scalar>();
              A = mat1_data;
              B = mat2_data;
            } else {
              auto mat1_acc = mat1.accessor<Scalar, 3>();
              auto mat2_acc = mat2.accessor<Scalar, 3>();
              A = mat1_acc[0].data();
              B = mat2_acc[0].data();
            }
            gemm(trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, res_data, ldc);
            return;
          }

          std::vector<const Scalar*> A;
          A.reserve(batch_size);
          std::vector<const Scalar*> B;
          B.reserve(batch_size);
          std::vector<Scalar*> C;
          C.reserve(batch_size);

          // avoid using tensor accessor in the case of mat1/mat2 not being transposed
          // or only transposed in the last two axis
          const auto res_sizes = res.sizes();
          if (canAvoidTensorAccessor) {
            Scalar* mat1_data = mat1.data_ptr<Scalar>();
            Scalar* mat2_data = mat2.data_ptr<Scalar>();
            for (i64 batch = 0; batch < batch_size; batch++) {
              A.emplace_back(mat1_data + batch * mat1_sizes[1] * mat1_sizes[2]);
              B.emplace_back(mat2_data + batch * mat2_sizes[1] * mat2_sizes[2]);
              C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
            }
          } else {
            auto mat1_acc = mat1.accessor<Scalar, 3>();
            auto mat2_acc = mat2.accessor<Scalar, 3>();
            for (i64 batch = 0; batch < batch_size; batch++) {
              A.emplace_back(mat1_acc[batch].data());
              B.emplace_back(mat2_acc[batch].data());
              C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
            }
          }

          gemm_batched(trans_A, trans_B, batch_size, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
            */
    }

    pub fn baddbmm_mkl<'a>(
            self_:  &mut Tensor,
            batch1: &Tensor,
            batch2: &Tensor,
            beta:   &Scalar,
            alpha:  &Scalar) -> &'a mut Tensor {
        
        todo!();
            /*
                // checks are done in native/LinearAlgebra.cpp
          AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "baddbmm__mkl", [&] {
              baddbmm_mkl_template<Scalar>(self, batch1, batch2, beta, alpha);
            });

          return self;
            */
    }
}

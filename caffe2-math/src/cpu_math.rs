/*!
  | Implements the math functions for CPU.
  |
  | The implementation in this file allows us to
  | route the underlying numerical computation
  | library to different backends. Notably:
  |
  | -(1) For all BLAS-related functions, one can
  |     explicitly request a BLAS backend such as
  |     MKL, openblas or Atlas. To see the set of
  |     supported backends currently provided,
  |     check //third_party/blas/.
  |
  | -(2) If one chooses to link against MKL, we
  |     utilize MKL's vector math library (VML) for
  |     a few functions such as Exp and Log.
  |
  | -(3) Fallback implementations are provided in
  |     Eigen for cross-platform support. Since
  |     Eigen is a header-only library and supports
  |     a number of platforms, it allows one to
  |     quickly port Caffe2 to different platforms
  |     where BLAS may not be present.
  */

crate::ix!();

/**
  | BLAS alternatives.
  |
  | Depending on whether we have specified an
  | external BLAS library or not, we will delegate
  | the Caffe math functions that are BLAS-related
  | to either the CBLAS call or the Eigen
  | implementation.
  */

/**
  | Caffe2 gemm provides a simpler interface to the
  | gemm functions, with the limitation that the
  | data has to be contiguous in memory.
  |
  | The gemm call implements the following
  | operation:
  |
  |                  C = alpha * op(A) * op(B) + beta * C
  |
  | where op(A) has size M x K, op(B) has size
  | K x N, and C has size M x N. Each of A, B, and
  | C are matrices and alpha and beta are
  | scalars. Note that the most common use case of
  | gemm will involve setting alpha to 1 and beta
  | to 0.
  |
  | op(A) and op(B) represent the transformations
  | that are done to A and B before the matrix
  | multiply; depending on the flags set, op(A) is
  | equal to A or A^T (transpose) if the argument
  | TransA or TransB is set to CblasNoTrans or
  | CblasTrans, respectively, for each of A and B.
  */
#[cfg(caffe2_use_eigen_for_blas)]
#[inline] pub fn gemm_f32cpu_context(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    trans_B:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    k:         i32,
    alpha:     f32,
    a:         *const f32,
    b:         *const f32,
    beta:      f32,
    c:         *mut f32,
    context:   *mut CPUContext,
    math_type: TensorProto_DataType)  {
    
    todo!();
    /*
        auto C_mat = EigenMatrixMap<float>(C, N, M);
      if (beta == 0) {
        C_mat.setZero();
      } else {
        C_mat *= beta;
      }
      switch (trans_A) {
        case CblasNoTrans: {
          switch (trans_B) {
            case CblasNoTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenMatrixMap<float>(B, N, K) *
                   ConstEigenMatrixMap<float>(A, K, M));
              return;
            case CblasTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenMatrixMap<float>(B, K, N).transpose() *
                   ConstEigenMatrixMap<float>(A, K, M));
              return;
            default:
              LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
              return; // The line above calls `abort()`. Should never reach here.
          }
        }
        case CblasTrans: {
          switch (trans_B) {
            case CblasNoTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenMatrixMap<float>(B, N, K) *
                   ConstEigenMatrixMap<float>(A, M, K).transpose());
              return;
            case CblasTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenMatrixMap<float>(B, K, N).transpose() *
                   ConstEigenMatrixMap<float>(A, M, K).transpose());
              return;
            default:
              LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
              return; // The line above calls `abort()`. Should never reach here.
          }
        }
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_A";
      }
    */
}

#[cfg(caffe2_use_eigen_for_blas)]
#[inline] pub fn gemm_ex_f32cpu_context(
    trans_A: cblas_sys::CBLAS_TRANSPOSE,
    trans_B: cblas_sys::CBLAS_TRANSPOSE,
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
    ldc:     i32,
    context: *mut CPUContext)  {
    
    todo!();
    /*
        EigenOuterStridedMatrixMap<float> C_mat(C, N, M, EigenOuterStride(ldc));
      if (beta == 0) {
        C_mat.setZero();
      } else {
        C_mat *= beta;
      }
      switch (trans_A) {
        case CblasNoTrans: {
          switch (trans_B) {
            case CblasNoTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenOuterStridedMatrixMap<float>(
                       B, N, K, EigenOuterStride(ldb)) *
                   ConstEigenOuterStridedMatrixMap<float>(
                       A, K, M, EigenOuterStride(lda)));
              return;
            case CblasTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenOuterStridedMatrixMap<float>(
                       B, K, N, EigenOuterStride(ldb))
                       .transpose() *
                   ConstEigenOuterStridedMatrixMap<float>(
                       A, K, M, EigenOuterStride(lda)));
              return;
            default:
              LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
              return; // The line above calls `abort()`. Should never reach here.
          }
        }
        case CblasTrans: {
          switch (trans_B) {
            case CblasNoTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenOuterStridedMatrixMap<float>(
                       B, N, K, EigenOuterStride(ldb)) *
                   ConstEigenOuterStridedMatrixMap<float>(
                       A, M, K, EigenOuterStride(lda))
                       .transpose());
              return;
            case CblasTrans:
              C_mat.noalias() += alpha *
                  (ConstEigenOuterStridedMatrixMap<float>(
                       B, K, N, EigenOuterStride(ldb))
                       .transpose() *
                   ConstEigenOuterStridedMatrixMap<float>(
                       A, M, K, EigenOuterStride(lda))
                       .transpose());
              return;
            default:
              LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
              return; // The line above calls `abort()`. Should never reach here.
          }
        }
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_A";
      }
    */
}


#[cfg(caffe2_use_eigen_for_blas)]
#[inline] pub fn gemv_f32cpu_context(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    alpha:     f32,
    a:         *const f32,
    x:         *const f32,
    beta:      f32,
    y:         *mut f32,
    context:   *mut CPUContext,
    math_type: TensorProto::DataType)  {
    
    todo!();
    /*
        EigenVectorMap<float> y_vec(y, trans_A == CblasNoTrans ? M : N);
      if (beta == 0) {
        // In Caffe2 we often do a lazy initialization, which may contain NaNs in
        // the float values. As a result, if beta is 0, we explicitly do a setzero.
        y_vec.setZero();
      } else {
        y_vec *= beta;
      }
      switch (trans_A) {
        case CblasNoTrans: {
          y_vec.noalias() += alpha *
              (ConstEigenMatrixMap<float>(A, N, M).transpose() *
               ConstEigenVectorMap<float>(x, N));
          return;
        }
        case CblasTrans: {
          y_vec.noalias() += alpha *
              (ConstEigenMatrixMap<float>(A, N, M) *
               ConstEigenVectorMap<float>(x, M));
          return;
        }
        default:
          LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input.";
      }
    */
}

#[cfg(caffe2_use_eigen_for_blas)]
#[macro_export] macro_rules! caffe2_specialized_dot {
    ($T:ident) => {
        /*
          template <>                                                            
          void Dot<T, CPUContext>(                                    
              const int N, const T* a, const T* b, T* y, CPUContext* context) {  
            *y = ConstEigenVectorMap<T>(a, N).dot(ConstEigenVectorMap<T>(b, N)); 
          }
        */
    }
}

#[cfg(caffe2_use_eigen_for_blas)]
caffe2_specialized_dot![f32];

#[cfg(not(caffe2_use_eigen_for_blas))]
#[inline] pub fn gemm_f32cpu_context(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    trans_B:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    k:         i32,
    alpha:     f32,
    a:         *const f32,
    b:         *const f32,
    beta:      f32,
    c:         *mut f32,
    context:   *mut CPUContext,
    math_type: TensorProto_DataType)  {
    
    todo!();
    /*
        // MKL expects ld? >= 1
      const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
      const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
      cblas_sgemm(
          CblasRowMajor,
          trans_A,
          trans_B,
          M,
          N,
          K,
          alpha,
          A,
          lda,
          B,
          ldb,
          beta,
          C,
          N);
    */
}

#[cfg(not(caffe2_use_eigen_for_blas))]
#[inline] pub fn gemm_ex_f32cpu_context(
    trans_A: cblas_sys::CBLAS_TRANSPOSE,
    trans_B: cblas_sys::CBLAS_TRANSPOSE,
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
    ldc:     i32,
    context: *mut CPUContext)  {
    
    todo!();
    /*
        cblas_sgemm(
          CblasRowMajor,
          trans_A,
          trans_B,
          M,
          N,
          K,
          alpha,
          A,
          lda,
          B,
          ldb,
          beta,
          C,
          ldc);
    */
}

#[cfg(not(caffe2_use_eigen_for_blas))]
#[inline] pub fn gemv_f32cpu_context(
    trans_A:   cblas_sys::CBLAS_TRANSPOSE,
    m:         i32,
    n:         i32,
    alpha:     f32,
    a:         *const f32,
    x:         *const f32,
    beta:      f32,
    y:         *mut f32,
    context:   *mut CPUContext,
    math_type: TensorProto_DataType)  {
    
    todo!();
    /*
        cblas_sgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
    */
}


#[cfg(not(caffe2_use_eigen_for_blas))]
#[macro_export] macro_rules! caffe2_specialized_dot {
    ($T:ident, $prefix:ident) => {
        /*
          template <>                                                   
          void Dot<T, CPUContext>(                           
              const int N, const T* a, const T* b, T* y, CPUContext*) { 
            *y = cblas_##prefix##dot(N, a, 1, b, 1);                    
          }
        */
    }
}

#[cfg(not(caffe2_use_eigen_for_blas))]
caffe2_specialized_dot![f32, s];

#[inline] pub fn gemm_batched_f32cpu_context(
    trans_A:    cblas_sys::CBLAS_TRANSPOSE,
    trans_B:    cblas_sys::CBLAS_TRANSPOSE,
    batch_size: i32,
    m:          i32,
    n:          i32,
    k:          i32,
    alpha:      f32,
    a:          *const *const f32,
    b:          *const *const f32,
    beta:       f32,
    c:          *mut *mut f32,
    context:    *mut CPUContext,
    math_type:  TensorProto_DataType)  {
    
    todo!();
    /*
        #ifdef CAFFE2_USE_MKL
      (void)context;
      // MKL expects ld? >= 1
      const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
      const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
      const int ldc = std::max(N, 1);
      cblas_sgemm_batch(
          CblasRowMajor,
          &trans_A,
          &trans_B,
          &M,
          &N,
          &K,
          &alpha,
          A,
          &lda,
          B,
          &ldb,
          &beta,
          C,
          &ldc,
          1,
          &batch_size);
    #else // CAFFE2_USE_MKL
      // loop over matrices in the batch
      for (int i = 0; i < batch_size; ++i) {
        math::Gemm<float, CPUContext>(
            trans_A, trans_B, M, N, K, alpha, A[i], B[i], beta, C[i], context);
      }
    #endif // CAFFE2_USE_MKL
    */
}

#[inline] pub fn gemm_strided_batched_f32cpu_context(
    trans_A:    cblas_sys::CBLAS_TRANSPOSE,
    trans_B:    cblas_sys::CBLAS_TRANSPOSE,
    batch_size: i32,
    m:          i32,
    n:          i32,
    k:          i32,
    alpha:      f32,
    a:          *const f32,
    a_stride:   i32,
    b:          *const f32,
    b_stride:   i32,
    beta:       f32,
    c:          *mut f32,
    c_stride:   i32,
    context:    *mut CPUContext,
    math_type:  TensorProto_DataType)  {
    
    todo!();
    /*
        #ifdef CAFFE2_USE_MKL
      (void)context;
      // MKL expects ld? >= 1
      const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
      const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
      const int ldc = std::max(N, 1);
      std::vector<const float*> A_array(batch_size);
      std::vector<const float*> B_array(batch_size);
      std::vector<float*> C_array(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        A_array[i] = A + i * A_stride;
        B_array[i] = B + i * B_stride;
        C_array[i] = C + i * C_stride;
      }
      cblas_sgemm_batch(
          CblasRowMajor,
          &trans_A,
          &trans_B,
          &M,
          &N,
          &K,
          &alpha,
          A_array.data(),
          &lda,
          B_array.data(),
          &ldb,
          &beta,
          C_array.data(),
          &ldc,
          1,
          &batch_size);
    #else // CAFFE2_USE_MKL
      // loop over matrices in the batch
      for (int i = 0; i < batch_size; ++i) {
        math::Gemm<float, CPUContext>(
            trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context);
        A += A_stride;
        B += B_stride;
        C += C_stride;
      }
    #endif
    */
}

/**
  | Common math functions being used in Caffe that
  | do not have a BLAS or MKL equivalent. For all
  | these functions, we will simply implement them
  | either via Eigen or via custom code.
  */
#[inline] pub fn broadcast_impl<T>(
    x_ndim:  i32,
    x_dims:  *const i32,
    y_ndim:  i32,
    y_dims:  *const i32,
    alpha:   T,
    x:       *const T,
    y:       *mut T,
    context: *mut CPUContext)  {
    todo!();
    /*
        CAFFE_ENFORCE_LE(X_ndim, Y_ndim);
      std::vector<int> X_dims_vector(Y_ndim);
      const int d = Y_ndim - X_ndim;
      std::fill(X_dims_vector.begin(), X_dims_vector.begin() + d, 1);
      for (int i = d; i < Y_ndim; ++i) {
        CAFFE_ENFORCE(X_dims[i - d] == 1 || X_dims[i - d] == Y_dims[i]);
        X_dims_vector[i] = X_dims[i - d];
      }
      X_dims = X_dims_vector.data();
      const int Y_size =
          std::accumulate(Y_dims, Y_dims + Y_ndim, 1, std::multiplies<int>());
      std::vector<int> index(Y_ndim, 0);
      for (int Y_index = 0; Y_index < Y_size; ++Y_index) {
        const int X_index = utils::GetIndexFromDims(Y_ndim, X_dims, index.data());
        Y[Y_index] = X[X_index];
        utils::IncreaseIndexInDims(Y_ndim, Y_dims, index.data());
      }
      Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
    */
}

#[macro_export] macro_rules! caffe2_specialized_broadcast {
    ($T:ident) => {
        /*
  template <>                                                               
  void Broadcast<T, CPUContext>(                                 
      const int X_ndim,                                                     
      const int* X_dims,                                                    
      const int Y_ndim,                                                     
      const int* Y_dims,                                                    
      const T alpha,                                                        
      const T* X,                                                           
      T* Y,                                                                 
      CPUContext* context) {                                                
    BroadcastImpl<T>(X_ndim, X_dims, Y_ndim, Y_dims, alpha, X, Y, context); 
  }
  */
    }
}

caffe2_specialized_broadcast!{i32}
caffe2_specialized_broadcast!{i64}
caffe2_specialized_broadcast!{f32}
caffe2_specialized_broadcast!{f64}

#[macro_export] macro_rules! caffe2_specialized_inv_std {
    ($T:ident) => {
        /*
  template <>                                                    
  void InvStd<T, CPUContext>(                                    
      const int N,                                               
      const T epsilon,                                           
      const T* var,                                              
      T* inv_std,                                                
      CPUContext* context) {                                     
    EigenVectorArrayMap<T>(inv_std, N) =                         
        (ConstEigenVectorArrayMap<T>(var, N) + epsilon).rsqrt(); 
  }
  */
    }
}

caffe2_specialized_inv_std!{f32}

#[macro_export] macro_rules! caffe2_specialized_rowwisemax {
    ($T:ident) => {
        /*
  template <>                                                    
  void RowwiseMax<T, CPUContext>(                     
      const int N, const int D, const T* x, T* y, CPUContext*) { 
    EigenVectorMap<T>(y, N) =                                    
        ConstEigenMatrixMap<T>(x, D, N).colwise().maxCoeff();    
  }
  */
    }
}

caffe2_specialized_rowwisemax!{f32}

#[macro_export] macro_rules! caffe2_specialized_colwisemax {
    ($T:ident) => {
        /*
  template <>                                                    
  void ColwiseMax<T, CPUContext>(                     
      const int N, const int D, const T* x, T* y, CPUContext*) { 
    EigenVectorMap<T>(y, D) =                                    
        ConstEigenMatrixMap<T>(x, D, N).rowwise().maxCoeff();    
  }
  */
    }
}

caffe2_specialized_colwisemax!{f32}

#[macro_export] macro_rules! caffe2_specialized_maximum {
    ($T:ident) => {
        /*
  template <>                                                                  
  void Maximum<T, CPUContext>(                                      
      const int N, const float alpha, const T* x, T* y, CPUContext* context) { 
    std::transform(                                                            
        x, x + N, y, [&alpha](const T& x_i) { return std::max(x_i, alpha); }); 
  }
  */
    }
}

caffe2_specialized_maximum!{f32}

/**
  | The actual implementation uses eigen
  | which is column major, so notice the
  | row/column swap in the actual implementation.
  |
  */
#[macro_export] macro_rules! delegate_eigen_2d_broadcast_1st_binary_function {
    ($T:ident, Func, expr) => {
        /*
  template <>                                                          
  void Rowwise##Func<T, CPUContext, true>(                  
      const int rows,                                                  
      const int cols,                                                  
      const T* A,                                                      
      const T* B,                                                      
      T* C,                                                            
      CPUContext*) {                                                   
    if (C == B) {                                                      
      EigenArrayMap<T>(C, cols, rows).colwise() expr## =               
          ConstEigenVectorArrayMap<T>(A, cols);                        
    } else {                                                           
      EigenArrayMap<T>(C, cols, rows) =                                
          ConstEigenArrayMap<T>(B, cols, rows)                         
              .colwise() expr ConstEigenVectorArrayMap<T>(A, cols);    
    }                                                                  
  }                                                                    
  template <>                                                          
  void Colwise##Func<T, CPUContext, true>(                  
      const int rows,                                                  
      const int cols,                                                  
      const T* A,                                                      
      const T* B,                                                      
      T* C,                                                            
      CPUContext*) {                                                   
    if (C == B) {                                                      
      EigenArrayMap<T>(C, cols, rows).rowwise() expr## =               
          ConstEigenVectorArrayMap<T>(A, rows).transpose();            
    } else {                                                           
      EigenArrayMap<T>(C, cols, rows) =                                
          ConstEigenArrayMap<T>(B, cols, rows)                         
              .rowwise() expr ConstEigenVectorArrayMap<T>(A, rows)     
              .transpose();                                            
    }                                                                  
  }
  */
    }
}

#[macro_export] macro_rules! delegate_eigen_2d_broadcast_2nd_binary_function {
    ($T:ident, $Func:ident, $expr:tt) => {
        /*
          template <>                                                          
          void Rowwise##Func<T, CPUContext, false>(                 
              const int rows,                                                  
              const int cols,                                                  
              const T* A,                                                      
              const T* B,                                                      
              T* C,                                                            
              CPUContext*) {                                                   
            if (C == A) {                                                      
              EigenArrayMap<T>(C, cols, rows).colwise() expr## =               
                  ConstEigenVectorArrayMap<T>(B, cols);                        
            } else {                                                           
              EigenArrayMap<T>(C, cols, rows) =                                
                  ConstEigenArrayMap<T>(A, cols, rows)                         
                      .colwise() expr ConstEigenVectorArrayMap<T>(B, cols);    
            }                                                                  
          }                                                                    
          template <>                                                          
          void Colwise##Func<T, CPUContext, false>(                 
              const int rows,                                                  
              const int cols,                                                  
              const T* A,                                                      
              const T* B,                                                      
              T* C,                                                            
              CPUContext*) {                                                   
            if (C == A) {                                                      
              EigenArrayMap<T>(C, cols, rows).rowwise() expr## =               
                  ConstEigenVectorArrayMap<T>(B, rows).transpose();            
            } else {                                                           
              EigenArrayMap<T>(C, cols, rows) =                                
                  ConstEigenArrayMap<T>(A, cols, rows)                         
                      .rowwise() expr ConstEigenVectorArrayMap<T>(B, rows)     
                      .transpose();                                            
            }                                                                  
          }
        */
    }
}

#[macro_export] macro_rules! delegate_eigen_2d_broadcast_binary_function {
    ($T:ident, $Func:ident, $expr:ident) => {
        /*
          DELEGATE_EIGEN_2D_BROADCAST_1ST_BINARY_FUNCTION(T, Func, expr)   
          DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Func, expr)
          */
    }
}

#[macro_export] macro_rules! define_eigen_2d_broadcast_binary_function {
    ($Func:ident, $expr:tt) => {
        /*
          DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(float, Func, expr)        
          DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(double, Func, expr)       
          DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, Func, expr) 
          DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, Func, expr)
        */
    }
}

define_eigen_2d_broadcast_binary_function!{Add, +}

define_eigen_2d_broadcast_binary_function!{Mul, *}

#[macro_export] macro_rules! define_eigen_2d_broadcast_sub_function {
    ($x:ident) => { 
        /*
        (T) => {

      template <>                                               
      void RowwiseSub<T, CPUContext, true>(          
          const int rows,                                       
          const int cols,                                       
          const T* A,                                           
          const T* B,                                           
          T* C,                                                 
          CPUContext*) {                                        
        EigenArrayMap<T>(C, cols, rows) =                       
            (-ConstEigenArrayMap<T>(B, cols, rows)).colwise() + 
            ConstEigenVectorArrayMap<T>(A, cols);               
      }                                                         
      template <>                                               
      void ColwiseSub<T, CPUContext, true>(          
          const int rows,                                       
          const int cols,                                       
          const T* A,                                           
          const T* B,                                           
          T* C,                                                 
          CPUContext*) {                                        
        EigenArrayMap<T>(C, cols, rows) =                       
            (-ConstEigenArrayMap<T>(B, cols, rows)).rowwise() + 
            ConstEigenVectorArrayMap<T>(A, rows).transpose();   
      }                                                         
      DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Sub, -)

          */
    }
}

define_eigen_2d_broadcast_sub_function!{f32}
define_eigen_2d_broadcast_sub_function!{f64}
define_eigen_2d_broadcast_sub_function!{i32}
define_eigen_2d_broadcast_sub_function!{i64}

#[macro_export] macro_rules! define_eigen_2d_broadcast_div_function {
    ($x:ident) => {
        /*
        (T) => {

          template <>                                                      
          void RowwiseDiv<T, CPUContext, true>(                 
              const int rows,                                              
              const int cols,                                              
              const T* A,                                                  
              const T* B,                                                  
              T* C,                                                        
              CPUContext*) {                                               
            EigenArrayMap<T>(C, cols, rows) =                              
                ConstEigenArrayMap<T>(B, cols, rows).inverse().colwise() * 
                ConstEigenVectorArrayMap<T>(A, cols);                      
          }                                                                
          template <>                                                      
          void ColwiseDiv<T, CPUContext, true>(                 
              const int rows,                                              
              const int cols,                                              
              const T* A,                                                  
              const T* B,                                                  
              T* C,                                                        
              CPUContext*) {                                               
            EigenArrayMap<T>(C, cols, rows) =                              
                ConstEigenArrayMap<T>(B, cols, rows).inverse().rowwise() * 
                ConstEigenVectorArrayMap<T>(A, rows).transpose();          
          }                                                                
          DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Div, /)

          */
    }
}

define_eigen_2d_broadcast_div_function!{f32}
define_eigen_2d_broadcast_div_function!{f64}
delegate_eigen_2d_broadcast_2nd_binary_function!{i32, Div, /}
delegate_eigen_2d_broadcast_2nd_binary_function!{i64, Div, /}

#[inline] pub fn not_bool_cpu_context(
    n:       i32,
    x:       *const bool,
    y:       *mut bool,
    context: *mut CPUContext)  {
    
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        y[i] = !x[i];
      }
    */
}

#[macro_export] macro_rules! caffe2_specialized_cpu_add_striped_batch {
    ($T:ident) => {
        /*

  template <>                                                   
  void AddStripedBatch(                              
      const int N,                                              
      const T* first,                                           
      T* y,                                                     
      const int stripe,                                         
      const int batch,                                          
      CPUContext* context) {                                    
    for (int j = 0; j < batch; j++) {                           
      Add<T, CPUContext>(N, first + j * stripe, y, y, context); 
    }                                                           
  }

  */
    }
}

caffe2_specialized_cpu_add_striped_batch!{f32}

#[inline] pub fn rowwise_binary_op<TIn, TOut, BinaryOperator, const kBroadcast1st: bool>(
    rows: i32,
    cols: i32,
    op:   &BinaryOperator,
    a:    *const TIn,
    b:    *const TIn,
    c:    *mut TOut)  {
    todo!();
    /*
        for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          const int C_index = i * cols + j;
          const int A_index = kBroadcast1st ? j : C_index;
          const int B_index = kBroadcast1st ? C_index : j;
          C[C_index] = op(A[A_index], B[B_index]);
        }
      }
    */
}

#[inline] pub fn colwise_binary_op<TIn, TOut, BinaryOperator, const kBroadcast1st: bool>(
    rows: i32,
    cols: i32,
    op:   &BinaryOperator,
    a:    *const TIn,
    b:    *const TIn,
    c:    *mut TOut)  {
    todo!();
    /*
        for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          const int C_index = i * cols + j;
          const int A_index = kBroadcast1st ? i : C_index;
          const int B_index = kBroadcast1st ? C_index : i;
          C[C_index] = op(A[A_index], B[B_index]);
        }
      }
    */
}

#[inline] pub fn broadcast_binary_op_impl<TIn, TOut, BinaryOperator>(
    ndim:   i32,
    a_dims: *const i32,
    b_dims: *const i32,
    c_dims: *const i32,
    op:     &BinaryOperator,
    a:      *const TIn,
    b:      *const TIn,
    c:      *mut TOut)  {
    todo!();
    /*
        std::vector<int> index(ndim, 0);
      const int C_size =
          std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
      for (int C_index = 0; C_index < C_size; ++C_index) {
        const int A_index = utils::GetIndexFromDims(ndim, A_dims, index.data());
        const int B_index = utils::GetIndexFromDims(ndim, B_dims, index.data());
        C[C_index] = op(A[A_index], B[B_index]);
        utils::IncreaseIndexInDims(ndim, C_dims, index.data());
      }
    */
}

#[macro_export] macro_rules! delegate_2d_broadcast_binary_function {
    ($T:ty, $In:ty, $TOut:tt, $Func:tt $(, $Op:tt)*) => {
        /*

  template <>                                                                  
  void Rowwise##Func<TIn, CPUContext, true>(                        
      const int rows,                                                          
      const int cols,                                                          
      const TIn* A,                                                            
      const TIn* B,                                                            
      TOut* C,                                                                 
      CPUContext*) {                                                           
    RowwiseBinaryOp<TIn, TOut, Op<TIn>, true>(rows, cols, Op<TIn>(), A, B, C); 
  }                                                                            
  template <>                                                                  
  void Rowwise##Func<TIn, CPUContext, false>(                       
      const int rows,                                                          
      const int cols,                                                          
      const TIn* A,                                                            
      const TIn* B,                                                            
      TOut* C,                                                                 
      CPUContext*) {                                                           
    RowwiseBinaryOp<TIn, TOut, Op<TIn>, false>(                                
        rows, cols, Op<TIn>(), A, B, C);                                       
  }                                                                            
  template <>                                                                  
  void Colwise##Func<TIn, CPUContext, true>(                        
      const int rows,                                                          
      const int cols,                                                          
      const TIn* A,                                                            
      const TIn* B,                                                            
      TOut* C,                                                                 
      CPUContext*) {                                                           
    ColwiseBinaryOp<TIn, TOut, Op<TIn>, true>(rows, cols, Op<TIn>(), A, B, C); 
  }                                                                            
  template <>                                                                  
  void Colwise##Func<TIn, CPUContext, false>(                       
      const int rows,                                                          
      const int cols,                                                          
      const TIn* A,                                                            
      const TIn* B,                                                            
      TOut* C,                                                                 
      CPUContext*) {                                                           
    ColwiseBinaryOp<TIn, TOut, Op<TIn>, false>(                                
        rows, cols, Op<TIn>(), A, B, C);                                       
  }

  */
    }
}

#[macro_export] macro_rules! define_2d_compare_function{
    ($Func:ident, $Op:ident) => {
        /*
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(float, bool, Func, Op)        
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(double, bool, Func, Op)       
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, bool, Func, Op) 
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, bool, Func, Op) 
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)
        */
    }
}

define_2d_compare_function!{EQ, equal_to}
define_2d_compare_function!{NE, not_equal_to}
define_2d_compare_function!{LT, less}
define_2d_compare_function!{LE, less_equal}
define_2d_compare_function!{GT, greater}
define_2d_compare_function!{GE, greater_equal}

delegate_2d_broadcast_binary_function!{bool, bool, And, logical_and}
delegate_2d_broadcast_binary_function!{bool, bool, Or, logical_or}
delegate_2d_broadcast_binary_function!{bool, bool, Xor, bit_xor}

#[macro_export] macro_rules! define_2d_broadcast_bitwise_binary_function {
    ($Func:ident, $Op:ident) => {
        /*
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)                 
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) 
          DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)
        */
    }
}

define_2d_broadcast_bitwise_binary_function!{BitwiseAnd, bit_and}
define_2d_broadcast_bitwise_binary_function!{BitwiseOr,  bit_or}
define_2d_broadcast_bitwise_binary_function!{BitwiseXor, bit_xor}

#[macro_export] macro_rules! define_2d_broadcast_1st_div_function {
    ($T:ident) => {
        /*

  template <>                                      
  void RowwiseDiv<T, CPUContext, true>( 
      const int rows,                              
      const int cols,                              
      const T* A,                                  
      const T* B,                                  
      T* C,                                        
      CPUContext*) {                               
    RowwiseBinaryOp<T, T, std::divides<T>, true>(  
        rows, cols, std::divides<T>(), A, B, C);   
  }                                                
  template <>                                      
  void ColwiseDiv<T, CPUContext, true>( 
      const int rows,                              
      const int cols,                              
      const T* A,                                  
      const T* B,                                  
      T* C,                                        
      CPUContext*) {                               
    ColwiseBinaryOp<T, T, std::divides<T>, true>(  
        rows, cols, std::divides<T>(), A, B, C);   
  }
  */
    }
}

define_2d_broadcast_1st_div_function!{i32}
define_2d_broadcast_1st_div_function!{i64}

#[macro_export] macro_rules! delegate_broadcast_binary_function {
    ($T:ident, $In:ident, $TOut:ident, $Func:ident $(,$Op:ident)*) => {
        /*

  template <>                                                                
  void Func<TIn, CPUContext>(                                     
      const int A_ndim,                                                      
      const int* A_dims,                                                     
      const int B_ndim,                                                      
      const int* B_dims,                                                     
      const TIn* A,                                                          
      const TIn* B,                                                          
      TOut* C,                                                               
      CPUContext* context) {                                                 
    const int ndim = std::max(A_ndim, B_ndim);                               
    std::vector<int> A_dims_array(ndim);                                     
    std::vector<int> B_dims_array(ndim);                                     
    std::vector<int> C_dims_array(ndim);                                     
    utils::ComputeBroadcastBinaryOpDims(                                     
        A_ndim,                                                              
        A_dims,                                                              
        B_ndim,                                                              
        B_dims,                                                              
        A_dims_array.data(),                                                 
        B_dims_array.data(),                                                 
        C_dims_array.data());                                                
    if (A_dims_array == B_dims_array) {                                      
      const int size = std::accumulate(                                      
          C_dims_array.cbegin(),                                             
          C_dims_array.cend(),                                               
          1,                                                                 
          std::multiplies<int>());                                           
      Func<TIn, CPUContext>(size, A, B, C, context);                         
      return;                                                                
    }                                                                        
    int rows;                                                                
    int cols;                                                                
    bool broadcast_1st;                                                      
    if (utils::IsRowwiseBroadcastBinaryOp(                                   
            ndim,                                                            
            A_dims_array.data(),                                             
            B_dims_array.data(),                                             
            &rows,                                                           
            &cols,                                                           
            &broadcast_1st)) {                                               
      if (broadcast_1st) {                                                   
        Rowwise##Func<TIn, CPUContext, true>(rows, cols, A, B, C, context);  
      } else {                                                               
        Rowwise##Func<TIn, CPUContext, false>(rows, cols, A, B, C, context); 
      }                                                                      
      return;                                                                
    }                                                                        
    if (utils::IsColwiseBroadcastBinaryOp(                                   
            ndim,                                                            
            A_dims_array.data(),                                             
            B_dims_array.data(),                                             
            &rows,                                                           
            &cols,                                                           
            &broadcast_1st)) {                                               
      if (broadcast_1st) {                                                   
        Colwise##Func<TIn, CPUContext, true>(rows, cols, A, B, C, context);  
      } else {                                                               
        Colwise##Func<TIn, CPUContext, false>(rows, cols, A, B, C, context); 
      }                                                                      
      return;                                                                
    }                                                                        
    int pre;                                                                 
    int mid;                                                                 
    int nxt;                                                                 
    if (utils::IsBothEndsBroadcastBinaryOp(                                  
            ndim,                                                            
            A_dims_array.data(),                                             
            B_dims_array.data(),                                             
            &pre,                                                            
            &mid,                                                            
            &nxt,                                                            
            &broadcast_1st)) {                                               
      const int stride = mid * nxt;                                          
      for (int i = 0; i < pre; ++i) {                                        
        if (broadcast_1st) {                                                 
          Colwise##Func<TIn, CPUContext, true>(                              
              mid, nxt, A, B + i * stride, C + i * stride, context);         
        } else {                                                             
          Colwise##Func<TIn, CPUContext, false>(                             
              mid, nxt, A + i * stride, B, C + i * stride, context);         
        }                                                                    
      }                                                                      
      return;                                                                
    }                                                                        
    BroadcastBinaryOpImpl(                                                   
        ndim,                                                                
        A_dims_array.data(),                                                 
        B_dims_array.data(),                                                 
        C_dims_array.data(),                                                 
        Op<TIn>(),                                                           
        A,                                                                   
        B,                                                                   
        C);                                                                  
  }
  */
    }
}

#[macro_export] macro_rules! define_broadcast_compare_function {
    ($Func:ident, $Op:ident) => {
        /*
          DELEGATE_BROADCAST_BINARY_FUNCTION(float, bool, Func, Op)        
          DELEGATE_BROADCAST_BINARY_FUNCTION(double, bool, Func, Op)       
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, bool, Func, Op) 
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, bool, Func, Op) 
          DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)
        */
    }
}

define_broadcast_compare_function!{EQ, equal_to}
define_broadcast_compare_function!{NE, not_equal_to}
define_broadcast_compare_function!{LT, less}
define_broadcast_compare_function!{LE, less_equal}
define_broadcast_compare_function!{GT, greater}
define_broadcast_compare_function!{GE, greater_equal}

#[macro_export] macro_rules! define_broadcast_binary_function {
    ($Func:ident, $Op:ident) => {
        /*
          DELEGATE_BROADCAST_BINARY_FUNCTION(float, float, Func, Op)               
          DELEGATE_BROADCAST_BINARY_FUNCTION(double, double, Func, Op)             
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) 
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)
        */
    }
}

define_broadcast_binary_function!{Add, plus}
define_broadcast_binary_function!{Sub, minus}
define_broadcast_binary_function!{Mul, multiplies}
define_broadcast_binary_function!{Div, divides}

delegate_broadcast_binary_function!{bool, bool, And, logical_and}
delegate_broadcast_binary_function!{bool, bool, Or,  logical_or}
delegate_broadcast_binary_function!{bool, bool, Xor, bit_xor}

#[macro_export] macro_rules! define_broadcast_bitwise_binary_function {
    ($Func:ident, $Op:ident) => {
        /*
          DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)                 
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) 
          DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)
        */
    }
}

define_broadcast_bitwise_binary_function!{BitwiseAnd, bit_and}
define_broadcast_bitwise_binary_function!{BitwiseOr,  bit_or}
define_broadcast_bitwise_binary_function!{BitwiseXor, bit_xor}

/**
  | incrementIfNotMax increments the
  | number if the value is not max for that
  | datatype. This ensures that the value
  | never overflows.
  |
  */
#[inline] pub fn increment_if_not_max<T>(a: T) -> T {
    todo!();
    /*
        if (a == T::max) {
        return a;
      }
      return a + 1;
    */
}

#[macro_export] macro_rules! caffe2_rand_uniform_real {
    ($T:ident) => {
        /*

  template <>                                                            
  void RandUniform<T, CPUContext>(                            
      const size_t n, const T a, const T b, T* r, CPUContext* context) { 
    at::uniform_real_distribution<T> distribution(a, b);                 
    for (size_t i = 0; i < n; ++i) {                                     
      r[i] = distribution(context->RandGenerator());                     
    }                                                                    
  }
  */
    }
}

caffe2_rand_uniform_real!{f32}
caffe2_rand_uniform_real!{f64}

#[macro_export] macro_rules! caffe2_rand_uniform_char {
    ($T:ident) => {
        /*

  template <>                                                            
  void RandUniform<T, CPUContext>(                            
      const size_t n, const T a, const T b, T* r, CPUContext* context) { 
    at::uniform_int_from_to_distribution<short> distribution(            
        incrementIfNotMax(b - a), a);                                    
    for (size_t i = 0; i < n; ++i) {                                     
      r[i] = static_cast<T>(distribution(context->RandGenerator()));     
    }                                                                    
  }
  */
    }
}

caffe2_rand_uniform_char!{i8}
caffe2_rand_uniform_char!{u8}

#[macro_export] macro_rules! caffe2_rand_uniform_int {
    ($T:ident) => {
        /*

  template <>                                                            
  void RandUniform<T, CPUContext>(                            
      const size_t n, const T a, const T b, T* r, CPUContext* context) { 
    at::uniform_int_from_to_distribution<T> distribution(                
        incrementIfNotMax(                                               
            static_cast<uint64_t>(b) - static_cast<uint64_t>(a)),        
        a);                                                              
    for (size_t i = 0; i < n; ++i) {                                     
      r[i] = distribution(context->RandGenerator());                     
    }                                                                    
  }

  */
    }
}

caffe2_rand_uniform_int!{i16}
caffe2_rand_uniform_int!{i32}
caffe2_rand_uniform_int!{i64}
caffe2_rand_uniform_int!{u16}
caffe2_rand_uniform_int!{u32}
caffe2_rand_uniform_int!{u64}

/**
  | This is not uniformly distributed between a and
  | b.
  |
  | It takes advantage of normal distribution to
  | generate numbers with mean = sum / n.
  |
  | Ideally the algorithm should be generating
  | n numbers between 0 and 1, sum them up as
  | scaled_sum, and use sum / scaled_sum to adjust
  | the values to between a and b.
  |
  | The algorithm is non-trivial given the
  | adjustment would be different towards each
  | value.
  */
#[macro_export] macro_rules! caffe2_rand_fixed_sum {
    ($T:ident) => {
        /*

  template <>                                                             
  void RandFixedSum<T, CPUContext>(                            
      const size_t n,                                                     
      const T a,                                                          
      const T b,                                                          
      const T sum,                                                        
      T* r,                                                               
      CPUContext* context) {                                              
    CAFFE_ENFORCE_GE(a, 0);                                               
    CAFFE_ENFORCE_GE(sum / (double)n, a);                                 
    CAFFE_ENFORCE_LE(sum / (double)n, b);                                 
    T current_sum = 0;                                                    
    T remaining_sum = sum;                                                
    for (size_t i = 0; i < n; ++i) {                                      
      auto remaining_numbers = n - 1 - i;                                 
      double mean = (sum - current_sum) / (remaining_numbers + 1);        
      double stdev = std::min(mean - a, b - mean);                        
      at::normal_distribution<double> distribution{mean, stdev / 4.0};    
      T value, remaining_sum_test;                                        
      do {                                                                
        value = distribution(context->RandGenerator());                   
        remaining_sum_test = remaining_sum - value;                       
      } while (value < a || remaining_sum_test < a * remaining_numbers || 
               value > b || remaining_sum_test > b * remaining_numbers);  
      r[i] = value;                                                       
      CAFFE_ENFORCE(a <= value && value <= b);                            
      current_sum += value;                                               
      remaining_sum -= value;                                             
      CAFFE_ENFORCE_GE(remaining_sum, a* remaining_numbers);              
      CAFFE_ENFORCE_LE(remaining_sum, b* remaining_numbers);              
    }                                                                     
    r[n - 1] += remaining_sum;                                            
    current_sum += remaining_sum;                                         
    CAFFE_ENFORCE(a <= r[n - 1] && r[n - 1] <= b);                        
    CAFFE_ENFORCE_EQ(current_sum, sum);                                   
  }
  */
    }
}

caffe2_rand_fixed_sum!{f32}
caffe2_rand_fixed_sum!{f64}
caffe2_rand_fixed_sum!{i8}
caffe2_rand_fixed_sum!{i16}
caffe2_rand_fixed_sum!{i32}
caffe2_rand_fixed_sum!{i64}
caffe2_rand_fixed_sum!{u8}
caffe2_rand_fixed_sum!{u16}
caffe2_rand_fixed_sum!{u32}
caffe2_rand_fixed_sum!{u64}

#[inline] pub fn generate_stack_distance<Type, ValueT, IndexT, ContextT, const cdf_app: bool>(
    cum_val: &mut Vec<IndexT>,
    cum_dis: &mut Vec<ValueT>,
    cum_map: &mut Vec<IndexT>,
    max_i:   IndexT,
    i:       IndexT,
    context: *mut ContextT) -> IndexT {
    todo!();
    /*
        /* Description:
         Inverse Transform Sampling method to generate values for random variable X
         that is described by the cumulative distribution F (cum_val,cum_dis).
         Notice, that we may choose to use the inverse map of F (cum_map) as an
         approximation to avoid searching. Also, scaling the probability so that
         the values are within max_i refs, because stack distance can not be >
         than the # of already generated refs (max_i).
      */
      Ind_t j, k, n;
      Val_t u, f, fi;

      // generate a random number u in [0,1] from a uniform distribution U
      math::RandUniform<Val_t, Context_t>(1, 0, 1, &u, context);

      // scale the random number u to be within range [0,f(i)], if needed
      if (i < max_i) {
        // approach 2: allows gaps in the distribution
        j = (std::upper_bound(cum_val.begin(), cum_val.end(), i) -
             cum_val.begin()) -
            1;
        fi = cum_dis[j];
        u *= fi;
      }
      // 2. compute the stack distance value of x, s.t. F(x)=u
      // notice that the cumulative distribution F increases monotonically up to 1
      if (cdf_app) {
        // look up cum_val corresponding to u <= cum_dis[j]
        k = cum_map.size();
        n = (Ind_t)round(u * k);
        j = cum_map[n];
        return cum_val[j];
      } else {
        // iterate until you find the cum_val corresponding to u <= cum_dis[j]
        for (j = 0; j < Ind_t(cum_dis.size()); j++) {
          f = cum_dis[j];
          if (u <= f) {
            return cum_val[j];
          }
        }
        return cum_val[j - 1];
      }
    */
}

#[inline] pub fn generate_trace_lru<Type, ValueT, IndexT, ContextT, const cdf_app: bool>(
    uni_ref:         &mut Vec<IndexT>,
    cum_val:         &mut Vec<IndexT>,
    cum_dis:         &mut Vec<ValueT>,
    cum_map:         &mut Vec<IndexT>,
    context:         *mut ContextT,
    cache_line_size: IndexT,
    n:               IndexT,
    min:             Type,
    max:             Type,
    syn_ref:         *mut Type)
{
    todo!();
    /*
        /* Description:
         Generate synthetic trace from a list of unique accesses uni_ref, and
         cumulative distribution of distances (cum_val,cum_dis) between them.
         Also, there is an option to use cum_map approximation to avoid searching.
      */
      Ind_t i, j, k, sd, line_ref, mem_ref, mem_ref_within_line;
      Ind_t max_sd = cum_val.back();
      Ind_t l = uni_ref.size();

      for (i = 0, j = 0; j < n; j++) {
        // generate stack distance
        sd = generate_stack_distance<Type, Val_t, Ind_t, Context_t, cdf_app>(
            cum_val, cum_dis, cum_map, max_sd, i, context);
        // fixed access within cache line
        mem_ref_within_line = 0;
        // random access within cache line
        // Val_t r;
        // math::RandUniform<Val_t, Context_t>(1, 0, 1, &r, context);
        // mem_ref_within_line = floor(r*cache_line_size);

        // generate memory reference
        if (sd == 0) {
          k = 0; /// new reference ///
          i++;
        } else {
          k = l - sd; /// existing reference ///
        }
        line_ref = uni_ref[k]; // pop k-th element
        uni_ref.erase(uni_ref.begin() + k);
        uni_ref.push_back(line_ref); // append it back
        mem_ref = line_ref * cache_line_size + mem_ref_within_line;
        /*
        //debug prints
        if ((mem_ref < min) || (mem_ref > max)) {
          //printf("mem_ref[%d]=%d (%ld) \n",j,mem_ref,syn_ref[j]);
          std::cout << "syn_ref[" << j << "]=" << (Type)mem_ref << " ";
          std::cout << "(" << mem_ref << ") ";
          std::cout << "[" << min << "," << max << "]" << std::endl;
          int scanf_temp;
          scanf("%d",&scanf_temp);
        }
        */

        // patch mem_ref to be within range
        // WARNING: this should not be needed if instantiation type and distribution
        // choice is correct. It is remeding a symptom of earlier mistakes.
        if (mem_ref < min) {
          mem_ref = min;
          // std::cout << "clamping (min) mem_ref=" << mem_ref << std::endl;
        }
        if (mem_ref > max) {
          mem_ref = max; // mem_ref % max;
          // std::cout << "clamping (max) mem_ref=" << mem_ref << std::endl;
        }

        // save generated memory reference
        syn_ref[j] = (Type)mem_ref;
      }
    */
}

/**
  | Generate n values from synthetic data
  | distribution, define by unique accesses and
  | stack distances
  |
  | WARNING: can create this for all tables or per
  | table, but in latter case we need to know the
  | table id, to sample from the right distribution
  */
#[macro_export] macro_rules! caffe2_rand_synthetic_data {
    ($T:ident) => {
        /*
  template <>                                                                 
  void RandSyntheticData<T, CPUContext>(                           
      const size_t n, const T a, const T b, T* r, CPUContext* context) {      
    /* unique memory references */                                            
    std::vector<int> mem_ref = {1, 2, 3, 4, 5, 6};                            
    /* cumulative distribution of distances */                                
    std::vector<int> cum_val = {0, 1, 3, 4, 5};                               
    std::vector<double> cum_dis = {0.55, 0.64, 0.82, 0.91, 1.0};              
    /* inverse map of cumulative distribution (for O(1) lookup) */            
    /* std::vector<int> cum_map = {0, 0, 0, 0, 0, 1, 2, 2, 3, 4}; */          
    int k = 10; /* 100; */                                                    
    std::vector<int> cum_map(k, 0);                                           
    for (int j = 0; j < cum_dis.size();) {                                    
      int sz = (int)round(cum_dis[j] * k);                                    
      for (int i = 0; i < sz; i++) {                                          
        cum_map[j + i] = j;                                                   
      }                                                                       
      j += sz;                                                                
    }                                                                         
                                                                              
    /* code to generate the synthetic data from the above values */           
    const int cache_line = 1; /* 64; */                                       
    generate_trace_lru<T, double, int, CPUContext, false>(                    
        mem_ref, cum_val, cum_dis, cum_map, context, cache_line, n, a, b, r); 
  }

  */
    }
}

caffe2_rand_synthetic_data!{f32}
caffe2_rand_synthetic_data!{f64}
caffe2_rand_synthetic_data!{i8}
caffe2_rand_synthetic_data!{i16}
caffe2_rand_synthetic_data!{i32}
caffe2_rand_synthetic_data!{i64}
caffe2_rand_synthetic_data!{u8}
caffe2_rand_synthetic_data!{u16}
caffe2_rand_synthetic_data!{u32}
caffe2_rand_synthetic_data!{u64}

#[macro_export] macro_rules! caffe2_specialized_rand_uniform_unique {
    ($T:ident) => {
        /*
  template <>                                                        
  void RandUniformUnique<T, CPUContext>(                  
      const size_t n,                                                
      const T a,                                                     
      const T b,                                                     
      T* r,                                                          
      const size_t m,                                                
      const T* avoid,                                                
      CPUContext* context) {                                         
    CAFFE_ENFORCE_LE(                                                
        n, b - a - m + 1, "Cannot satisfy the unique requirement");  
    std::unordered_set<T> avoid_set(n);                              
    if (m) {                                                         
      avoid_set.insert(avoid, avoid + m);                            
      CAFFE_ENFORCE_EQ(                                              
          m, avoid_set.size(), "Avoid should be unique"); 
    }                                                                
    at::uniform_int_from_to_distribution<T> distribution(            
        incrementIfNotMax(b - a), a);                                
    T v = 0;                                                         
    for (size_t i = 0; i < n; ++i) {                                 
      do {                                                           
        v = distribution(context->RandGenerator());                  
      } while (avoid_set.count(v));                                  
      r[i] = v;                                                      
      avoid_set.insert(v);                                           
    }                                                                
  }

  */
    }
}

caffe2_specialized_rand_uniform_unique!{i32}
caffe2_specialized_rand_uniform_unique!{i64}

#[inline] pub fn rand_gaussian_f32cpu_context(
    n:       usize,
    mean:    f32,
    std:     f32,
    r:       *mut f32,
    context: *mut CPUContext)  {
    
    todo!();
    /*
        at::normal_distribution<float> distribution(mean, std);
      for (size_t i = 0; i < n; ++i) {
        r[i] = distribution(context->RandGenerator());
      }
    */
}

#[macro_export] macro_rules! caffe2_specialized_sum {
    ($T:ident) => {
        /*
  template <>                                
  void Sum<T, CPUContext>(        
      const int N,                           
      const T* x,                            
      T* y,                                  
      CPUContext* /* unused */,              
      Tensor* /* unused */) {                
    *y = ConstEigenVectorMap<T>(x, N).sum(); 
  }

  */
    }
}

caffe2_specialized_sum!{f32}
caffe2_specialized_sum!{i32}
caffe2_specialized_sum!{i64}

#[inline] pub fn sum_sqr_f32cpu_context(
    n:           i32,
    x:           *const f32,
    y:           *mut f32,
    context:     *mut CPUContext,
    scratch_ptr: *mut Tensor)  {
    
    todo!();
    /*
        *y = ConstEigenVectorMap<float>(x, N).squaredNorm();
    */
}


#[inline] pub fn sum_sqr_f64cpu_context(
    n:           i32,
    x:           *const f64,
    y:           *mut f64,
    context:     *mut CPUContext,
    scratch_ptr: *mut Tensor)  {
    
    todo!();
    /*
        *y = ConstEigenVectorMap<double>(x, N).squaredNorm();
    */
}

#[inline] pub fn select_f32cpu_context(
    n:       i32,
    d:       i32,
    x:       *const f32,
    idx:     *const i32,
    y:       *mut f32,
    context: *mut CPUContext)  {
    
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        DCHECK_LT(idx[i], D);
        y[i] = x[i * D + idx[i]];
      }
    */
}


#[inline] pub fn copy_matrix_cpu_context(
    itemsize: usize,
    m:        i32,
    n:        i32,
    a:        *const c_void,
    lda:      i32,
    b:        *mut c_void,
    ldb:      i32,
    context:  *mut CPUContext,
    copy:     TypeMetaCopy)  {
    
    todo!();
    /*
        if (A == nullptr || B == nullptr) {
        return;
      }
      if (lda == N && ldb == N) {
        // can coalesce to a single memcpy of size M * N
        if (copy) {
          copy(static_cast<const char*>(A), static_cast<char*>(B), N * M);
        } else {
          memcpy(
              static_cast<char*>(B), static_cast<const char*>(A), itemsize * N * M);
        }
        return;
      }

      for (int i = 0; i < M; ++i) {
        if (copy) {
          copy(
              static_cast<const char*>(A) + lda * i * itemsize,
              static_cast<char*>(B) + ldb * i * itemsize,
              N);
        } else {
          memcpy(
              static_cast<char*>(B) + ldb * i * itemsize,
              static_cast<const char*>(A) + lda * i * itemsize,
              itemsize * N);
        }
      }
    */
}


#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! delegate_copy_matrix_function {
    ($T:ident, $Func:ident) => {
        /*
  template <>                                   
  void CopyMatrix<T, CPUContext>(    
      const int M,                              
      const int N,                              
      const T* A,                               
      const int lda,                            
      T* B,                                     
      const int ldb,                            
      CPUContext* /* context */) {              
    Func('R', 'N', M, N, T(1), A, lda, B, ldb); 
  }                                             
  template <>                                   
  void CopyMatrix<T, CPUContext>(    
      const int M,                              
      const int N,                              
      const T* A,                               
      const int A_outer_stride,                 
      const int A_inner_stride,                 
      T* B,                                     
      const int B_outer_stride,                 
      const int B_inner_stride,                 
      CPUContext* /* context */) {              
    Func##2(                                    
        'R',                                    
        'N',                                    
        M,                                      
        N,                                      
        T(1),                                   
        A,                                      
        A_outer_stride,                         
        A_inner_stride,                         
        B,                                      
        B_outer_stride,                         
        B_inner_stride);                        
  }
  */
    }
}

#[cfg(caffe2_use_mkl)] delegate_copy_matrix_function!{f32, mkl_somatcopy}
#[cfg(caffe2_use_mkl)] delegate_copy_matrix_function!{f64, mkl_domatcopy}

#[cfg(caffe2_use_mkl)]
#[macro_export] macro_rules! caffe2_specialized_copy_matrix {
    ($T:ty) => {
        /*
  template <>                                                            
  void CopyMatrix<T, CPUContext>(                             
      const int M,                                                       
      const int N,                                                       
      const T* A,                                                        
      const int lda,                                                     
      T* B,                                                              
      const int ldb,                                                     
      CPUContext* /* context */) {                                       
    if (M == 0 || N == 0) {                                              
      return;                                                            
    }                                                                    
    if (lda == N) {                                                      
      if (ldb == N) {                                                    
        std::memcpy(B, A, sizeof(T) * M * N);                            
      } else {                                                           
        EigenOuterStridedMatrixMap<T>(B, N, M, EigenOuterStride(ldb)) =  
            ConstEigenMatrixMap<T>(A, N, M);                             
      }                                                                  
    } else {                                                             
      if (ldb == N) {                                                    
        EigenMatrixMap<T>(B, N, M) = ConstEigenOuterStridedMatrixMap<T>( 
            A, N, M, EigenOuterStride(lda));                             
      } else {                                                           
        EigenOuterStridedMatrixMap<T>(B, N, M, EigenOuterStride(ldb)) =  
            ConstEigenOuterStridedMatrixMap<T>(                          
                A, N, M, EigenOuterStride(lda));                         
      }                                                                  
    }                                                                    
  }                                                                      
  template <>                                                            
  void CopyMatrix<T, CPUContext>(                             
      const int M,                                                       
      const int N,                                                       
      const T* A,                                                        
      const int A_outer_stride,                                          
      const int A_inner_stride,                                          
      T* B,                                                              
      const int B_outer_stride,                                          
      const int B_inner_stride,                                          
      CPUContext* context) {                                             
    if (A_inner_stride == 1 && B_inner_stride == 1) {                    
      CopyMatrix<T, CPUContext>(                                         
          M, N, A, A_outer_stride, B, B_outer_stride, context);          
      return;                                                            
    }                                                                    
    EigenStridedMatrixMap<T>(                                            
        B, N, M, EigenStride(B_outer_stride, B_inner_stride)) =          
        ConstEigenStridedMatrixMap<T>(                                   
            A, N, M, EigenStride(A_outer_stride, A_inner_stride));       
  }

  */
    }
}

#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{f32}
#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{f64}
#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{i32}
#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{i64}
#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{u8}
#[cfg(caffe2_use_mkl)] caffe2_specialized_copy_matrix!{u16}

#[inline] pub fn im_2col_zero_padding_and_no_dilationNCHW<T>(
    c:        i32,
    h:        i32,
    w:        i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    img_data: *const T,
    col_data: *mut T,
    context:  *mut CPUContext)  {
    todo!();
    /*
        const int output_h = (H - kernel_h) / stride_h + 1;
      const int output_w = (W - kernel_w) / stride_w + 1;
      const int output_size = output_h * output_w;
      for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
            const T* src = img_data + kh * W + kw;
            if (stride_w == 1) {
              CopyMatrix<T, CPUContext>(
                  output_h,
                  output_w,
                  src,
                  stride_h * W,
                  col_data,
                  output_w,
                  context);
            } else {
              CopyMatrix<T, CPUContext>(
                  output_h,
                  output_w,
                  src,
                  stride_h * W,
                  stride_w,
                  col_data,
                  output_w,
                  1,
                  context);
            }
            col_data += output_size;
          }
        }
        img_data += H * W;
      }
    */
}


#[inline] pub fn col_2im_zero_padding_and_no_dilationNCHW<T>(
    c:        i32,
    h:        i32,
    w:        i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    col_data: *const T,
    img_data: *mut T,
    context:  *mut CPUContext)  {
    todo!();
    /*
        Set<T, CPUContext>(C * H * W, T(0), img_data, context);
      const int output_h = (H - kernel_h) / stride_h + 1;
      const int output_w = (W - kernel_w) / stride_w + 1;
      const int output_size = output_h * output_w;
      for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
            T* dst = img_data + kh * W + kw;
            if (stride_w == 1) {
              EigenOuterStridedArrayMap<T>(
                  dst, output_w, output_h, EigenOuterStride(stride_h * W)) +=
                  ConstEigenArrayMap<T>(col_data, output_w, output_h);
            } else {
              EigenStridedArrayMap<T>(
                  dst, output_w, output_h, EigenStride(stride_h * W, stride_w)) +=
                  ConstEigenArrayMap<T>(col_data, output_w, output_h);
            }
            col_data += output_size;
          }
        }
        img_data += H * W;
      }
    */
}

#[inline] pub fn im_2col_zero_padding_and_no_dilationNHWC<T>(
    c:        i32,
    h:        i32,
    w:        i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    img_data: *const T,
    col_data: *mut T,
    context:  *mut CPUContext)  {
    todo!();
    /*
        const int output_h = (H - kernel_h) / stride_h + 1;
      const int output_w = (W - kernel_w) / stride_w + 1;
      const int kernel_size = kernel_h * kernel_w;
      for (int yh = 0; yh < output_h; ++yh) {
        for (int yw = 0; yw < output_w; ++yw) {
          const T* src = img_data + (yh * stride_h * W + yw * stride_w) * C;
          CopyMatrix<T, CPUContext>(
              kernel_h, kernel_w * C, src, W * C, col_data, kernel_w * C, context);
          col_data += kernel_size * C;
        }
      }
    */
}


#[inline] pub fn col_2im_zero_padding_and_no_dilationNHWC<T>(
    c:        i32,
    h:        i32,
    w:        i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    col_data: *const T,
    img_data: *mut T,
    context:  *mut CPUContext)  {
    todo!();
    /*
        Set<T, CPUContext>(H * W * C, T(0), img_data, context);
      const int output_h = (H - kernel_h) / stride_h + 1;
      const int output_w = (W - kernel_w) / stride_w + 1;
      const int kernel_size = kernel_h * kernel_w;
      for (int yh = 0; yh < output_h; ++yh) {
        for (int yw = 0; yw < output_w; ++yw) {
          T* dst = img_data + (yh * stride_h * W + yw * stride_w) * C;
          EigenOuterStridedArrayMap<T>(
              dst, kernel_w * C, kernel_h, EigenOuterStride(W * C)) +=
              ConstEigenArrayMap<T>(col_data, kernel_w * C, kernel_h);
          col_data += kernel_size * C;
        }
      }
    */
}

#[inline] pub fn im_2col_nd_nchwimpl<T, const kCol2Im: bool>(
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    x_data:       *const f32,
    y_data:       *mut f32)  {
    todo!();
    /*
        if (kCol2Im) {
        std::memset(Y_data, 0, img_size * sizeof(float));
      }
      const int outer_size = col_shape[0];
      const int inner_size = col_size / outer_size;
      const int kernel_size = std::accumulate(
          kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
      std::vector<FixedDivisor<int>> kernel_shape_div(N);
      for (int i = 0; i < N; ++i) {
        kernel_shape_div[i] = FixedDivisor<int>(kernel_shape[i]);
      }
      std::vector<int> d_offset(N, 0);
      std::vector<int> d_iter(N, 0);
      for (int i = 0; i < outer_size; ++i) {
        // Loop over spatial axes in reverse order to compute a per-axis offset.
        int offset = i;
        for (int d_i = N - 1; d_i >= 0; --d_i) {
          kernel_shape_div[d_i].DivMod(offset, &offset, &d_offset[d_i]);
        }
        for (int j = 0; j < inner_size; ++j) {
          // Loop over spatial axes in forward order to compute the indices in the
          // image and column, and whether the index lies in the padding.
          const int col_index = i * inner_size + j;
          int img_index = i / kernel_size;
          bool is_padding = false;
          for (int d_i = 0; d_i < N; ++d_i) {
            const int d_img = d_iter[d_i] * stride[d_i] - pad[d_i] +
                d_offset[d_i] * dilation[d_i];
            is_padding |= !utils::IsAGeZeroAndALtB(d_img, img_shape[d_i + 1]);
            img_index = img_index * img_shape[d_i + 1] + d_img;
          }
          if (!kCol2Im) {
            Y_data[col_index] = is_padding ? 0 : X_data[img_index];
          } else if (!is_padding) {
            Y_data[img_index] += X_data[col_index];
          }
          utils::IncreaseIndexInDims(N, col_shape + 1, d_iter.data());
        }
      }
    */
}


#[inline] pub fn im_2col3d_nchwimpl<T>(
    channels:   i32,
    clip_len:   i32,
    height:     i32,
    width:      i32,
    kernel_t:   i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_t: i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_p:      i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_a:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_t:   i32,
    stride_h:   i32,
    stride_w:   i32,
    img_data:   *const T,
    col_data:   *mut T)  {
    todo!();
    /*
        const int output_t =
          (clip_len + pad_p + pad_a - (dilation_t * (kernel_t - 1) + 1)) /
              stride_t +
          1;
      const int output_h =
          (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
          1;
      const int output_w =
          (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
          1;
      const int kernel_size = kernel_t * kernel_h * kernel_w;
      const int kernel_hw_size = kernel_h * kernel_w;
      const int output_size = output_t * output_h * output_w;
      const int channel_size = clip_len * height * width;
      const int output_hw_size = output_h * output_w;
      const int channel_hw_size = height * width;

      // Fast path for zero padding and no dilation
      // From Torch, THNN_(unfolded_copy)
      if (dilation_t == 1 && dilation_h == 1 && dilation_w == 1 && pad_a == 0 &&
          pad_p == 0 && pad_l == 0 && pad_r == 0 && pad_t == 0 && pad_b == 0) {
        for (auto k = 0; k < channels * kernel_size; k++) {
          const auto nip = k / kernel_size;
          const auto rest = k % kernel_size;
          const auto kt = rest / kernel_hw_size;
          const auto rest_hw = rest % kernel_hw_size;
          const auto kh = rest_hw / kernel_w;
          const auto kw = rest_hw % kernel_w;
          auto* dst = col_data + nip * (kernel_size * output_size) +
              kt * (kernel_hw_size * output_size) + kh * (kernel_w * output_size) +
              kw * output_size;
          const auto* src = img_data + nip * channel_size;
          for (auto t = 0; t < output_t; t++) {
            const auto it = t * stride_t + kt;
            for (auto y = 0; y < output_h; y++) {
              const auto iy = y * stride_h + kh;
              const auto ix = kw;
              if (stride_w == 1) {
                memcpy(
                    dst + (t * output_hw_size + y * output_w),
                    src + (it * channel_hw_size + iy * width + ix),
                    sizeof(T) * output_w);
              } else {
                for (auto x = 0; x < output_w; x++) {
                  memcpy(
                      dst + (t * output_hw_size + y * output_w + x),
                      src + (it * channel_hw_size + iy * width + ix + x * stride_w),
                      sizeof(T));
                }
              }
            }
          }
        }
        return;
      }
      // Fast path for equal padding
      if (pad_a == pad_p && pad_l == pad_r && pad_t == pad_b) {
        const int pad_f = pad_a;
        const int pad_h = pad_t;
        const int pad_w = pad_l;
        for (int channel = channels; channel--; img_data += channel_size) {
          for (int kernel_frame = 0; kernel_frame < kernel_t; kernel_frame++) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
              for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_frame = -pad_f + kernel_frame * dilation_t;
                for (int output_frames = output_t; output_frames; output_frames--) {
                  if (!utils::IsAGeZeroAndALtB(input_frame, clip_len)) {
                    for (int output_rows = output_h; output_rows; output_rows--) {
                      for (int output_cols = output_w; output_cols; output_cols--) {
                        *(col_data++) = 0;
                      }
                    }
                  } else {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                      if (!utils::IsAGeZeroAndALtB(input_row, height)) {
                        for (int output_cols = output_w; output_cols;
                             output_cols--) {
                          *(col_data++) = 0;
                        }
                      } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                          if (utils::IsAGeZeroAndALtB(input_col, width)) {
                            *(col_data++) = img_data
                                [(input_frame * height + input_row) * width +
                                 input_col];
                          } else {
                            *(col_data++) = 0;
                          }
                          input_col += stride_w;
                        }
                      }
                      input_row += stride_h;
                    }
                  }
                  input_frame += stride_t;
                }
              }
            }
          }
        }
        return;
      }

      // Baseline
      const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
      const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
      const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

      int clip_col = (clip_len + pad_p + pad_a - dkernel_t) / stride_t + 1;
      int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
      int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

      int channels_col = channels * kernel_t * kernel_h * kernel_w;
      for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int t_offset = (c / kernel_w / kernel_h) % kernel_t;
        int c_im = c / kernel_h / kernel_w / kernel_t;
        for (int t = 0; t < clip_col; ++t) {
          for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
              int t_pad = t * stride_t - pad_p + t_offset * dilation_t;
              int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
              int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
              if (t_pad >= 0 && t_pad < clip_len && h_pad >= 0 && h_pad < height &&
                  w_pad >= 0 && w_pad < width) {
                col_data[((c * clip_col + t) * height_col + h) * width_col + w] =
                    img_data
                        [((c_im * clip_len + t_pad) * height + h_pad) * width +
                         w_pad];
              } else {
                col_data[((c * clip_col + t) * height_col + h) * width_col + w] = 0;
              }
            }
          }
        }
      }
    */
}

#[inline] pub fn im_2col_nd_f32cpu_contextNCHW(
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    img_data:     *const f32,
    col_data:     *mut f32,
    context:      *mut CPUContext,
    groups:       i32)  {
    
    todo!();
    /*
        // In NCHW, the number of groups doesn't affect Im2Col.
      if (N == 3) {
        const int channels =
            col_shape[0] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
        Im2Col3dNCHWImpl<float>(
            channels,
            img_shape[1],
            img_shape[2],
            img_shape[3],
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            dilation[0],
            dilation[1],
            dilation[2],
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
            stride[0],
            stride[1],
            stride[2],
            img_data,
            col_data);
      } else {
        Im2ColNdNCHWImpl<float, false>(
            N,
            img_size,
            col_size,
            img_shape,
            col_shape,
            kernel_shape,
            stride,
            dilation,
            pad,
            img_data,
            col_data);
      }
    */
}


#[inline] pub fn col_2im_nd_f32cpu_contextNCHW(
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    col_data:     *const f32,
    img_data:     *mut f32,
    context:      *mut CPUContext,
    groups:       i32)  {
    
    todo!();
    /*
        // In NCHW, the number of groups doesn't affect Col2Im.
      Im2ColNdNCHWImpl<float, true>(
          N,
          img_size,
          col_size,
          img_shape,
          col_shape,
          kernel_shape,
          stride,
          dilation,
          pad,
          col_data,
          img_data);
    */
}

#[inline] pub fn im_2col_f32cpu_contextNCHW(
    c:          i32,
    h:          i32,
    w:          i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    img_data:   *const f32,
    col_data:   *mut f32,
    context:    *mut CPUContext,
    groups:     i32)  {
    
    todo!();
    /*
        // In NCHW, the number of groups doesn't affect Im2Col.

      // Fast path for zero padding and no dilation
      if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
          dilation_w == 1) {
        Im2ColZeroPaddingAndNoDilationNCHW<float>(
            C,
            H,
            W,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            img_data,
            col_data,
            context);
        return;
      }

      // Baseline
      const int output_h =
          (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
      const int output_w =
          (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
      const int output_size = output_h * output_w;
      for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
            for (int h = 0; h < output_h; ++h) {
              const int h_pad = h * stride_h - pad_t + kh * dilation_h;
              if (!utils::IsAGeZeroAndALtB(h_pad, H)) {
                std::memset(col_data + h * output_w, 0, output_w * sizeof(float));
                continue;
              }
              for (int w = 0; w < output_w; ++w) {
                const int w_pad = w * stride_w - pad_l + kw * dilation_w;
                col_data[h * output_w + w] = utils::IsAGeZeroAndALtB(w_pad, W)
                    ? img_data[(c * H + h_pad) * W + w_pad]
                    : 0;
              }
            }
            col_data += output_size;
          }
        }
      }
    */
}

#[inline] pub fn im_2col_f32cpu_contextNHWC(
    c:          i32,
    h:          i32,
    w:          i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    img_data:   *const f32,
    col_data:   *mut f32,
    context:    *mut CPUContext,
    groups:     i32)  {
    
    todo!();
    /*
        // Fast path for zero padding and no dilation
      if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
          dilation_w == 1 && groups == 1) {
        Im2ColZeroPaddingAndNoDilationNHWC<float>(
            C,
            H,
            W,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            img_data,
            col_data,
            context);
        return;
      }

      const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
      const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
      const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
      const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
      int h_pad = -pad_t;
      if (groups == 1) {
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
              if (!utils::IsAGeZeroAndALtB(ih, H)) {
                std::memset(col_data, 0, sizeof(float) * kernel_w * C);
                col_data += kernel_w * C;
                continue;
              }
              for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
                if (utils::IsAGeZeroAndALtB(iw, W)) {
                  std::memcpy(
                      col_data, img_data + (ih * W + iw) * C, sizeof(float) * C);
                } else {
                  std::memset(col_data, 0, sizeof(float) * C);
                }
                col_data += C;
              } // iw
            } // ih
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
      } else {
        /**
         * img_data in N H W G C/G layout
         * col_data in N G H W R S C/G layout
         * Note that groups are pulled out to an outer dimension so that we can use
         * GEMMs efficiently.
         */
        const int C_per_G = C / groups;
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            int r = 0;
            for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
              int s = 0;
              for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
                if (utils::IsAGeZeroAndALtB(ih, H) &&
                    utils::IsAGeZeroAndALtB(iw, W)) {
                  for (int g = 0; g < groups; ++g) {
                    std::memcpy(
                        col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                        img_data + (ih * W + iw) * C + g * C_per_G,
                        sizeof(float) * C_per_G);
                  }
                } else {
                  for (int g = 0; g < groups; ++g) {
                    std::memset(
                        col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                        0,
                        sizeof(float) * C_per_G);
                  }
                }
              } // iw
            } // ih
            col_data += kernel_h * kernel_w * C;
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
      }
    */
}

/**
 | The layout of the result is N H W G R S C/G.
 |
 | Note that groups are pulled out to an outer
 | dimension so that we can use GEMMs efficiently.
 |
 | pad_p previous frame
 | pad_t top
 | pad_l left
 | pad_n next frame
 | pad_b bottom
 | pad_r right
 */
#[inline] pub fn im_2col3d_nhwcimpl<TData>(
    c:          i32,
    t:          i32,
    h:          i32,
    w:          i32,
    kernel_t:   i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_t: i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_p:      i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_n:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_t:   i32,
    stride_h:   i32,
    stride_w:   i32,
    img_data:   *const TData,
    col_data:   *mut TData,
    groups:     i32)  {
    todo!();
    /*
        const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
      const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
      const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
      const int output_t = (T + pad_p + pad_n - dkernel_t) / stride_t + 1;
      const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
      const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
      const int C_per_G = C / groups;
      int t_pad = -pad_p;
      for (int t = 0; t < output_t; ++t) {
        int h_pad = -pad_t;
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            int q = 0;
            for (int it = t_pad; it < t_pad + dkernel_t; it += dilation_t, ++q) {
              int r = 0;
              for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
                int s = 0;
                for (int iw = w_pad; iw < w_pad + dkernel_w;
                     iw += dilation_w, ++s) {
                  if (utils::IsAGeZeroAndALtB(it, T) &&
                      utils::IsAGeZeroAndALtB(ih, H) &&
                      utils::IsAGeZeroAndALtB(iw, W)) {
                    for (int g = 0; g < groups; ++g) {
                      std::memcpy(
                          col_data +
                              (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                                  C_per_G,
                          img_data + ((it * H + ih) * W + iw) * C + g * C_per_G,
                          sizeof(TData) * C_per_G);
                    }
                  } else {
                    for (int g = 0; g < groups; ++g) {
                      std::memset(
                          col_data +
                              (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                                  C_per_G,
                          0,
                          sizeof(TData) * C_per_G);
                    }
                  }
                } // iw
              } // ih
            } // it
            col_data += kernel_t * kernel_h * kernel_w * C;
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
        t_pad += stride_t;
      } // t
    */
}

#[inline] pub fn im_2col_nd_f32cpu_contextNHWC(
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    img_data:     *const f32,
    col_data:     *mut f32,
    context:      *mut CPUContext,
    groups:       i32)  {
    
    todo!();
    /*
        if (N == 3) {
        const int channels =
            col_shape[3] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
        Im2Col3dNHWCImpl<float>(
            channels,
            img_shape[0],
            img_shape[1],
            img_shape[2],
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            dilation[0],
            dilation[1],
            dilation[2],
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
            stride[0],
            stride[1],
            stride[2],
            img_data,
            col_data,
            groups);
      } else {
        CAFFE_NOT_IMPLEMENTED;
      }
    */
}

#[inline] pub fn col_2im_f32cpu_contextNCHW(
    c:          i32,
    h:          i32,
    w:          i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    col_data:   *const f32,
    img_data:   *mut f32,
    context:    *mut CPUContext,
    groups:     i32)  {
    
    todo!();
    /*
        // In NCHW, the number of groups doesn't affect Col2Im.

      // Fast path for zero padding and no dilation
      if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
          dilation_w == 1) {
        Col2ImZeroPaddingAndNoDilationNCHW<float>(
            C,
            H,
            W,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            col_data,
            img_data,
            context);
        return;
      }

      // Fallback
      Set<float, CPUContext>(C * H * W, 0.0f, img_data, context);
      const int output_h =
          (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
      const int output_w =
          (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
      const int output_size = output_h * output_w;
      for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
            for (int h = 0; h < output_h; ++h) {
              const int h_pad = h * stride_h - pad_t + kh * dilation_h;
              if (!utils::IsAGeZeroAndALtB(h_pad, H)) {
                continue;
              }
              for (int w = 0; w < output_w; ++w) {
                const int w_pad = w * stride_w - pad_l + kw * dilation_w;
                if (utils::IsAGeZeroAndALtB(w_pad, W)) {
                  img_data[(c * H + h_pad) * W + w_pad] +=
                      col_data[h * output_w + w];
                }
              }
            }
            col_data += output_size;
          }
        }
      }
    */
}

#[inline] pub fn col_2im_f32cpu_contextNHWC(
    c:          i32,
    h:          i32,
    w:          i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_h:   i32,
    stride_w:   i32,
    col_data:   *const f32,
    img_data:   *mut f32,
    context:    *mut CPUContext,
    groups:     i32)  {
    
    todo!();
    /*
        // Fast path for zero padding and no dilation
      if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
          dilation_w == 1 && groups == 1) {
        Col2ImZeroPaddingAndNoDilationNHWC<float>(
            C,
            H,
            W,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            col_data,
            img_data,
            context);
        return;
      }

      Set<float, CPUContext>(H * W * C, 0, img_data, context);
      const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
      const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
      const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
      const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;

      int h_pad = -pad_t;
      if (groups == 1) {
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
              if (!utils::IsAGeZeroAndALtB(ih, H)) {
                col_data += kernel_w * C;
                continue;
              }
              for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
                if (utils::IsAGeZeroAndALtB(iw, W)) {
                  float* img_data_patch = img_data + (ih * W + iw) * C;
                  Add<float, CPUContext>(
                      C, img_data_patch, col_data, img_data_patch, context);
                }
                col_data += C;
              } // iw
            } // ih
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
      } else {
        const int C_per_G = C / groups;
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            int r = 0;
            for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
              int s = 0;
              for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
                if (utils::IsAGeZeroAndALtB(ih, H) &&
                    utils::IsAGeZeroAndALtB(iw, W)) {
                  float* img_data_patch = img_data + (ih * W + iw) * C;
                  for (int g = 0; g < groups; ++g) {
                    Add<float, CPUContext>(
                        C_per_G,
                        img_data_patch + g * C_per_G,
                        col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                        img_data_patch + g * C_per_G,
                        context);
                  }
                }
              } // iw
            } // ih
            col_data += kernel_h * kernel_w * C;
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
      }
    */
}


/**
 | The layout of the result is N H W G R S C/G.
 |
 | Note that groups are pulled out to an outer
 | dimension so that we can use GEMMs efficiently.
 |
 |  pad_p - previous frame
 |  pad_t - top
 |  pad_l - left
 |  pad_n - next frame
 |  pad_b - bottom
 |  pad_r - right
 */
#[inline] pub fn col_2im3d_nhwcimpl<TData>(
    c:          i32,
    t:          i32,
    h:          i32,
    w:          i32,
    kernel_t:   i32,
    kernel_h:   i32,
    kernel_w:   i32,
    dilation_t: i32,
    dilation_h: i32,
    dilation_w: i32,
    pad_p:      i32,
    pad_t:      i32,
    pad_l:      i32,
    pad_n:      i32,
    pad_b:      i32,
    pad_r:      i32,
    stride_t:   i32,
    stride_h:   i32,
    stride_w:   i32,
    col_data:   *const TData,
    img_data:   *mut TData,
    context:    *mut CPUContext,
    groups:     i32)  {
    todo!();
    /*
        Set<float, CPUContext>(T * H * W * C, 0, img_data, context);
      const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
      const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
      const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
      const int output_t = (T + pad_p + pad_n - dkernel_t) / stride_t + 1;
      const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
      const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
      const int C_per_G = C / groups;

      int t_pad = -pad_p;
      for (int t = 0; t < output_t; ++t) {
        int h_pad = -pad_t;
        for (int h = 0; h < output_h; ++h) {
          int w_pad = -pad_l;
          for (int w = 0; w < output_w; ++w) {
            int q = 0;
            for (int it = t_pad; it < t_pad + dkernel_t; it += dilation_t, ++q) {
              int r = 0;
              for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
                int s = 0;
                for (int iw = w_pad; iw < w_pad + dkernel_w;
                     iw += dilation_w, ++s) {
                  if (utils::IsAGeZeroAndALtB(it, T) &&
                      utils::IsAGeZeroAndALtB(ih, H) &&
                      utils::IsAGeZeroAndALtB(iw, W)) {
                    float* img_data_patch = img_data + ((it * T + ih) * W + iw) * C;
                    for (int g = 0; g < groups; ++g) {
                      Add<float, CPUContext>(
                          C_per_G,
                          img_data_patch + g * C_per_G,
                          col_data +
                              (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                                  C_per_G,
                          img_data_patch + g * C_per_G,
                          context);
                    }
                  }
                } // iw
              } // ih
            } // it
            col_data += kernel_t * kernel_h * kernel_w * C;
            w_pad += stride_w;
          } // w
          h_pad += stride_h;
        } // h
        t_pad += stride_t;
      } // t
    */
}

#[inline] pub fn col_2im_nd_f32cpu_contextNHWC(
    n:            i32,
    img_size:     i32,
    col_size:     i32,
    img_shape:    *const i32,
    col_shape:    *const i32,
    kernel_shape: *const i32,
    stride:       *const i32,
    dilation:     *const i32,
    pad:          *const i32,
    col_data:     *const f32,
    img_data:     *mut f32,
    context:      *mut CPUContext,
    groups:       i32)  {
    
    todo!();
    /*
        if (N == 3) {
        const int channels =
            col_shape[3] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
        Col2Im3dNHWCImpl<float>(
            channels,
            img_shape[0],
            img_shape[1],
            img_shape[2],
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            dilation[0],
            dilation[1],
            dilation[2],
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
            stride[0],
            stride[1],
            stride[2],
            col_data,
            img_data,
            context,
            groups);
      } else {
        CAFFE_NOT_IMPLEMENTED;
      }
    */
}

#[inline] pub fn bias_chwf32cpu_context(
    bias:            *const f32,
    bias_multiplier: *const f32,
    bias_channels:   i32,
    image_size:      i32,
    image:           *mut f32,
    context:         *mut CPUContext)  {
    
    todo!();
    /*
        // Sum the per-channel bias into every image plane
      for (int c = 0; c < bias_channels; ++c) {
        float b = bias[c];

    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        float32x4_t vBias = vdupq_n_f32(b);

        // We give alignment hints for additional speed, so handle the
        // non-vectorizable prologue separately
        constexpr int kVecSizeInFloat = sizeof(float32x4_t) / sizeof(float);

        // FIXME: if input < kVecSizeInFloat, can't vectorize at all

        int prologue = kVecSizeInFloat -
            // remainder in floats
            (((uintptr_t)image) % (sizeof(float32x4_t))) / sizeof(float);

        int i = 0;
        // Prologue loop
        for (; i < prologue; ++i) {
          image[i] += b;
        }

        // The loop is manually unrolled by 8
        constexpr int kUnroll = 8;
        constexpr int kFloatsPerLoop = kUnroll * kVecSizeInFloat;

        int remainder = image_size - prologue;
        int vectorizable = prologue + (remainder / kFloatsPerLoop) * kFloatsPerLoop;

        // Vectorizable body
        for (; i < vectorizable; i += kFloatsPerLoop) {
          // Manually unrolled
          float32x4_t v0 = vld1q_f32_aligned(image + i + 0);
          float32x4_t v1 = vld1q_f32_aligned(image + i + 4);
          float32x4_t v2 = vld1q_f32_aligned(image + i + 8);
          float32x4_t v3 = vld1q_f32_aligned(image + i + 12);
          float32x4_t v4 = vld1q_f32_aligned(image + i + 16);
          float32x4_t v5 = vld1q_f32_aligned(image + i + 20);
          float32x4_t v6 = vld1q_f32_aligned(image + i + 24);
          float32x4_t v7 = vld1q_f32_aligned(image + i + 28);

          v0 = vaddq_f32(v0, vBias);
          v1 = vaddq_f32(v1, vBias);
          v2 = vaddq_f32(v2, vBias);
          v3 = vaddq_f32(v3, vBias);
          v4 = vaddq_f32(v4, vBias);
          v5 = vaddq_f32(v5, vBias);
          v6 = vaddq_f32(v6, vBias);
          v7 = vaddq_f32(v7, vBias);

          vst1q_f32_aligned(image + i + 0, v0);
          vst1q_f32_aligned(image + i + 4, v1);
          vst1q_f32_aligned(image + i + 8, v2);
          vst1q_f32_aligned(image + i + 12, v3);
          vst1q_f32_aligned(image + i + 16, v4);
          vst1q_f32_aligned(image + i + 20, v5);
          vst1q_f32_aligned(image + i + 24, v6);
          vst1q_f32_aligned(image + i + 28, v7);
        }

        // Non-vectorizable epilogue
        for (; i < image_size; ++i) {
          image[i] += b;
        }
    #else
        // Non-NEON CPU implementation
        for (int i = 0; i < image_size; ++i) {
          image[i] += b;
        }
    #endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

        image += image_size;
      }
    */
}

#[macro_export] macro_rules! caffe2_specialized_copyvector {
    ($T:ident) => {
        /*
          template <>                                                       
          void CopyVector<T, CPUContext>(                        
              const int N, const T* src, T* dst, CPUContext* /*context*/) { 
            if (src != dst && N > 0) {                                      
              memcpy(dst, src, sizeof(T) * N);                              
            }                                                               
          }
          */
    }
}

caffe2_specialized_copyvector!{f32}
caffe2_specialized_copyvector!{i32}


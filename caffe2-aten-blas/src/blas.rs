crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/CPUBlas.h]

pub enum TransposeType {
    Transpose,
    NoTranspose,
    // ConjTranspose, -- Not implemented
}

pub type GemmFn = fn(
    ty:     ScalarType,
    transa: TransposeType,
    transb: TransposeType,
    m:      i64,
    n:      i64,
    k:      i64,
    alpha:  &Scalar,
    a:      *const c_void,
    lda:    i64,
    b:      *const c_void,
    ldb:    i64,
    beta:   &Scalar,
    c:      *mut c_void,
    ldc:    i64
) -> ();

declare_dispatch!{gemm_fn, gemm_stub}

pub fn gemm_scalar(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        alpha:  Scalar,
        a:      *const Scalar,
        lda:    i64,
        b:      *const Scalar,
        ldb:    i64,
        beta:   Scalar,
        c:      *mut Scalar,
        ldc:    i64)  {

    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
      gemm_stub(
        kCPU, CppTypeToScalarType<scalar_t>::value,
        transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

pub type AxpyFn = fn(
        ty:   ScalarType,
        n:    i64,
        a:    &Scalar,
        x:    *const c_void,
        incx: i64,
        y:    *mut c_void,
        incy: i64
) -> ();

declare_dispatch!{axpy_fn, axpy_stub}

pub fn axpy<scalar_t>(
        n:    i64,
        a:    Scalar,
        x:    *const Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64)  {

    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      axpy_stub(
          kCPU, CppTypeToScalarType<scalar_t>::value,
          n, a, x, incx, y, incy);
        */
}

pub type CopyFn = fn(
        ty:   ScalarType,
        n:    i64,
        x:    *const c_void,
        incx: i64,
        y:    *mut c_void,
        incy: i64
) -> ();

declare_dispatch!{copy_fn, copy_stub}

pub fn copy_<scalar_t>(
        n:    i64,
        x:    *const Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64)  {

    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      copy_stub(
          kCPU, CppTypeToScalarType<scalar_t>::value,
          n, x, incx, y, incy);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/CPUBlas.cpp]

#[cfg(AT_BUILD_WITH_BLAS)]
lazy_static!{
    /*
    extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc);
    extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);
    extern "C" void cgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
    extern "C" void zgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
    */
}

#[cfg(AT_BUILD_WITH_BLAS)]
lazy_static!{
    /*
    extern "C" void cswap_(int *n, const void *x, int *incx, void *y, int *incy);
    extern "C" void dcopy_(int *n, const double *x, int *incx, double *y, int *incy);
    extern "C" void scopy_(int *n, const float *x, int *incx, float *y, int *incy);
    extern "C" void zcopy_(int *n, const void *x, int *incx, void *y, int *incy);
    extern "C" void ccopy_(int *n, const void *x, int *incx, void *y, int *incy);
    extern "C" void daxpy_(int *n, double *a, const double *x, int *incx, double *y, int *incy);
    extern "C" void saxpy_(int *n, float *a, const float *x, int *incx, float *y, int *incy);
    extern "C" void caxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
    extern "C" void zaxpy_(int *n, void *a, const void *x, int *incx, void *y, int *incy);
    */
}

pub fn normalize_last_dims(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        lda:    *mut i64,
        ldb:    *mut i64,
        ldc:    *mut i64)  {
    
    todo!();
        /*
            if (n == 1) {
        *ldc = m;
      }

      if(transa != NoTranspose) {
        if (m == 1) {
          *lda = k;
        }
      } else if(k == 1) {
        *lda = m;
      }

      if(transb != NoTranspose) {
        if (k == 1) {
          *ldb = n;
        }
      } else if (n == 1) {
        *ldb = k;
      }
        */
}

pub fn use_blas_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        lda:    &mut i64,
        ldb:    &mut i64,
        ldc:    &mut i64) -> bool {
    
    todo!();
        /*
            const bool transa_ = transa != NoTranspose;
      const bool transb_ = transb != NoTranspose;
      return (
          (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
          (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) &&
          (lda >= max(i64{1}, (transa_ ? k : m))) &&
          (ldb >= max(i64{1}, (transb_ ? n : k))) &&
          (ldc >= max(i64{1}, m)));
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn to_blas(trans: TransposeType) -> u8 {
    
    todo!();
        /*
            switch (trans) {
      case Transpose: return 't';
      case NoTranspose: return 'n';
      // case ConjTranspose: return 'c';
      }
      TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
        */
}

#[cfg(feature = "fbgemm")]
pub fn to_fbgemm(trans: TransposeType) -> FbgemmMatrixOp {
    
    todo!();
        /*
            switch (trans) {
      case Transpose: return fbgemm::matrix_op_t::Transpose;
      case NoTranspose: return fbgemm::matrix_op_t::NoTranspose;
      // case ConjTranspose: return fbgemm::matrix_op_t::Transpose;
      }
      TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
        */
}

define_dispatch!{gemm_stub}

pub fn gemm_f64(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        alpha:  f64,
        a:      *const f64,
        lda:    i64,
        b:      *const f64,
        ldb:    i64,
        beta:   f64,
        c:      *mut f64,
        ldc:    i64)  {
    
    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    #if AT_BUILD_WITH_BLAS()
      if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
        char transa_ = to_blas(transa), transb_ = to_blas(transb);
        double alpha_ = alpha, beta_ = beta;
        dgemm_(
            &transa_, &transb_,
            &m_, &n_, &k_,
            &alpha_,
            a, &lda_,
            b, &ldb_,
            &beta_,
            c, &ldc_);
        return;
      }
    #endif
      gemm_stub(
          kCPU, kDouble,
          transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

pub fn gemm_f32(
    transa: TransposeType,
    transb: TransposeType,
    m:      i64,
    n:      i64,
    k:      i64,
    alpha:  f32,
    a:      *const f32,
    lda:    i64,
    b:      *const f32,
    ldb:    i64,
    beta:   f32,
    c:      *mut f32,
    ldc:    i64)  {

    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    #if AT_BUILD_WITH_BLAS()
      if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
        char transa_ = to_blas(transa), transb_ = to_blas(transb);
        float alpha_ = alpha, beta_ = beta;
        sgemm_(
            &transa_, &transb_,
            &m_, &n_, &k_,
            &alpha_,
            a, &lda_,
            b, &ldb_,
            &beta_,
            c, &ldc_);
        return;
      }
    #endif
      gemm_stub(
          kCPU, kFloat,
          transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

pub fn gemm_complex_f64(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        alpha:  Complex<f64>,
        a:      *const Complex<f64>,
        lda:    i64,
        b:      *const Complex<f64>,
        ldb:    i64,
        beta:   Complex<f64>,
        c:      *mut Complex<f64>,
        ldc:    i64)  {
    
    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    #if AT_BUILD_WITH_BLAS()
      if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
        char transa_ = to_blas(transa), transb_ = to_blas(transb);
        complex<double> alpha_ = alpha, beta_ = beta;
        zgemm_(
            &transa_, &transb_,
            &m_, &n_, &k_,
            &alpha_,
            a, &lda_,
            b, &ldb_,
            &beta_,
            c, &ldc_);
        return;
      }
    #endif
      gemm_stub(
          kCPU, kComplexDouble,
          transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

pub fn gemm_complex_f32(
    transa: TransposeType,
    transb: TransposeType,
    m:      i64,
    n:      i64,
    k:      i64,
    alpha:  Complex<f32>,
    a:      *const Complex<f32>,
    lda:    i64,
    b:      *const Complex<f32>,
    ldb:    i64,
    beta:   Complex<f32>,
    c:      *mut Complex<f32>,
    ldc:    i64)  {

    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    #if AT_BUILD_WITH_BLAS()
      if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
        char transa_ = to_blas(transa), transb_ = to_blas(transb);
        complex<float> alpha_ = alpha, beta_ = beta;
        cgemm_(
            &transa_, &transb_,
            &m_, &n_, &k_,
            &alpha_,
            a, &lda_,
            b, &ldb_,
            &beta_,
            c, &ldc_);
        return;
      }
    #endif
      gemm_stub(
          kCPU, kComplexFloat,
          transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

pub fn gemm_i64(
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        alpha:  i64,
        a:      *const i64,
        lda:    i64,
        b:      *const i64,
        ldb:    i64,
        beta:   i64,
        c:      *mut i64,
        ldc:    i64)  {
    
    todo!();
        /*
            internal::normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
    #ifdef USE_FBGEMM
      if (alpha == 1 && (beta == 0 || beta == 1)) {
        // In FBGEMM, we assume row-major ordering; However, here we assume the
        // column-major ordering following the FORTRAN tradition in BLAS interface
        // in this function: we can configure the layout (row/column-major ordering)
        // of A and B by changing transa_ and transb_, but we cannot change the
        // layout of C with this FORTRAN-style BLAS interface.
        //
        // The workaround is that we compute
        // C^T (n x m) = B^T (n x k) * A^T (k x m) instead.
        //
        // In this way we view C^T as the row-major ordering when passing to FBGEMM.
        fbgemm::cblas_gemm_i64_i64acc(
            to_fbgemm(transb),
            to_fbgemm(transa),
            n,
            m,
            k,
            b,
            ldb,
            a,
            lda,
            beta == 1,
            c,
            ldc);
        return;
      }
    #endif

      gemm_stub(
          kCPU, kLong,
          transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        */
}

define_dispatch!{axpy_stub}

pub fn axpy_f64(
    n:    i64,
    a:    f64,
    x:    *const f64,
    incx: i64,
    y:    *mut f64,
    incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
      {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      axpy_stub(
          kCPU, kDouble,
          n, a, x, incx, y, incy);
        */
}

pub fn axpy_f32(
    n:    i64,
    a:    f32,
    x:    *const f32,
    incx: i64,
    y:    *mut f32,
    incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
      {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      axpy_stub(
          kCPU, kFloat,
          n, a, x, incx, y, incy);
        */
}

pub fn axpy_complex_f64(
    n:    i64,
    a:    Complex<f64>,
    x:    *const Complex<f64>,
    incx: i64,
    y:    *mut Complex<f64>,
    incy: i64)  {

    todo!();
    /*
       if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
      {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        zaxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      axpy_stub(
          kCPU, kComplexDouble,
          n, a, x, incx, y, incy);
        */
}

pub fn axpy_complex_f32(
    n:    i64,
    a:    Complex<f32>,
    x:    *const Complex<f32>,
    incx: i64,
    y:    *mut Complex<f32>,
    incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
      {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        caxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      axpy_stub(
          kCPU, kComplexFloat,
          n, a, x, incx, y, incy);
        */
}

define_dispatch!{copy_stub}

pub fn copy_f64(
    n:    i64,
    x:    *const f64,
    incx: i64,
    y:    *mut f64,
    incy: i64)  {

    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        dcopy_(&i_n, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      copy_stub(
          kCPU, kDouble,
          n, x, incx, y, incy);
        */
}

pub fn copy_f32(
        n:    i64,
        x:    *const f32,
        incx: i64,
        y:    *mut f32,
        incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        scopy_(&i_n, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      copy_stub(
          kCPU, kFloat,
          n, x, incx, y, incy);
        */
}

pub fn copy_complex_f64(
        n:    i64,
        x:    *const Complex<f64>,
        incx: i64,
        y:    *mut Complex<f64>,
        incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        zcopy_(&i_n, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      copy_stub(
          kCPU, kComplexDouble,
          n, x, incx, y, incy);
        */
}

pub fn copy_complex_f32(
        n:    i64,
        x:    *const Complex<f32>,
        incx: i64,
        y:    *mut Complex<f32>,
        incy: i64)  {
    
    todo!();
        /*
            if(n == 1)
      {
        incx = 1;
        incy = 1;
      }
      #if AT_BUILD_WITH_BLAS()
      if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) ) {
        int i_n = (int)n;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        ccopy_(&i_n, x, &i_incx, y, &i_incy);
        return;
      }
      #endif
      copy_stub(
          kCPU, kComplexFloat,
          n, x, incx, y, incy);
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/BlasKernel.cpp]

lazy_static!{
    /*
    #if AT_BUILD_WITH_BLAS()
    extern "C" double ddot_(int *n, double *x, int *incx, double *y, int *incy);
    extern "C" void dscal_(int *n, double *a, double *x, int *incx);
    extern "C" void sscal_(int *n, float *a, float *x, int *incx);
    extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
    extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

    #ifdef BLAS_F2C
    # define ffloat double
    #else
    # define ffloat float
    #endif

    #ifdef BLAS_USE_CBLAS_DOT
      extern "C" float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
      extern "C" void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
      extern "C" void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
      extern "C" void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);
      extern "C" void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);

      static inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
      {
        return cblas_sdot(*n, x, *incx, y, *incy);
      }
      static inline void cdotu_(complex<float> *res, const int *n, const complex<float> *x, const int *incx,
      const complex<float> *y, const int *incy) {
        cblas_cdotu_sub(*n, x, *incx, y, *incy, res);
      }
      static inline void zdotu_(complex<double> *res, const int *n, const complex<double> *x, const int *incx,
      const complex<double> *y, const int *incy) {
        cblas_zdotu_sub(*n, x, *incx, y, *incy, res);
      }
      static inline void cdotc_(complex<float> *res, const int *n, const complex<float> *x, const int *incx,
      const complex<float> *y, const int *incy) {
        cblas_cdotc_sub(*n, x, *incx, y, *incy, res);
      }
      static inline void zdotc_(complex<double> *res, const int *n, const complex<double> *x, const int *incx,
      const complex<double> *y, const int *incy) {
        cblas_zdotc_sub(*n, x, *incx, y, *incy, res);
      }

    #else
      extern "C" ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
      extern "C" void cdotu_(complex<float> *res, int *n, complex<float> *x, int *incx, complex<float> *y, int *incy);
      extern "C" void zdotu_(complex<double> *res, int *n, complex<double> *x, int *incx, complex<double> *y, int *incy);
      extern "C" void cdotc_(complex<float> *res, int *n, complex<float> *x, int *incx, complex<float> *y, int *incy);
      extern "C" void zdotc_(complex<double> *res, int *n, complex<double> *x, int *incx, complex<double> *y, int *incy);
    #endif // BLAS_USE_CBLAS_DOT
    #endif // AT_BUILD_WITH_BLAS
    */
}

pub fn scal_use_fast_path<Scalar>(
        n:    i64,
        incx: i64) -> bool {

    todo!();
        /*
            return false;
        */
}

pub fn gemv_use_fast_path(
        m:    i64,
        n:    i64,
        lda:  i64,
        incx: i64,
        incy: i64) -> bool {
    
    todo!();
        /*
            return false;
        */
}

pub fn scal_fast_path(
        n:    *mut i32,
        a:    *mut Scalar,
        x:    *mut Scalar,
        incx: *mut i32)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
        */
}

pub fn gemv_fast_path(
        trans: *mut u8,
        m:     *mut i32,
        n:     *mut i32,
        alpha: *mut Scalar,
        a:     *mut Scalar,
        lda:   *mut i32,
        x:     *mut Scalar,
        incx:  *mut i32,
        beta:  *mut Scalar,
        y:     *mut Scalar,
        incy:  *mut i32)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
        */
}

#[macro_export] macro_rules! instantiate {
    ($Scalar:ident) => {
        /*
        
        template bool scal_use_fast_path<Scalar>(i64 n, i64 incx);                                                                                                              
        template bool gemv_use_fast_path<Scalar>(i64 m, i64 n, i64 lda, i64 incx, i64 incy);                                                                        
        template void gemv_fast_path<Scalar>(char *trans, int *m, int *n, Scalar *alpha, Scalar *a, int *lda, Scalar *x, int *incx, Scalar *beta, Scalar *y, int *incy);      
        template void scal_fast_path<Scalar>(int *n, Scalar *a, Scalar *x, int *incx);
        */
    };
    ($Scalar:ident, $_:ident) => {
        /*
        
        template void gemv<Scalar>(char trans, i64 m, i64 n, Scalar alpha, Scalar *a, i64 lda, Scalar *x, i64 incx, Scalar beta, Scalar *y, i64 incy);
        */
    }
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn scal_use_fast_path_double(
        n:    i64,
        incx: i64) -> bool {
    
    todo!();
        /*
            auto intmax = int::max;
      return n <= intmax && incx <= intmax;
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn scal_use_fast_path_float(
        n:    i64,
        incx: i64) -> bool {
    
    todo!();
        /*
            return scal_use_fast_path<double>(n, incx);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn scal_fast_path_double(
        n:    *mut i32,
        a:    *mut f64,
        x:    *mut f64,
        incx: *mut i32)  {
    
    todo!();
        /*
            dscal_(n, a, x, incx);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn scal_fast_path_float(
        n:    *mut i32,
        a:    *mut f32,
        x:    *mut f32,
        incx: *mut i32)  {
    
    todo!();
        /*
            sscal_(n, a, x, incx);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn gemv_use_fast_path_float(
        m:    i64,
        n:    i64,
        lda:  i64,
        incx: i64,
        incy: i64) -> bool {
    
    todo!();
        /*
            auto intmax = int::max;
      return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&
             (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn gemv_use_fast_path_double(
        m:    i64,
        n:    i64,
        lda:  i64,
        incx: i64,
        incy: i64) -> bool {
    
    todo!();
        /*
            return gemv_use_fast_path<float>(m, n, lda, incx, incy);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn gemv_fast_path_double(
        trans: *mut u8,
        m:     *mut i32,
        n:     *mut i32,
        alpha: *mut f64,
        a:     *mut f64,
        lda:   *mut i32,
        x:     *mut f64,
        incx:  *mut i32,
        beta:  *mut f64,
        y:     *mut f64,
        incy:  *mut i32)  {
    
    todo!();
        /*
            dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn gemv_fast_path_float(
        trans: *mut u8,
        m:     *mut i32,
        n:     *mut i32,
        alpha: *mut f32,
        a:     *mut f32,
        lda:   *mut i32,
        x:     *mut f32,
        incx:  *mut i32,
        beta:  *mut f32,
        y:     *mut f32,
        incy:  *mut i32)  {
    
    todo!();
        /*
            sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
        */
}

#[cfg(not(AT_BUILD_WITH_BLAS))]
instantiate!{f32}

#[cfg(not(AT_BUILD_WITH_BLAS))]
instantiate!{f64}

instantiate!{u8}
instantiate!{i8}
instantiate!{i16}
instantiate!{int}
instantiate!{i64}
instantiate!{BFloat16}

#[inline] pub fn scal(
        n:    i64,
        a:    Scalar,
        x:    *mut Scalar,
        incx: i64)  {
    
    todo!();
        /*
            if (n == 1) incx = 1;
      if (blas_impl::scal_use_fast_path<Scalar>(n, incx)) {
        int i_n = (int)n;
        int i_incx = (int)incx;
        blas_impl::scal_fast_path<Scalar>(&i_n, &a, x, &i_incx);
        return;
      }
      for (i64 i = 0; i < n; i++) {
        if (a == Scalar(0)) {
          x[i * incx] = 0;
        } else {
          x[i * incx] *= a;
        }
      }
        */
}

pub fn gemv(
        trans: u8,
        m:     i64,
        n:     i64,
        alpha: Scalar,
        a:     *mut Scalar,
        lda:   i64,
        x:     *mut Scalar,
        incx:  i64,
        beta:  Scalar,
        y:     *mut Scalar,
        incy:  i64)  {
    
    todo!();
        /*
            if(n == 1) lda = m;

      if (blas_impl::gemv_use_fast_path<Scalar>(m, n, lda, incx, incy)) {
        TORCH_CHECK(lda >= max<i64>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
        int i_m = (int)m;
        int i_n = (int)n;
        int i_lda = (int)lda;
        int i_incx = (int)incx;
        int i_incy = (int)incy;
        blas_impl::gemv_fast_path<Scalar>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
        return;
      }

      if ((trans == 'T') || (trans == 't')) {
        for (i64 i = 0; i < n; i++)
        {
          Scalar sum = 0;
          Scalar *row_ = a + lda * i;
          for (i64 j = 0; j < m; j++) {
            sum += x[j * incx] * row_[j];
          }
          if (beta == Scalar(0)) {
            y[i * incy] = alpha * sum;
          } else {
            y[i * incy] = beta * y[i * incy] + alpha * sum;
          }
        }
      } else {
        if (beta != Scalar(1) && beta != Scalar(0)) scal<Scalar>(m, beta, y, incy);

        for (i64 j = 0; j < n; j++) {
          Scalar *column_ = a + lda * j;
          Scalar z = alpha * x[j * incx];
          for (i64 i = 0; i < m; i++) {
            //output values are ignored if beta is 0, and set to 0, nans and infs are not propagated
            if (j==0 && beta==Scalar(0)) {
             y[i * incy] = Scalar(0);
            }
            y[i * incy] += z * column_[i];
          }
        }
      }
      return;
        */
}

lazy_static!{
    /*
    at_forall_scalar_types_and!{BFloat16, INSTANTIATE}
    at_forall_complex_types!{INSTANTIATE}
    */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn dot_fast_path(
        n:    i32,
        x:    *mut f32,
        incx: i32,
        y:    *mut f32,
        incy: i32) -> f32 {
    
    todo!();
        /*
      return sdot_(&n, x, &incx, y, &incy);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn dot_fast_path(
        n:    i32,
        x:    *mut f64,
        incx: i32,
        y:    *mut f64,
        incy: i32) -> f64 {
    
    todo!();
        /*
            return ddot_(&n, x, &incx, y, &incy);
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn vdot_fast_path(
        n:    i32,
        x:    *mut Complex<f32>,
        incx: i32,
        y:    *mut Complex<f32>,
        incy: i32) -> Complex<f32> {
    
    todo!();
        /*
            complex<float> result;
      cdotc_(reinterpret_cast<complex<float>* >(&result), &n, reinterpret_cast<complex<float>*>(x), &incx, reinterpret_cast<complex<float>*>(y), &incy);
      return result;
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn vdot_fast_path(
        n:    i32,
        x:    *mut Complex<f64>,
        incx: i32,
        y:    *mut Complex<f64>,
        incy: i32) -> Complex<f64> {
    
    todo!();
        /*
            complex<double> result;
      zdotc_(reinterpret_cast<complex<double>* >(&result), &n, reinterpret_cast<complex<double>*>(x), &incx, reinterpret_cast<complex<double>*>(y), &incy);
      return result;
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn dot_fast_path(
        n:    i32,
        x:    *mut Complex<f64>,
        incx: i32,
        y:    *mut Complex<f64>,
        incy: i32) -> Complex<f64> {
    
    todo!();
        /*
            complex<double> result;
      zdotu_(reinterpret_cast<complex<double>* >(&result), &n, reinterpret_cast<complex<double>*>(x), &incx, reinterpret_cast<complex<double>*>(y), &incy);
      return result;
        */
}

#[cfg(AT_BUILD_WITH_BLAS)]
pub fn dot_fast_path(
        n:    i32,
        x:    *mut Complex<f32>,
        incx: i32,
        y:    *mut Complex<f32>,
        incy: i32) -> Complex<f32> {
    
    todo!();
        /*
            complex<float> result;
      cdotu_(reinterpret_cast<complex<float>* >(&result), &n, reinterpret_cast<complex<float>*>(x), &incx, reinterpret_cast<complex<float>*>(y), &incy);
      return result;
        */
}

pub fn dot_naive<Scalar, Functor>(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64,
        op:   Functor) -> Scalar {

    todo!();
        /*
      i64 i;
      Scalar sum = 0;
      for (i = 0; i < n; i++) {
        sum += op(x[i * incx], y[i * incy]);
      }
      return sum;
        */
}

pub fn dot_impl_floating<Scalar>(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64) -> Scalar {

    todo!();
        /*
            if (n == 1) {
        incx = 1;
        incy = 1;
      }
    #if AT_BUILD_WITH_BLAS()
            if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
              return blas_impl::dot_fast_path(n, x, incx, y, incy);
            } else {
              return blas_impl::dot_naive(n, x, incx, y, incy, multiplies<Scalar>{});
            }
    #else
            { return blas_impl::dot_naive(n, x, incx, y, incy, multiplies<Scalar>{}); }
    #endif
        */
}

pub fn dot_impl_scalar(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64) -> Scalar {

    todo!();
        /*
            if (n == 1) {
        incx = 1;
        incy = 1;
      }
      return blas_impl::dot_naive(n, x, incx, y, incy, multiplies<Scalar>{});
        */
}

pub fn dot_impl_f32(
        n:    i64,
        x:    *mut f32,
        incx: i64,
        y:    *mut f32,
        incy: i64) -> f32 {
    
    todo!();
        /*
            return dot_impl_floating(n, x, incx, y, incy);
        */
}

pub fn dot_impl_f64(
        n:    i64,
        x:    *mut f64,
        incx: i64,
        y:    *mut f64,
        incy: i64) -> f64 {
    
    todo!();
        /*
            return dot_impl_floating(n, x, incx, y, incy);
        */
}

pub fn dot_impl_complex_f64(
        n:    i64,
        x:    *mut Complex<f64>,
        incx: i64,
        y:    *mut Complex<f64>,
        incy: i64) -> Complex<f64> {
    
    todo!();
        /*
            return dot_impl_floating(n, x, incx, y, incy);
        */
}

pub fn dot_impl_complex_f32(
        n:    i64,
        x:    *mut Complex<f32>,
        incx: i64,
        y:    *mut Complex<f32>,
        incy: i64) -> Complex<f32> {
    
    todo!();
        /*
            return dot_impl_floating(n, x, incx, y, incy);
        */
}

pub struct VdotOp {

}

impl VdotOp {
    
    pub fn invoke(&mut self, 
        x: Scalar,
        y: Scalar) -> Scalar {
        
        todo!();
        /*
            return conj(x) * y;
        */
    }
}

pub fn vdot_impl<Scalar>(
        n:    i64,
        x:    *mut Scalar,
        incx: i64,
        y:    *mut Scalar,
        incy: i64) -> Scalar {

    todo!();
        /*
            if (n == 1) {
        incx = 1;
        incy = 1;
      }
    #if AT_BUILD_WITH_BLAS()
            if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
              return blas_impl::vdot_fast_path(n, x, incx, y, incy);
            } else {
              return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<Scalar>{});
            }
    #else
            { return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<Scalar>{}); }
    #endif
        */
}

/// Skip reinstantiating the explicitly
/// specialized types `float` and `double`.
///
#[macro_export] macro_rules! instantiate_dot_impl {
    ($Scalar:ty) => {
        /*
        
          template Scalar dot_impl<Scalar>( 
              i64 n, Scalar * x, i64 incx, Scalar * y, i64 incy);
        */
    }
}

instantiate_dot_impl!{u8}
instantiate_dot_impl!{i8}
instantiate_dot_impl!{i16}
instantiate_dot_impl!{int}
instantiate_dot_impl!{i64}
instantiate_dot_impl!{Half}
instantiate_dot_impl!{BFloat16}

#[macro_export] macro_rules! instantiate_vdot_impl {
    ($Scalar:ty) => {
        /*
        
          template Scalar vdot_impl<Scalar>( 
              i64 n, Scalar * x, i64 incx, Scalar * y, i64 incy);
        */
    }
}

instantiate_vdot_impl!{Complex<f32>}
instantiate_vdot_impl!{Complex<f64>}

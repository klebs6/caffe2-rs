crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp]

pub fn scale(
        m:     i64,
        n:     i64,
        alpha: Scalar,
        a:     *mut Scalar,
        lda:   i64)  {

    todo!();
        /*
            if (alpha == Scalar(1)) {
        return;  // identity
      }

      if (alpha == Scalar(0)) {
        for (i64 j = 0; j < n; j++) {
          for (i64 i = 0; i < m; i++) {
            a[j * lda + i] = Scalar(0);
          }
        }
        return;
      }

      for (i64 j = 0; j < n; j++) {
        for (i64 i = 0; i < m; i++) {
          a[j * lda + i] *= alpha;
        }
      }
        */
}

pub fn gemm_notrans(
        m:     i64,
        n:     i64,
        k:     i64,
        alpha: Scalar,
        a:     *const Scalar,
        lda:   i64,
        b:     *const Scalar,
        ldb:   i64,
        beta:  Scalar,
        c:     *mut Scalar,
        ldc:   i64)  {
    
    todo!();
        /*
            // c *= beta
      scale_(m, n, beta, c, ldc);

      // c += alpha * (a @ b)
      for (i64 l = 0; l < k; l++) {
        for (i64 j = 0; j < n; j++) {
          Scalar val = b[l + j * ldb] * alpha;
          i64 i_m = m / 4;
          for (i64 i_i = 0; i_i < i_m; i_i++) {
            c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
            c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
            c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
            c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
          }
          i64 i = i_m * 4;
          for (; i < m; i++)
            c[j * ldc + i] += a[i + l * lda] * val;
        }
      }
        */
}

pub fn gemm_transa<Scalar>(
        m:     i64,
        n:     i64,
        k:     i64,
        alpha: Scalar,
        a:     *const Scalar,
        lda:   i64,
        b:     *const Scalar,
        ldb:   i64,
        beta:  Scalar,
        c:     *mut Scalar,
        ldc:   i64)  {

    todo!();
        /*
            // c = alpha * (a.T @ b) + beta * c
      const Scalar *a_ = a;
      for (i64 i = 0; i < m; i++)
      {
        const Scalar *b_ = b;
        for (i64 j = 0; j < n; j++)
        {
          Scalar sum = 0;
          for(i64 l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
          if (beta == Scalar(0))
            c[j*ldc+i] = alpha*sum;
          else
            c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
        */
}

pub fn gemm_transb<Scalar>(
        m:     i64,
        n:     i64,
        k:     i64,
        alpha: Scalar,
        a:     *const Scalar,
        lda:   i64,
        b:     *const Scalar,
        ldb:   i64,
        beta:  Scalar,
        c:     *mut Scalar,
        ldc:   i64)  {

    todo!();
        /*
            // c *= beta
      scale_(m, n, beta, c, ldc);

      // c += alpha * (a @ b.T)
      for (i64 l = 0; l < k; l++) {
        for (i64 j = 0; j < n; j++) {
          Scalar val = b[j + l * ldb] * alpha;
          i64 i_m = m / 4;
          for (i64 i_i = 0; i_i < i_m; i_i++) {
            c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
            c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
            c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
            c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
          }
          i64 i = i_m * 4;
          for (; i < m; i++)
            c[j * ldc + i] += a[i + l * lda] * val;
        }
      }
        */
}

pub fn gemm_transab<Scalar>(
        m:     i64,
        n:     i64,
        k:     i64,
        alpha: Scalar,
        a:     *const Scalar,
        lda:   i64,
        b:     *const Scalar,
        ldb:   i64,
        beta:  Scalar,
        c:     *mut Scalar,
        ldc:   i64)  {

    todo!();
        /*
            // c *= beta
      scale_(m, n, beta, c, ldc);

      // c += alpha * (a.T @ b.T)
      for (i64 i = 0; i < m; i++) {
        for (i64 j = 0; j < n; j++) {
          i64 l_k = k / 4;
          for (i64 l_l = 0; l_l < l_k; l_l++) {
            c[j * ldc + i] += a[i * lda + l_l * 4 + 0] //
              * b[(l_l * 4 + 0) * ldb + j] * alpha;
            c[j * ldc + i] += a[i * lda + l_l * 4 + 1] //
              * b[(l_l * 4 + 1) * ldb + j] * alpha;
            c[j * ldc + i] += a[i * lda + l_l * 4 + 2] //
              * b[(l_l * 4 + 2) * ldb + j] * alpha;
            c[j * ldc + i] += a[i * lda + l_l * 4 + 3] //
              * b[(l_l * 4 + 3) * ldb + j] * alpha;
          }
          i64 l = l_k * 4;
          for (; l < k; l++)
            c[j * ldc + i] += a[i * lda + l] * b[l * ldb + j] * alpha;
        }
      }
        */
}

pub fn gemm_core<Scalar>(
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
            if(transa == NoTranspose && transb == NoTranspose) {
        return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      } else if(transa == Transpose && transb != Transpose) {
        gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      } else if(transa == NoTranspose && transb == Transpose) {
        gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      } else {  // transa == Transpose && transb == Transpose
        gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
        */
}

pub fn cpublas_gemm_impl(
        ty:     ScalarType,
        transa: TransposeType,
        transb: TransposeType,
        m:      i64,
        n:      i64,
        k:      i64,
        alpha:  &Scalar,
        a:      *const void,
        lda:    i64,
        b:      *const void,
        ldb:    i64,
        beta:   &Scalar,
        c:      *mut void,
        ldc:    i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16,
        type, "cpublas_gemm_impl",
          [&]{
            gemm_core_(
                transa, transb, m, n, k,
                alpha.to<Scalar>(),
                static_cast<const Scalar *>(a), lda,
                static_cast<const Scalar *>(b), ldb,
                beta.to<Scalar>(),
                static_cast<Scalar *>(c), ldc);
          });
        */
}

pub fn cpublas_axpy_impl(
        ty:   ScalarType,
        n:    i64,
        a:    &Scalar,
        x:    *const void,
        incx: i64,
        y:    *mut void,
        incy: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX(type, "cpublas_axpy_impl",
        [&] {
          auto a = _a.to<Scalar>();
          auto x = static_cast<const Scalar *>(_x);
          auto y = static_cast<Scalar *>(_y);
          i64 i;
          for(i = 0; i < n; i++)
            y[i*incy] += a*x[i*incx];
        });
        */
}

pub fn cpublas_copy_impl(
        ty:   ScalarType,
        n:    i64,
        x:    *const void,
        incx: i64,
        y:    *mut void,
        incy: i64)  {
    
    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX(type, "cpublas_copy_impl",
        [&] {
          auto x = static_cast<const Scalar *>(_x);
          auto y = static_cast<Scalar *>(_y);
          i64 i;
          for(i = 0; i < n; i++)
            y[i*incy] = x[i*incx];
        });
        */
}

register_dispatch!{cpublas::gemm_stub, &cpublas::cpublas_gemm_impl}
register_dispatch!{cpublas::axpy_stub, &cpublas::cpublas_axpy_impl}
register_dispatch!{cpublas::copy_stub, &cpublas::cpublas_copy_impl}

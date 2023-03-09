crate::ix!();

impl SpatialBNGradientOp<CPUContext> {

    #[inline] pub fn compute_scale_bias_gradients_and_fused_params<T>(
        &mut self,
        n:       i32,
        c:       i32,
        hxW:     i32,
        dY:      *const T,
        x:       *const T,
        scale:   *const T,
        mean:    *const T,
        rstd:    *const T,
        dscale:  *mut T,
        dbias:   *mut T,
        alpha:   *mut T,
        beta:    *mut T,
        gamma:   *mut T,
        scratch: *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> scale_arr(scale, C);
          ConstEigenVectorArrayMap<T> mean_arr(mean, C);
          ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
          EigenVectorArrayMap<T> dscale_arr(dscale, C);
          EigenVectorArrayMap<T> dbias_arr(dbias, C);
          EigenVectorArrayMap<T> alpha_arr(alpha, C);
          EigenVectorArrayMap<T> beta_arr(beta, C);
          EigenVectorArrayMap<T> gamma_arr(gamma, C);
          math::Set<T, CPUContext>(C, T(0), dscale, &context_);
          math::Set<T, CPUContext>(C, T(0), dbias, &context_);
          if (order_ == StorageOrder::NCHW) {
            ConstEigenArrayMap<T> dY_arr(dY, HxW, N * C);
            ConstEigenArrayMap<T> X_arr(X, HxW, N * C);
            for (int i = 0; i < N; ++i) {
              for (int j = 0; j < C; ++j) {
                const int c = i * C + j;
                dscale_arr(j) +=
                    (dY_arr.col(c) * (X_arr.col(c) - mean_arr(j)) * rstd_arr(j)).sum();
                dbias_arr(j) += dY_arr.col(c).sum();
              }
            }
          } else {
            const int outer_size = N * HxW;
            ConstEigenArrayMap<T> dY_arr(dY, C, outer_size);
            ConstEigenArrayMap<T> X_arr(X, C, outer_size);
            for (int i = 0; i < outer_size; ++i) {
              dscale_arr += dY_arr.col(i) * (X_arr.col(i) - mean_arr) * rstd_arr;
              dbias_arr += dY_arr.col(i);
            }
          }
          const T inv_nhw = T(1) / static_cast<T>(N * HxW);
          alpha_arr = scale_arr * rstd_arr;
          beta_arr = dscale_arr * rstd_arr;
          gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
          beta_arr *= -alpha_arr * inv_nhw;
        */
    }

    #[inline] pub fn compute_xgradient<T>(
        &mut self,
        n:      i32,
        c:      i32,
        hxW:    i32,
        dY:     *const T,
        x:      *const T,
        alpha:  *const T,
        beta:   *const T,
        gamma:  *const T,
        dX:     *mut T) {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> alpha_arr(alpha, C);
          ConstEigenVectorArrayMap<T> beta_arr(beta, C);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
          if (order_ == NCHW) {
            const int stride = C * HxW;
            const T* dY_ptr = dY;
            const T* X_ptr = X;
            T* dX_ptr = dX;
            for (int i = 0; i < N; ++i) {
              EigenArrayMap<T>(dX_ptr, HxW, C) =
                  (ConstEigenArrayMap<T>(dY_ptr, HxW, C).rowwise() *
                       alpha_arr.transpose() +
                   ConstEigenArrayMap<T>(X_ptr, HxW, C).rowwise() *
                       beta_arr.transpose())
                      .rowwise() +
                  gamma_arr.transpose();
              dY_ptr += stride;
              X_ptr += stride;
              dX_ptr += stride;
            }
          } else {
            EigenArrayMap<T>(dX, C, N * HxW) =
                (ConstEigenArrayMap<T>(dY, C, N * HxW).colwise() * alpha_arr +
                 ConstEigenArrayMap<T>(X, C, N * HxW).colwise() * beta_arr)
                    .colwise() +
                gamma_arr;
          }
        */
    }

    #[inline] pub fn compute_multi_batch_scale_bias_gradients_and_fused_params<T>(
        &mut self,
        n:          i32,
        c:          i32,
        hxW:        i32,
        scale:      *const T,
        mean:       *const T,
        rstd:       *const T,
        dscale_sum: *const T,
        dbias_sum:  *const T,
        dscale:     *mut T,
        dbias:      *mut T,
        alpha:      *mut T,
        beta:       *mut T,
        gamma:      *mut T) 
    {
        todo!();
        /*
            ConstEigenVectorArrayMap<T> scale_arr(scale, C);
          ConstEigenVectorArrayMap<T> mean_arr(mean, C);
          ConstEigenVectorArrayMap<T> rstd_arr(rstd, C);
          EigenVectorArrayMap<T> dscale_arr(dscale, C);
          EigenVectorArrayMap<T> dbias_arr(dbias, C);
          EigenVectorArrayMap<T> alpha_arr(alpha, C);
          EigenVectorArrayMap<T> beta_arr(beta, C);
          EigenVectorArrayMap<T> gamma_arr(gamma, C);
          const T inv_num_batches = T(1) / static_cast<T>(num_batches_);
          math::Scale<T, T, CPUContext>(
              C, inv_num_batches, dscale_sum, dscale, &context_);
          math::Scale<T, T, CPUContext>(
              C, inv_num_batches, dbias_sum, dbias, &context_);
          const T inv_nhw = T(1) / static_cast<T>(N * HxW);
          alpha_arr = scale_arr * rstd_arr;
          beta_arr = dscale_arr * rstd_arr;
          gamma_arr = alpha_arr * (mean_arr * beta_arr - dbias_arr) * inv_nhw;
          beta_arr *= -alpha_arr * inv_nhw;
        */
    }
}

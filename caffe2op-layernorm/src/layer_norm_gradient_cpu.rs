crate::ix!();

impl LayerNormGradientOp<CPUContext> {

    #[inline] pub fn compute_internal_gradients<T>(
        &mut self,
        m:       i32,
        n:       i32,
        dY:      *const T,
        x:       *const T,
        gamma:   *const T,
        d_yxX:   *mut T,
        ds:      *mut T,
        db:      *mut T) 
    {
        todo!();
        /*
            math::Mul<T, CPUContext>(M * N, dY, X, dYxX, &context_);
          ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          if (gamma != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            for (int i = 0; i < M; ++i) {
              ds[i] = (dYxX_arr.col(i) * gamma_arr).sum();
              db[i] = (dY_arr.col(i) * gamma_arr).sum();
            }
          } else {
            EigenVectorArrayMap<T>(ds, M) = dYxX_arr.colwise().sum();
            EigenVectorArrayMap<T>(db, M) = dY_arr.colwise().sum();
          }
        */
    }

    #[inline] pub fn compute_fused_params<T>(
        &mut self,
        m:         i32,
        n:         i32,
        mean:      *const T,
        sigma:     *const T,
        ds:        *const T,
        db:        *const T,
        rstd:      *mut T,
        x_scale:   *mut T,
        bias:      *mut T,
        g_scale:   *mut T) 
    {
        todo!();
        /*
            const T scale = T(1) / static_cast<T>(N);
          ConstEigenVectorArrayMap<T> mean_arr(mean, M);
          ConstEigenVectorArrayMap<T> ds_arr(ds, M);
          ConstEigenVectorArrayMap<T> db_arr(db, M);
          EigenVectorArrayMap<T> rstd_arr(rstd, M);
          EigenVectorArrayMap<T> X_scale_arr(X_scale, M);
          rstd_arr = ConstEigenVectorArrayMap<T>(sigma, M).inverse();
          X_scale_arr = (db_arr * mean_arr - ds_arr) * rstd_arr.cube() * scale;
          EigenVectorArrayMap<T>(bias, M) =
              -X_scale_arr * mean_arr - db_arr * rstd_arr * scale;
          if (g_scale != nullptr) {
            EigenVectorArrayMap<T>(g_scale, M) = -rstd_arr * mean_arr;
          }
        */
    }

    #[inline] pub fn layer_norm_backward<T>(
        &mut self,
        m:         i32,
        n:         i32,
        dY:        *const T,
        x:         *const T,
        gamma:     *const T,
        dY_scale:  *const T,
        x_scale:   *const T,
        bias:      *const T,
        dX:        *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          EigenArrayMap<T> dX_arr(dX, N, M);
          if (gamma != nullptr) {
            ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
            for (int i = 0; i < M; ++i) {
              dX_arr.col(i) = dY_arr.col(i) * gamma_arr * dY_scale[i] +
                  X_arr.col(i) * X_scale[i] + bias[i];
            }
          } else {
            ConstEigenVectorArrayMap<T> dY_scale_arr(dY_scale, M);
            ConstEigenVectorArrayMap<T> X_scale_arr(X_scale, M);
            ConstEigenVectorArrayMap<T> bias_arr(bias, M);
            dX_arr = (dY_arr.rowwise() * dY_scale_arr.transpose() +
                      X_arr.rowwise() * X_scale_arr.transpose())
                         .rowwise() +
                bias_arr.transpose();
          }
        */
    }

    #[inline] pub fn gamma_beta_backward<T>(
        &mut self,
        m:        i32,
        n:        i32,
        d_yxX:    *const T,
        dY:       *const T,
        rstd:     *const T,
        g_scale:  *const T,
        dgamma:   *mut T,
        dbeta:    *mut T)  
    {
        todo!();
        /*
            math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
          math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
          ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
          EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
          for (int i = 0; i < M; ++i) {
            dgamma_arr += dYxX_arr.col(i) * rstd[i] + dY_arr.col(i) * g_scale[i];
            dbeta_arr += dY_arr.col(i);
          }
        */
    }
}


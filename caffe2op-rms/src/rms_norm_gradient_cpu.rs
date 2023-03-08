crate::ix!();

impl RMSNormGradientOp<CPUContext> {

    #[inline] pub fn rms_norm_backward<T>(
        &mut self,
        m:      i64,
        n:      i64,
        dY:     *const T,
        x:      *const T,
        gamma:  *const T,
        rrms:   *const T,
        dX:     *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
          EigenArrayMap<T> dX_arr(dX, N, M);
          const T scale = T(1) / static_cast<T>(N);
          at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
              const T ds = (dY_arr.col(i) * X_arr.col(i) * gamma_arr).sum();
              const T c1 = rrms[i];
              const T c2 = -scale * ds * math::utils::Cube<T>(rrms[i]);
              dX_arr.col(i) = c1 * dY_arr.col(i) * gamma_arr + c2 * X_arr.col(i);
            }
          });
        */
    }

    #[inline] pub fn gamma_beta_backward<T>(
        &mut self,
        m:        i64,
        n:        i64,
        dY:       *const T,
        x:        *const T,
        rrms:     *const T,
        dgamma:   *mut T,
        dbeta:    *mut T) 
    {
        todo!();
        /*
            math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
          math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
          EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
          for (int64_t i = 0; i < M; ++i) {
            dgamma_arr += dY_arr.col(i) * X_arr.col(i) * rrms[i];
            dbeta_arr += dY_arr.col(i);
          }
        */
    }
}

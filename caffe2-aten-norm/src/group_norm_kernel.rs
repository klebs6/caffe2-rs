crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/group_norm_kernel.cpp]

pub fn group_norm_kernel_impl_internal<T>(
        X:     &Tensor,
        gamma: &Tensor,
        beta:  &Tensor,
        N:     i64,
        C:     i64,
        hxw:   i64,
        group: i64,
        eps:   T,
        Y:     &mut Tensor,
        mean:  &mut Tensor,
        rstd:  &mut Tensor)  {

    todo!();
        /*
            TORCH_CHECK(X.numel() == N * C * HxW);
      TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
      TORCH_CHECK(!beta.defined() || beta.numel() == C);
      const i64 G = group;
      const i64 D = C / G;
      const T* X_data = X.data_ptr<T>();
      const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
      const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
      T* Y_data = Y.data_ptr<T>();
      T* mean_data = mean.data_ptr<T>();
      T* rstd_data = rstd.data_ptr<T>();
      const bool gamma_null = (gamma_data == nullptr);
      const bool beta_null = beta_data == nullptr;
      const i64 inner_size = D * HxW;

      parallel_for(0, N * G, 1, [&](i64 start, i64 end) {
        for (i64 i = start; i < end; ++i) {
          const T* X_ptr = X_data + i * inner_size;
          T mean_val;
          T rstd_val;
          tie(mean_val, rstd_val) = utils::RowwiseMoments(X_ptr, inner_size);
          rstd_val = T(1) / sqrt(max(rstd_val, T(0)) + eps);
          if (gamma_null && beta_null) {
            T* Y_ptr = Y_data + i * inner_size;
            for (int j = 0; j < inner_size; ++j) {
              Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
            }
          } else {
            const i64 g = i % G;
            for (i64 j = 0; j < D; ++j) {
              const i64 c = g * D + j;
              const T scale = rstd_val * (gamma_null ? T(1) : gamma_data[c]);
              const T bias = -scale * mean_val + (beta_null ? T(0) : beta_data[c]);
              X_ptr = X_data + (i * D + j) * HxW;
              T* Y_ptr = Y_data + (i * D + j) * HxW;
              for (i64 k = 0; k < HxW; ++k) {
                Y_ptr[k] = scale * X_ptr[k] + bias;
              }
            }
          }
          mean_data[i] = mean_val;
          rstd_data[i] = rstd_val;
        }
      });
        */
}

pub fn group_norm_kernel_impl(
        X:     &Tensor,
        gamma: &Tensor,
        beta:  &Tensor,
        N:     i64,
        C:     i64,
        hxw:   i64,
        group: i64,
        eps:   f64,
        Y:     &mut Tensor,
        mean:  &mut Tensor,
        rstd:  &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GroupNormKernelImpl", [&]() {
        GroupNormKernelImplInternal<Scalar>(
            X,
            gamma,
            beta,
            N,
            C,
            HxW,
            group,
            static_cast<Scalar>(eps),
            Y,
            mean,
            rstd);
      });
        */
}

pub fn compute_internal_gradients<T>(
        N:   i64,
        C:   i64,
        hxw: i64,
        dy:  *const T,
        X:   *const T,
        ds:  *mut T,
        db:  *mut T)  {

    todo!();
        /*
            parallel_for(0, N * C, 1, [=](i64 start, i64 end) {
        constexpr i64 K = vec::Vectorized<T>::size();
        const i64 inner_size = HxW / K * K;
        array<T, K> ds_arr;
        array<T, K> db_arr;
        for (i64 i = start; i < end; ++i) {
          const T* dY_ptr = dY + i * HxW;
          const T* X_ptr = X + i * HxW;
          vec::Vectorized<T> ds_vec(0);
          vec::Vectorized<T> db_vec(0);
          for (i64 j = 0; j < inner_size; j += K) {
            const vec::Vectorized<T> dy_vec = vec::Vectorized<T>::loadu(dY_ptr + j);
            const vec::Vectorized<T> x_vec = vec::Vectorized<T>::loadu(X_ptr + j);
            ds_vec = ds_vec + dy_vec * x_vec;
            db_vec = db_vec + dy_vec;
          }
          ds_vec.store(ds_arr.data());
          db_vec.store(db_arr.data());
          T ds_val = accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
          T db_val = accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
          for (i64 j = inner_size; j < HxW; ++j) {
            ds_val += dY_ptr[j] * X_ptr[j];
            db_val += dY_ptr[j];
          }
          ds[i] = ds_val;
          db[i] = db_val;
        }
      });
        */
}

pub fn group_norm_input_backward<T>(
        N:     i64,
        C:     i64,
        hxw:   i64,
        group: i64,
        dy:    *const T,
        X:     *const T,
        mean:  *const T,
        rstd:  *const T,
        gamma: *const T,
        ds:    *const T,
        db:    *const T,
        dx:    *mut T)  {

    todo!();
        /*
            const i64 G = group;
      const i64 D = C / G;
      const T s = T(1) / static_cast<T>(D * HxW);
      const bool gamma_null = (gamma == nullptr);
      parallel_for(0, N * G, 1, [=](i64 start, i64 end) {
        constexpr i64 K = vec::Vectorized<T>::size();
        const i64 d = D / K * K;
        array<T, K> ds_arr;
        array<T, K> db_arr;
        for (i64 i = start; i < end; ++i) {
          const i64 g = i % G;
          const T* ds_ptr = ds + i * D;
          const T* db_ptr = db + i * D;
          vec::Vectorized<T> ds_vec(0);
          vec::Vectorized<T> db_vec(0);
          for (i64 j = 0; j < d; j += K) {
            const vec::Vectorized<T> gamma_vec = gamma_null
                ? vec::Vectorized<T>(1)
                : vec::Vectorized<T>::loadu(gamma + g * D + j);
            ds_vec = ds_vec + vec::Vectorized<T>::loadu(ds_ptr + j) * gamma_vec;
            db_vec = db_vec + vec::Vectorized<T>::loadu(db_ptr + j) * gamma_vec;
          }
          ds_vec.store(ds_arr.data());
          db_vec.store(db_arr.data());
          T ds_val = accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
          T db_val = accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
          for (i64 j = d; j < D; ++j) {
            const T gamma_v = gamma_null ? T(1) : gamma[g * D + j];
            ds_val += ds_ptr[j] * gamma_v;
            db_val += db_ptr[j] * gamma_v;
          }
          const T c2 =
              (db_val * mean[i] - ds_val) * rstd[i] * rstd[i] * rstd[i] * s;
          const T c3 = -c2 * mean[i] - db_val * rstd[i] * s;
          for (i64 j = 0; j < D; ++j) {
            const i64 c = g * D + j;
            const T* dY_ptr = dY + (i * D + j) * HxW;
            const T* X_ptr = X + (i * D + j) * HxW;
            T* dX_ptr = dX + (i * D + j) * HxW;
            const T c1 = rstd[i] * (gamma_null ? T(1) : gamma[c]);
            for (i64 k = 0; k < HxW; ++k) {
              dX_ptr[k] = c1 * dY_ptr[k] + c2 * X_ptr[k] + c3;
            }
          }
        }
      });
        */
}

pub fn gamma_backward<T>(
        N:      i64,
        C:      i64,
        group:  i64,
        mean:   *const T,
        rstd:   *const T,
        ds:     *const T,
        db:     *const T,
        dgamma: *mut T)  {

    todo!();
        /*
            const i64 G = group;
      const i64 D = C / G;
      constexpr i64 K = vec::Vectorized<T>::size();
      parallel_for(0, D, K, [=](i64 start, i64 end) {
        for (i64 i = 0; i < G; ++i) {
          memset(dgamma + i * D + start, 0, (end - start) * sizeof(T));
        }
        for (i64 i = 0; i < N * G; ++i) {
          const T* ds_ptr = ds + i * D;
          const T* db_ptr = db + i * D;
          const i64 g = i % G;
          for (i64 j = start; j < end; ++j) {
            const i64 c = g * D + j;
            dgamma[c] += (ds_ptr[j] - db_ptr[j] * mean[i]) * rstd[i];
          }
        }
      });
        */
}

pub fn beta_backward<T>(
        N:     i64,
        C:     i64,
        db:    *const T,
        dbeta: *mut T)  {

    todo!();
        /*
            constexpr i64 K = vec::Vectorized<T>::size();
      parallel_for(0, C, K, [=](i64 start, i64 end) {
        memset(dbeta + start, 0, (end - start) * sizeof(T));
        for (i64 i = 0; i < N; ++i) {
          const T* db_ptr = db + i * C;
          for (i64 j = start; j < end; ++j) {
            dbeta[j] += db_ptr[j];
          }
        }
      });
        */
}

pub fn group_norm_backward_kernel_impl_internal<T>(
        dy:     &Tensor,
        X:      &Tensor,
        mean:   &Tensor,
        rstd:   &Tensor,
        gamma:  &Tensor,
        N:      i64,
        C:      i64,
        hxw:    i64,
        group:  i64,
        dx:     &mut Tensor,
        dgamma: &mut Tensor,
        dbeta:  &mut Tensor)  {

    todo!();
        /*
            TORCH_CHECK(dY.numel() == N * C * HxW);
      TORCH_CHECK(X.numel() == N * C * HxW);
      TORCH_CHECK(mean.numel() == N * group);
      TORCH_CHECK(rstd.numel() == N * group);
      TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

      const T* dY_data = dY.data_ptr<T>();
      const T* X_data = X.data_ptr<T>();
      const T* mean_data = mean.data_ptr<T>();
      const T* rstd_data = rstd.data_ptr<T>();
      const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
      T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
      T* dgamma_data = dgamma.defined() ? dgamma.data_ptr<T>() : nullptr;
      T* dbeta_data = dbeta.defined() ? dbeta.data_ptr<T>() : nullptr;
      Tensor ds = empty({N, C}, X.options());
      Tensor db = empty({N, C}, X.options());
      T* ds_data = ds.data_ptr<T>();
      T* db_data = db.data_ptr<T>();

      ComputeInternalGradients<T>(N, C, HxW, dY_data, X_data, ds_data, db_data);

      if (dX_data != nullptr) {
        GroupNormInputBackward<T>(
            N,
            C,
            HxW,
            group,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            gamma_data,
            ds_data,
            db_data,
            dX_data);
      }
      if (dgamma_data != nullptr) {
        GammaBackward<T>(
            N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
      }
      if (dbeta_data != nullptr) {
        BetaBackward<T>(N, C, db_data, dbeta_data);
      }
        */
}

pub fn group_norm_backward_kernel_impl(
        dy:     &Tensor,
        X:      &Tensor,
        mean:   &Tensor,
        rstd:   &Tensor,
        gamma:  &Tensor,
        N:      i64,
        C:      i64,
        hxw:    i64,
        group:  i64,
        dx:     &mut Tensor,
        dgamma: &mut Tensor,
        dbeta:  &mut Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(
          X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
            GroupNormBackwardKernelImplInternal<Scalar>(
                dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
          });
        */
}

register_dispatch!{GroupNormKernel         , &GroupNormKernelImpl}
register_dispatch!{GroupNormBackwardKernel , &GroupNormBackwardKernelImpl}

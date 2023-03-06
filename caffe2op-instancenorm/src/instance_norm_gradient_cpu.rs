crate::ix!();

impl InstanceNormGradientOp<f32, CPUContext> {

    #[inline] pub fn compute_moments(
        n:    i64,
        c:    i64,
        hxW:  i64,
        x:    *const f32,
        mean: *mut f32,
        rstd: *mut f32) 
    {
        todo!();
        /*
            if (order_ == StorageOrder::NCHW) {
            const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                               static_cast<int>(HxW)};
            const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
            math::Moments<float, CPUContext>(
                2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
            math::InvStd<float, CPUContext>(N * C, epsilon_, rstd, rstd, &context_);
          } else {
            const float c = 1.0f / static_cast<float>(HxW);
            EigenArrayMap<float> mean_arr(mean, C, N);
            EigenArrayMap<float> rstd_arr(rstd, C, N);
            for (int64_t i = 0; i < N; ++i) {
              ConstEigenArrayMap<float> X_arr(X + i * HxW * C, C, HxW);
              mean_arr.col(i) = X_arr.col(0);
              rstd_arr.col(i) = X_arr.col(0).square();
              for (int64_t j = 1; j < HxW; ++j) {
                mean_arr.col(i) += X_arr.col(j);
                rstd_arr.col(i) += X_arr.col(j).square();
              }
            }
            mean_arr *= c;
            rstd_arr =
                ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
          }
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nchw(
        &mut self, 
        n:      i64,
        c:      i64,
        hxW:    i64,
        dY:     *const f32,
        x:      *const f32,
        mean:   *const f32,
        rstd:   *const f32,
        gamma:  *const f32,
        dX:     *mut f32,
        dgamma: *mut f32,
        dbeta:  *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      InstanceNormBackwardNCHW<float>(
          N, C, HxW, dY, X, mean, rstd, gamma, dX, ds_data, db_data);
      GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(
        &mut self, 
        n:       i64,
        c:       i64,
        hxW:     i64,
        dY:      *const f32,
        x:       *const f32,
        mean:    *const f32,
        rstd:    *const f32,
        gamma:   *const f32,
        dX:      *mut f32,
        dgamma:  *mut f32,
        dbeta:   *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      ComputeInternalGradientsNHWC<float>(N, C, HxW, dY, X, ds_data, db_data);
      ReinitializeTensor(&c1_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&c2_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&c3_, {N, C}, at::dtype<float>().device(CPU));
      float* c1_data = c1_.mutable_data<float>();
      float* c2_data = c2_.mutable_data<float>();
      float* c3_data = c3_.mutable_data<float>();
      InstanceNormBackwardNHWC<float>(
          N,
          C,
          HxW,
          dY,
          X,
          ds_data,
          db_data,
          mean,
          rstd,
          gamma,
          dX,
          c1_data,
          c2_data,
          c3_data);
      GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
      return true;
        */
    }
}

register_cpu_operator!{
    InstanceNormGradient,
    InstanceNormGradientOp<f32, CPUContext>
}

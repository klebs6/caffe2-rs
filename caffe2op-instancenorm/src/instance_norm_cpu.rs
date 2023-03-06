crate::ix!();

impl<T, Context> InstanceNormOp<T, Context> {
    
    #[inline] pub fn run_f32_on_cpu_device_with_order_nchw(
        &mut self, 
        n:      i64,
        c:      i64,
        hxW:    i64,
        x:      *const f32,
        gamma:  *const f32,
        beta:   *const f32,
        y:      *mut f32,
        mean:   *mut f32,
        rstd:   *mut f32) -> bool 
    {

        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int64_t i = 0; i < N * C; ++i) {
        const float mean_val = X_arr.col(i).mean();
        float rstd_val =
            std::max(X_arr.col(i).square().mean() - mean_val * mean_val, 0.0f);
        rstd_val = 1.0f / std::sqrt(rstd_val + epsilon_);
        const int64_t c = i % C;
        const float scale = gamma[c] * rstd_val;
        const float bias = beta[c] - scale * mean_val;
        for (int64_t j = 0; j < HxW; ++j) {
          Y[i * HxW + j] = scale * X[i * HxW + j] + bias;
        }
        mean[i] = mean_val;
        rstd[i] = rstd_val;
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(
        &mut self, 
        n:       i64,
        c:       i64,
        hxW:     i64,
        x:       *const f32,
        gamma:   *const f32,
        beta:    *const f32,
        y:       *mut f32,
        mean:    *mut f32,
        rstd:    *mut f32) -> bool 
    {
        todo!();
        /*
            ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
      float* scale_data = scale_.template mutable_data<float>();
      float* bias_data = bias_.template mutable_data<float>();
      const float c = 1.0f / static_cast<float>(HxW);
      EigenArrayMap<float> mean_arr(mean, C, N);
      EigenArrayMap<float> rstd_arr(rstd, C, N);
      for (int64_t n = 0; n < N; ++n) {
        ConstEigenArrayMap<float> X_arr(X + n * HxW * C, C, HxW);
        mean_arr.col(n) = X_arr.col(0);
        rstd_arr.col(n) = X_arr.col(0).square();
        for (int64_t i = 1; i < HxW; ++i) {
          mean_arr.col(n) += X_arr.col(i);
          rstd_arr.col(n) += X_arr.col(i).square();
        }
      }
      mean_arr *= c;
      rstd_arr = ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
      ComputeFusedParams<float>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
      InstanceNormForwardNHWC<float>(N, C, HxW, X, scale_data, bias_data, Y);
      return true;
        */
    }
}

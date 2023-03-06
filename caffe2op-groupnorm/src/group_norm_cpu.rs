crate::ix!();

impl GroupNormOp<f32, CPUContext> {

    #[inline] pub fn compute_fused_params(
        &mut self, 
        n:     i32,
        g:     i32,
        k:     i32,
        mu:    *const f32,
        rsig:  *const f32,
        gamma: *const f32,
        beta:  *const f32,
        scale: *mut f32,
        bias:  *mut f32)  
    {
        todo!();
        /*
            const int C = G * K;
      ConstEigenArrayMap<float> mu_arr(mu, G, N);
      ConstEigenArrayMap<float> rsig_arr(rsig, G, N);
      ConstEigenArrayMap<float> gamma_arr(gamma, K, G);
      ConstEigenArrayMap<float> beta_arr(beta, K, G);
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float> scale_arr(scale + i * C, K, G);
        EigenArrayMap<float> bias_arr(bias + i * C, K, G);
        scale_arr = gamma_arr.rowwise() * rsig_arr.col(i).transpose();
        bias_arr = beta_arr - scale_arr.rowwise() * mu_arr.col(i).transpose();
      }
        */
    }
    
    #[inline] pub fn group_norm_forwardNCHW(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        scale:  *const f32,
        bias:   *const f32,
        y:      *mut f32)  
    {
        todo!();
        /*
            EigenArrayMap<float>(Y, HxW, N * C) =
          (ConstEigenArrayMap<float>(X, HxW, N * C).rowwise() *
           ConstEigenVectorArrayMap<float>(scale, N * C).transpose())
              .rowwise() +
          ConstEigenVectorArrayMap<float>(bias, N * C).transpose();
        */
    }
    
    #[inline] pub fn group_norm_forwardNHWC(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        scale:  *const f32,
        bias:   *const f32,
        y:      *mut f32)  
    {
        todo!();
        /*
            const int stride = HxW * C;
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float>(Y + i * stride, C, HxW) =
            (ConstEigenArrayMap<float>(X + i * stride, C, HxW).colwise() *
             ConstEigenVectorArrayMap<float>(scale + i * C, C))
                .colwise() +
            ConstEigenVectorArrayMap<float>(bias + i * C, C);
      }
        */
    }
    
    #[inline] pub fn run_f32_on_cpu_device_with_orderNHWC(
        &mut self, 
        n:      i32,
        g:      i32,
        k:      i32,
        hxW:    i32,
        x:      *const f32,
        gamma:  *const f32,
        beta:   *const f32,
        y:      *mut f32,
        mu:     *mut f32,
        rsig:   *mut f32) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
      ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
      float* scale_data = scale_.mutable_data<float>();
      float* bias_data = bias_.mutable_data<float>();
      EigenVectorArrayMap<float> mu_arr(mu, N * G);
      EigenVectorArrayMap<float> rsig_arr(rsig, N * G);
      mu_arr.setZero();
      rsig_arr.setZero();
      for (int n = 0; n < N; ++n) {
        for (int i = 0; i < HxW; ++i) {
          const int m = n * HxW + i;
          ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
          for (int j = 0; j < G; ++j) {
            mu_arr(n * G + j) += X_arr.col(j).sum();
            rsig_arr(n * G + j) += X_arr.col(j).square().sum();
          }
        }
      }
      const float scale = 1.0f / static_cast<float>(K * HxW);
      mu_arr *= scale;
      rsig_arr = (rsig_arr * scale - mu_arr.square() + epsilon_).rsqrt();
      ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
      GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
      return true;
        */
    }
}

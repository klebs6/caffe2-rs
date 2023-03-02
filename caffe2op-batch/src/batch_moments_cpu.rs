crate::ix!();

impl BatchMomentsOp<f32, CPUContext> {

    #[inline] pub fn compute_batch_moments_nchw(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        mu:     *mut f32,
        var:    *mut f32) -> bool 
    {
        todo!();
        /*
            math::Set<float, CPUContext>(C, 0.0f, mu, &context_);
      math::Set<float, CPUContext>(C, 0.0f, var, &context_);
      EigenVectorArrayMap<float> mu_arr(mu, C);
      EigenVectorArrayMap<float> var_arr(var, C);
      const float* X_ptr = X;
      const int stride = C * HxW;
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> X_arr(X_ptr, HxW, C);
        mu_arr += X_arr.colwise().sum();
        var_arr += X_arr.square().colwise().sum();
        X_ptr += stride;
      }
      const float scale = 1.0f / static_cast<float>(N * HxW);
      math::Scale<float, float, CPUContext>(C, scale, mu, mu, &context_);
      math::Scale<float, float, CPUContext>(C, scale, var, var, &context_);
      return true;
        */
    }
    
    #[inline] pub fn compute_batch_moments_nhwc(
        &mut self, 
        n:    i32,
        c:    i32,
        hxW:  i32,
        x:    *const f32,
        mu:   *mut f32,
        var:  *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
      EigenVectorMap<float>(mu, C) = X_arr.rowwise().mean();
      EigenVectorMap<float>(var, C) = X_arr.square().rowwise().mean();
      return true;
        */
    }
}


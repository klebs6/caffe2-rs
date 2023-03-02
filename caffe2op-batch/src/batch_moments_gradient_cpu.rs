crate::ix!();

impl BatchMomentsGradientOp<f32, CPUContext> {

    #[inline] pub fn compute_batch_moments_gradientNCHW(
        &mut self, 
        n:     i32,
        c:     i32,
        hxW:   i32,
        dmu:   *const f32,
        dvar:  *const f32,
        x:     *const f32,
        dX:    *mut f32) -> bool 
    {

        todo!();
        /*
            ConstEigenVectorArrayMap<float> dmu_arr(dmu, C);
      ConstEigenVectorArrayMap<float> dvar_arr(dvar, C);
      const float* X_ptr = X;
      float* dX_ptr = dX;
      const int stride = C * HxW;
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<float> dX_arr(dX_ptr, HxW, C);
        dX_arr = ConstEigenArrayMap<float>(X_ptr, HxW, C).rowwise() *
            dvar_arr.transpose() * 2.0f;
        dX_arr.rowwise() += dmu_arr.transpose();
        X_ptr += stride;
        dX_ptr += stride;
      }
      const float scale = 1.0f / static_cast<float>(N * HxW);
      math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
      return true;
        */
    }
    
    #[inline] pub fn compute_batch_moments_gradientNHWC(
        &mut self, 
        n:     i32,
        c:     i32,
        hxW:   i32,
        dmu:   *const f32,
        dvar:  *const f32,
        x:     *const f32,
        dX:    *mut f32) -> bool 
    {
        todo!();
        /*
            const float scale = 1.0f / static_cast<float>(N * HxW);
      EigenArrayMap<float> dX_arr(dX, C, N * HxW);
      dX_arr = ConstEigenArrayMap<float>(X, C, N * HxW).colwise() *
          ConstEigenVectorArrayMap<float>(dvar, C) * 2.0f;
      dX_arr.colwise() += ConstEigenVectorArrayMap<float>(dmu, C);
      math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
      return true;
        */
    }
}


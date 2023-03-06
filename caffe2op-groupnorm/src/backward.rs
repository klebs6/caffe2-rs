crate::ix!();

#[inline] pub fn group_norm_backward_i32_nchw(
    n:         i32,
    g:         i32,
    k:         i32,
    hxW:       i32,
    dY_scale:  *const f32,
    dY:        *const f32,
    x_scale:   *const f32,
    x:         *const f32,
    bias:      *const f32,
    dX:        *mut f32)  
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      EigenArrayMap<float> dX_arr(dX, HxW, N * C);
      for (int i = 0; i < N * G; ++i) {
        for (int j = 0; j < K; ++j) {
          const int c = i * K + j;
          dX_arr.col(c) =
              dY_arr.col(c) * dY_scale[c] + X_arr.col(c) * X_scale[i] + bias[i];
        }
      }
    */
}

#[inline] pub fn group_norm_backward_f32_nhwc(
    n:          i32,
    g:          i32,
    k:          i32,
    hxW:        i32,
    dY_scale:   *const f32,
    dY:         *const f32,
    x_scale:    *const f32,
    x:          *const f32,
    bias:       *const f32,
    dX:         *mut f32)
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<float> X_scale_arr(X_scale, G, N);
      ConstEigenArrayMap<float> bias_arr(bias, G, N);
      for (int n = 0; n < N; ++n) {
        ConstEigenArrayMap<float> dY_scale_arr(dY_scale + n * C, K, G);
        for (int i = 0; i < HxW; ++i) {
          const int m = n * HxW + i;
          ConstEigenArrayMap<float> dY_arr(dY + m * C, K, G);
          ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
          EigenArrayMap<float> dX_arr(dX + m * C, K, G);
          dX_arr = (dY_arr * dY_scale_arr +
                    X_arr.rowwise() * X_scale_arr.col(n).transpose())
                       .rowwise() +
              bias_arr.col(n).transpose();
        }
      }
    */
}

crate::ix!();

#[inline] pub fn affine_channel_scale_bias_backwardNCHW<T>(
    n:        i32,
    c:        i32,
    hxW:      i32,
    dY:       *const T,
    x:        *const T,
    dscale:   *mut T,
    dbias:    *mut T) 
{
    todo!();
    /*
        const T* dY_ptr = dY;
      const T* X_ptr = X;
      const int stride = C * HxW;
      EigenVectorArrayMap<T> dscale_arr(dscale, C);
      EigenVectorArrayMap<T> dbias_arr(dbias, C);
      dscale_arr.setZero();
      dbias_arr.setZero();
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY_ptr, HxW, C);
        ConstEigenArrayMap<T> X_arr(X_ptr, HxW, C);
        dscale_arr += (dY_arr * X_arr).colwise().sum();
        dbias_arr += dY_arr.colwise().sum();
        dY_ptr += stride;
        X_ptr += stride;
      }
    */
}

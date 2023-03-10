crate::ix!();

#[inline] pub fn affine_channel_scale_bias_backwardNHWC<T>(
    n:         i32,
    c:         i32,
    hxW:       i32,
    dY:        *const T,
    x:         *const T,
    dscale:    *mut T,
    dbias:     *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> dY_arr(dY, C, N * HxW);
      ConstEigenArrayMap<T> X_arr(X, C, N * HxW);
      EigenVectorMap<T>(dscale, C) = (dY_arr * X_arr).rowwise().sum();
      EigenVectorMap<T>(dbias, C) = dY_arr.rowwise().sum();
    */
}

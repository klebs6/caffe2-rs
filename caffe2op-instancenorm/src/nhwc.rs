crate::ix!();

#[inline] pub fn instance_norm_forwardNHWC<T>(
    n:     i64,
    c:     i64,
    hxW:   i64,
    x:     *const T,
    scale: *const T,
    bias:  *const T,
    y:     *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> scale_arr(scale, C, N);
      ConstEigenArrayMap<T> bias_arr(bias, C, N);
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
        EigenArrayMap<T> Y_arr(Y + i * HxW * C, C, HxW);
        Y_arr = (X_arr.colwise() * scale_arr.col(i)).colwise() + bias_arr.col(i);
      }
    */
}

#[inline] pub fn compute_internal_gradientsNHWC<T>(
    n:   i64,
    c:   i64,
    hxW: i64,
    dY:  *const T,
    x:   *const T,
    ds:  *mut T,
    db:  *mut T) 
{
    todo!();
    /*
        EigenArrayMap<T> ds_arr(ds, C, N);
      EigenArrayMap<T> db_arr(db, C, N);
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY + i * C * HxW, C, HxW);
        ConstEigenArrayMap<T> X_arr(X + i * C * HxW, C, HxW);
        ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
        db_arr.col(i) = dY_arr.col(0);
        for (int j = 1; j < HxW; ++j) {
          ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
          db_arr.col(i) += dY_arr.col(j);
        }
      }
    */
}

#[inline] pub fn instance_norm_backwardNHWC<T>(
    n:       i64,
    c:       i64,
    hxW:     i64,
    dY:      *const T,
    x:       *const T,
    ds:      *const T,
    db:      *const T,
    mean:    *const T,
    rstd:    *const T,
    gamma:   *const T,
    dX:      *mut T,
    c1:      *mut T,
    c2:      *mut T,
    c3:      *mut T) 
{
    todo!();
    /*
        const T scale = T(1) / static_cast<T>(HxW);
      ConstEigenArrayMap<T> ds_arr(ds, C, N);
      ConstEigenArrayMap<T> db_arr(db, C, N);
      ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
      EigenArrayMap<T> c1_arr(c1, C, N);
      EigenArrayMap<T> c2_arr(c2, C, N);
      EigenArrayMap<T> c3_arr(c3, C, N);
      c1_arr = rstd_arr.colwise() * gamma_arr;
      c2_arr = ds_arr.colwise() * gamma_arr;
      c3_arr = db_arr.colwise() * gamma_arr;
      c2_arr = (c3_arr * mean_arr - c2_arr) * rstd_arr.cube() * scale;
      c3_arr = -c2_arr * mean_arr - c3_arr * rstd_arr * scale;
      for (int64_t i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY + i * HxW * C, C, HxW);
        ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
        EigenArrayMap<T> dX_arr(dX + i * HxW * C, C, HxW);
        dX_arr =
            (dY_arr.colwise() * c1_arr.col(i) + X_arr.colwise() * c2_arr.col(i))
                .colwise() +
            c3_arr.col(i);
      }
    */
}

crate::ix!();

#[inline] pub fn instance_norm_backwardNCHW<T>(
    n:      i64,
    c:      i64,
    hxW:    i64,
    dY:     *const T,
    x:      *const T,
    mean:   *const T,
    rstd:   *const T,
    gamma:  *const T,
    dX:     *mut T,
    ds:     *mut T,
    db:     *mut T) 
{
    todo!();
    /*
       const T scale = T(1) / static_cast<T>(HxW);
      ConstEigenArrayMap<T> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<T> X_arr(X, HxW, N * C);
      for (int64_t i = 0; i < N * C; ++i) {
        const T ds_sum = (dY_arr.col(i) * X_arr.col(i)).sum();
        const T db_sum = dY_arr.col(i).sum();
        const int64_t c = i % C;
        const T c1 = rstd[i] * gamma[c];
        T c2 = ds_sum * gamma[c];
        T c3 = db_sum * gamma[c];
        c2 = (c3 * mean[i] - c2) * rstd[i] * rstd[i] * rstd[i] * scale;
        c3 = -c2 * mean[i] - c3 * rstd[i] * scale;
        for (int64_t j = 0; j < HxW; ++j) {
          const int64_t index = i * HxW + j;
          dX[index] = c1 * dY[index] + c2 * X[index] + c3;
        }
        ds[i] = ds_sum;
        db[i] = db_sum;
      }
    */
}

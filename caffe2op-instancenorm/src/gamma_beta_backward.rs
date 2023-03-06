crate::ix!();

#[inline] pub fn gamma_beta_backward<T>(
    n:      i64,
    c:      i64,
    ds:     *const T,
    db:     *const T,
    mean:   *const T,
    rstd:   *const T,
    dgamma: *mut T,
    dbeta:  *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> ds_arr(ds, C, N);
      ConstEigenArrayMap<T> db_arr(db, C, N);
      ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      EigenVectorArrayMap<T> dgamma_arr(dgamma, C);
      EigenVectorArrayMap<T> dbeta_arr(dbeta, C);
      dgamma_arr =
          (ds_arr.col(0) - db_arr.col(0) * mean_arr.col(0)) * rstd_arr.col(0);
      dbeta_arr = db_arr.col(0);
      for (int64_t i = 1; i < N; ++i) {
        dgamma_arr +=
            (ds_arr.col(i) - db_arr.col(i) * mean_arr.col(i)) * rstd_arr.col(i);
        dbeta_arr += db_arr.col(i);
      }
    */
}



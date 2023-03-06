crate::ix!();

#[inline] pub fn gamma_beta_backward<T>(
    n:        i32,
    g:        i32,
    k:        i32,
    ds:       *const T,
    db:       *const T,
    mu:       *const T,
    rsig:     *const T,
    dgamma:   *mut T,
    dbeta:    *mut T) 
{
    todo!();
    /*
        const int C = G * K;
      ConstEigenArrayMap<T> ds0_arr(ds, K, G);
      ConstEigenArrayMap<T> db0_arr(db, K, G);
      ConstEigenArrayMap<T> mu_arr(mu, G, N);
      ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
      EigenArrayMap<T> dgamma_arr(dgamma, K, G);
      EigenArrayMap<T> dbeta_arr(dbeta, K, G);
      dgamma_arr =
          (ds0_arr - db0_arr.rowwise() * mu_arr.col(0).transpose()).rowwise() *
          rsig_arr.col(0).transpose();
      dbeta_arr = db0_arr;
      for (int i = 1; i < N; ++i) {
        ConstEigenArrayMap<T> dsi_arr(ds + i * C, K, G);
        ConstEigenArrayMap<T> dbi_arr(db + i * C, K, G);
        dgamma_arr +=
            (dsi_arr - dbi_arr.rowwise() * mu_arr.col(i).transpose()).rowwise() *
            rsig_arr.col(i).transpose();
        dbeta_arr += dbi_arr;
      }
    */
}

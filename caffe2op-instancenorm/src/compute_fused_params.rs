crate::ix!();

#[inline] pub fn compute_fused_params<T>(
    n:       i64,
    c:       i64,
    mean:    *const T,
    rstd:    *const T,
    gamma:   *const T,
    beta:    *const T,
    scale:   *mut T,
    bias:    *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> mean_arr(mean, C, N);
      ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
      ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
      ConstEigenVectorArrayMap<T> beta_arr(beta, C);
      EigenArrayMap<T> scale_arr(scale, C, N);
      EigenArrayMap<T> bias_arr(bias, C, N);
      scale_arr = rstd_arr.colwise() * gamma_arr;
      bias_arr = (-scale_arr * mean_arr).colwise() + beta_arr;
    */
}



/**
  | GroupNorm op in Caffe2 for CPU
  | 
  | Written by Kaiming He
  | 
  | Improved by Xiaomeng Yang see https://arxiv.org/abs/1803.08494
  | 
  | This is a stand-alone op: Y = gamma * (X
  | - mu) / sig + beta
  |
  */

crate::ix!();

/**
  | Math:
  | Y = gamma * (X - mu) * rsig + beta
  | let s = gamma * rsig
  | let b = beta - gamma * mu * rsig
  | Y = s * X + b
  | let n = K * HxW
  | dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
  | d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
  | db/dX = -gamma * u * drsig/dX - gamma * rsig * dmu/dX
  | drsig/dX = -rsig^3 * (X - mu) / n
  | dmu/dX = 1 / n
  */
pub fn compute_internal_gradients<T, StorageOrder>(
    N: i32,
    C: i32,
    HxW: i32,
    dY: *const T,
    X:  *const T,
    ds: *mut T,
    db: *mut T) 
{
    todo!("dispatch"); 
    /* */ 
}

#[inline] pub fn compute_internal_gradients_f32_nchw(
    n:    i32,
    c:    i32,
    hxW:  i32,
    dY:   *const f32,
    x:    *const f32,
    ds:   *mut f32,
    db:   *mut f32)  
{
    todo!();
    /*
        ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
      ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int i = 0; i < N * C; ++i) {
        ds[i] = (dY_arr.col(i) * X_arr.col(i)).sum();
        db[i] = dY_arr.col(i).sum();
      }
    */
}

#[inline] pub fn compute_internal_gradients_f32_nhwc(
    n:   i32,
    c:   i32,
    hxW: i32,
    dY:  *const f32,
    x:   *const f32,
    ds:  *mut f32,
    db:  *mut f32)  
{
    todo!();
    /*
        EigenArrayMap<float> ds_arr(ds, C, N);
      EigenArrayMap<float> db_arr(db, C, N);
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<float> dY_arr(dY + i * C * HxW, C, HxW);
        ConstEigenArrayMap<float> X_arr(X + i * C * HxW, C, HxW);
        ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
        db_arr.col(i) = dY_arr.col(0);
        for (int j = 1; j < HxW; ++j) {
          ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
          db_arr.col(i) += dY_arr.col(j);
        }
      }
    */
}

#[inline] pub fn compute_gradient_fused_params<T>(
    n:          i32,
    g:          i32,
    k:          i32,
    hxW:        i32,
    ds:         *const T,
    db:         *const T,
    mu:         *const T,
    rsig:       *const T,
    gamma:      *const T,
    dY_scale:   *mut T,
    x_scale:    *mut T,
    bias:       *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
      ConstEigenArrayMap<T> gamma_arr(gamma, K, G);
      for (int i = 0; i < N; ++i) {
        EigenArrayMap<T>(dY_scale + i * G * K, K, G) =
            gamma_arr.rowwise() * (rsig_arr.col(i).transpose());
      }
      ConstEigenVectorArrayMap<T> mu_arr(mu, N * G);
      ConstEigenVectorArrayMap<T> rsig_vec(rsig, N * G);
      EigenVectorArrayMap<T> X_scale_arr(X_scale, N * G);
      EigenVectorArrayMap<T> bias_arr(bias, N * G);
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> ds_arr(ds + i * G * K, K, G);
        ConstEigenArrayMap<T> db_arr(db + i * G * K, K, G);
        for (int j = 0; j < G; ++j) {
          X_scale_arr(i * G + j) = (ds_arr.col(j) * gamma_arr.col(j)).sum();
          bias_arr(i * G + j) = (db_arr.col(j) * gamma_arr.col(j)).sum();
        }
      }
      const T alpha = T(1) / static_cast<T>(K * HxW);
      X_scale_arr = (bias_arr * mu_arr - X_scale_arr) * rsig_vec.cube() * alpha;
      bias_arr = -X_scale_arr * mu_arr - bias_arr * rsig_vec * alpha;
    */
}

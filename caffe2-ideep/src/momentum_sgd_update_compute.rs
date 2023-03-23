crate::ix!();

#[inline] pub fn momentum_sgd_update(
    n:         i32,
    g:         *const f32,
    m:         *const f32,
    ng:        *mut f32,
    nm:        *mut f32,
    lr:        *const f32,
    momentum:  f32,
    nesterov:  bool,
    param:     *mut f32)  
{
    todo!();
    /*
        const float LR = lr[0];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
      for (auto i = 0; i < N; ++i) {
        if (!nesterov) {
          const float adjusted_gradient = LR * g[i] + momentum * m[i];
          nm[i] = adjusted_gradient;
          ng[i] = adjusted_gradient;
        } else {
          const float mi = m[i];
          const float mi_new = momentum * mi + LR * g[i];
          nm[i] = mi_new;
          ng[i] = (1 + momentum) * mi_new - momentum * mi;
        }

        if (param) {
          param[i] -= ng[i];
        }
      }
    */
}

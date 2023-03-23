crate::ix!();

#[inline] pub fn adam_ideep_compute_output_grad(
    n:           i32,
    w:           *const f32,
    g:           *const f32,
    m:           *const f32,
    v:           *const f32,
    nw:          *mut f32,
    nm:          *mut f32,
    nv:          *mut f32,
    ng:          *mut f32,
    beta1:       f32,
    beta2:       f32,
    eps_hat:     f32,
    correction:  f32,
    lr:          *const f32)  
{
    todo!();
    /*
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
      for (auto i = 0; i < N; ++i) {
        float gi = g[i];
        float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        float ngi = ng[i] = correction * mi / (std::sqrt(vi) + eps_hat);
        nw[i] = w[i] + lr[0] * ngi;
      }
    */
}

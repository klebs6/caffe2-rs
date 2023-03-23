crate::ix!();

#[inline] pub unsafe fn adam_ideep_update(
    n:            i32,
    g:            &[f32],
    m:            &[f32],
    v:            &[f32],
    ng:           &mut [f32],
    nm:           &mut [f32],
    nv:           &mut [f32],
    beta1:        f32,
    beta2:        f32,
    eps_hat:      f32,
    correction:   f32,
    lr:           &[f32])  
{
    //can parallel for
    for i in 0..n {

        let i = i as usize;

        let gi = g[i];
        let mi = m[i] * beta1 + gi * (1.0 - beta1);
        nm[i]  = mi;
        nv[i]  = v[i] * beta2 + gi * gi * (1.0 - beta2);
        let vi = nv[i];
        ng[i]  = lr[0] * correction * mi / (vi.sqrt() + eps_hat);
    }
}

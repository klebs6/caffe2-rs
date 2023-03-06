crate::ix!();

/**
  | Repeat masking along continuous segments
  | (right axes) of size D
  |
  */
#[inline] pub fn repeated_mask_with_functor<Functor>(
    n:        i32,
    m:        i32,
    d:        i32,
    input:    *const f32,
    func:     Functor,
    fill_val: f32,
    out:      *mut f32)
{
    todo!();
    /*
        for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          for (int k = 0; k < D; ++k) {
            auto val = in[M * D * i + D * j + k];
            out[M * D * i + D * j + k] = (fn(i, j, val) ? fill_val : val);
          }
        }
      }
    */
}

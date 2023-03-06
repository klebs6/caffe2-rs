crate::ix!();

pub const minf: f32 = -1.0 * f32::INFINITY;

/**
  | Template this on a functor object so
  | we can generate different implementations
  | at compile time and have a better chance
  | of inlining
  |
  */
#[inline] pub fn mask_with_functor<Functor>(
    n:        i32,
    m:        i32,
    b:        i32,
    input:    *const f32,
    func:     Functor,
    fill_val: f32,
    out:      *mut f32) 
{
    todo!();
    /*
       if (B >= 0) { // with batching
        // collapse tensor to 3-dim view [B, N, M] where:
        // B is product of dims up to and including batch
        // N is product of dims between batch and axis, exclusive
        // M is product of dimensions at/after axis
        // then mask each batch [i, :, :] (note that this is N x M matrix)
        for (int i = 0; i < B; ++i) {
          for (int j = 0; j < N; ++j) {
            for (int k = 0; k < M; ++k) {
              // when [i, :, :] is laid out in row major order
              // N * M * i + M * j + k is index of entry in N x M matrix
              // with coordinates (row = j, col = k)
              auto val = in[N * M * i + M * j + k];
              out[N * M * i + M * j + k] = (fn(j, k, val) ? fill_val : val);
            }
          }
        }
      } else { // without batching
        // TODO(T20952436): vector implementation
        // collapse tensor to 2-dim view [N, M], where
        // N is product of dimensions before axis
        // M is product of dimensions at/after axis
        // and mask N by M matrix
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            auto val = in[M * i + j];
            out[M * i + j] = (fn(i, j, val) ? fill_val : val);
          }
        }
      }
    */
}

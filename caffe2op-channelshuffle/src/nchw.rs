crate::ix!();

#[inline] pub fn run_channel_shuffleNCHW<T>(
    n:         i32,
    g:         i32,
    k:         i32,
    hxW:       i32,
    x:         *const T,
    y:         *mut T,
    context:   *mut CPUContext) 
{
    todo!();
    /*
        const int stride = G * K * HxW;
      for (int i = 0; i < N; ++i) {
        if (G < K) {
          for (int j = 0; j < G; ++j) {
            math::CopyMatrix<T, CPUContext>(
                K, HxW, X + j * K * HxW, HxW, Y + j * HxW, G * HxW, context);
          }
        } else {
          for (int j = 0; j < K; ++j) {
            math::CopyMatrix<T, CPUContext>(
                G, HxW, X + j * HxW, K * HxW, Y + j * G * HxW, HxW, context);
          }
        }
        X += stride;
        Y += stride;
      }
    */
}

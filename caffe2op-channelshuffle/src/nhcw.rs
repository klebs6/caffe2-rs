crate::ix!();

#[inline] pub fn run_channel_shuffleNHWC<T>(
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
        const std::array<std::int64_t, 2> dims = {G, K};
      const std::array<std::int32_t, 2> axes = {1, 0};
      const int M = N * HxW;
      const int C = G * K;
      for (int i = 0; i < M; ++i) {
        math::Transpose<std::int64_t, T, CPUContext>(
            2, dims.data(), axes.data(), X, Y, context);
        X += C;
        Y += C;
      }
    */
}

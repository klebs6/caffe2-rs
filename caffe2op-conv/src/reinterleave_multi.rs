crate::ix!();

#[inline] pub fn reinterleave_multithreaded<const N: i32, T, Context>(
    y0:         *const T,
    bias_data:  *const T,
    y:          *mut T,
    outputC:    i32,
    output_h:    i32,
    output_w:    i32,
    inputW:     i32,
    kernel_w:    i32,
    stride_w:    i32,
    adjH:       i32,
    pool:       *mut ThreadPool) 
{
    todo!();
    /*
        // # channels times height
      size_t totalTiles = (size_t)outputC * output_h;
      FixedDivisor<int> divOutputH(output_h);

    #define REINTERLEAVE(N)  \
      do {                   \
        reinterleaveRows<N>( \
            y0,              \
            bias_data,       \
            c,               \
            h,               \
            y,               \
            outputC,         \
            output_h,         \
            output_w,         \
            inputW,          \
            kernel_w,         \
            stride_w,         \
            adjH);           \
      } while (false)

      std::function<void(int, size_t)> fnReinterleave = [&](int threadId,
                                                            size_t tile_id) {
        int h;
        int c;
        divOutputH.DivMod((int)tile_id, &c, &h);

        REINTERLEAVE(N);
      };

    #undef REINTERLEAVE

      pool->run(fnReinterleave, totalTiles);
    */
}

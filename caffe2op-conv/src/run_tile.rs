crate::ix!();

#[inline] pub fn run_tile_contiguous<T, Context>(
    tile_id:           i32,
    n:                 i32,
    m:                 i32,
    h:                 i32,
    w:                 i32,
    output_h:          i32,
    output_w:          i32,
    c:                 i32,
    kernel_h:          i32,
    kernel_w:          i32,
    stride_h:          i32,
    stride_w:          i32,
    padT:              i32,
    filter_data:       *const T,
    xdata:             *const T,
    col_buffer_data:   *mut T,
    ydata:             *mut T,
    context:           *mut Context) 
{
    /**
      | The tile size is exactly the length of
      | a single row
      |
      */
      let tile_size: i32 = w;

      let kernel_data_size = c * kernel_h * kernel_w;
      let current_tile_start = tile_size * tile_id;

    todo!();
    /*
      // gemm tile
      math::GemmEx<T, Context>(
          CblasTrans,
          CblasNoTrans,
          kernel_data_size,
          tile_size,
          M,
          1,
          filterData,
          kernel_data_size,
          Xdata + current_tile_start,
          H * W,
          0,
          colBufferData,
          tile_size,
          context);

      // col2im tile
      // We assume that there is no padding in the columns (padL and padR
      // == 0).
      // FIXME: it is actually possible for us to handle padding, figure
      // out how to adjust the bounds

      // We write into Y in a de-interleaved fashion; in other words,
      // every column (mod stride_w) == 0 together in one block,
      // every column (mod stride_w) == 1 in another,
      // ... and so on.
      int colBlockSize = (W + kernel_w / stride_w);
      int numColBlocks = stride_w;

      for (int c = 0; c < kernel_data_size; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        // Each row is a separate tile that we handle. First determine the
        // row into which we are writing the output.
        // We can properly handle padding for the rows.
        int rowY = tile_id * stride_h - padT + h_offset;

        // If this row is out of bounds, then skip it
        if (!math::utils::IsAGeZeroAndALtB(rowY, output_h)) {
          continue;
        }

        // FIXME: we don't actually handle a dynamic padL > 0
        constexpr int kPadL = 0;
        int colOffsetStart = -kPadL + w_offset;
        int colBlockY = colOffsetStart % stride_w;

        // However, within a block we may not start writing at offset
        // 0. The offset at which we begin writing is determined by
        // colOffsetStart
        int colWithinBlockOffsetY = colOffsetStart / stride_w;

        // So, this is where we begin reading/writing in Y
        int colY = colBlockY * colBlockSize + colWithinBlockOffsetY;

        // This is the complete offset into Y from the start
        // Each row has stride_w blocks of size colBlockSize
        int offsetY = rowY * colBlockSize * numColBlocks + colY;

        T* colBufferPointer = colBufferData + c * tile_size;
        T* yPointer =
            Ydata + c_im * output_h * (colBlockSize * numColBlocks) + offsetY;

        int b = 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
        // We vectorize the loop within the row
        {
          constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float)) * 4;
          int limit = (tile_size / kUnroll) * kUnroll;

          for (; b < limit; b += kUnroll) {
            float32x4_t cb0 = vld1q_f32(colBufferPointer + 0);
            float32x4_t cb1 = vld1q_f32(colBufferPointer + 4);
            float32x4_t cb2 = vld1q_f32(colBufferPointer + 8);
            float32x4_t cb3 = vld1q_f32(colBufferPointer + 12);

            float32x4_t y0 = vld1q_f32(yPointer + 0);
            float32x4_t y1 = vld1q_f32(yPointer + 4);
            float32x4_t y2 = vld1q_f32(yPointer + 8);
            float32x4_t y3 = vld1q_f32(yPointer + 12);

            y0 = vaddq_f32(y0, cb0);
            y1 = vaddq_f32(y1, cb1);
            y2 = vaddq_f32(y2, cb2);
            y3 = vaddq_f32(y3, cb3);

            vst1q_f32(yPointer + 0, y0);
            vst1q_f32(yPointer + 4, y1);
            vst1q_f32(yPointer + 8, y2);
            vst1q_f32(yPointer + 12, y3);

            colBufferPointer += kUnroll;
            yPointer += kUnroll;
          }
        }

        {
          constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float));
          int limit = (tile_size / kUnroll) * kUnroll;

          for (; b < limit; b += kUnroll) {
            float32x4_t cb0 = vld1q_f32(colBufferPointer);
            float32x4_t y0 = vld1q_f32(yPointer);

            y0 = vaddq_f32(y0, cb0);

            vst1q_f32(yPointer, y0);

            colBufferPointer += kUnroll;
            yPointer += kUnroll;
          }
        }
    #endif

        // Handle un-vectorizable epilogue
        for (; b < tile_size; ++b) {
          *yPointer += *colBufferPointer;
          ++yPointer;
          ++colBufferPointer;
        }
      }
    */
}

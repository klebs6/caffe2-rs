crate::ix!();

/// Vectorizes 4x4p0s0 average pooling for ARM NEON
#[cfg(target_feature = "neon")]
#[inline] pub fn avg_pool_neon4x4p0s_0plane(
    inputH: i32,
    inputW: i32,
    input:  *const f32,
    output: *mut f32)  {

    todo!();
    /*
        constexpr int kKernelHeight = 4;
      constexpr int kKernelWidth = 4;
      constexpr float kDiv = (1.0f / ((float)kKernelHeight * (float)kKernelWidth));

      // Handle portion that can be unrolled by 4
      constexpr int kUnroll = 4;
      constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
      constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

      if (inputW % kLoadCols == 0) {
        //
        // Manually unroll by 4 (kUnroll)
        //

        for (int h = 0; h < inputH; h += kKernelHeight) {
          float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
          const float* curInput = input + h * inputW;

          for (int w = 0; w < inputW; w += kLoadCols) {
            float32x4_t out = {};

            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
              float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
              float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
              out = vsetq_lane_f32(v0, out, 0);
            }
            curInput += kLoadSizeFloat;

            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
              float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
              float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
              out = vsetq_lane_f32(v0, out, 1);
            }
            curInput += kLoadSizeFloat;

            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
              float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
              float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
              out = vsetq_lane_f32(v0, out, 2);
            }
            curInput += kLoadSizeFloat;

            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
              float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
              float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
              out = vsetq_lane_f32(v0, out, 3);
            }
            curInput += kLoadSizeFloat;

            out = vmulq_f32(out, vdupq_n_f32(kDiv));
            vst1q_f32_aligned(&outputRow[w / kKernelWidth], out);
          }
        }
      } else {
        //
        // Not unrolled
        //

        for (int h = 0; h < inputH; h += kKernelHeight) {
          const float* inputRow = input + h * inputW;
          float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

          for (int w = 0; w < inputW; w += kKernelWidth) {
            const float* curInput = inputRow + w;

            float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
            float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
            float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
            float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
            float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3) * kDiv;
            outputRow[w / kKernelWidth] = v0;
          }
        }
      }
    */
}

#[inline] pub fn run_neon_average_pool4x4p0s0NCHW(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    x: *const f32,
    y: *mut f32)  {

    todo!();
    /*
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      const int X_stride = H * W;
      const int Y_stride = (H / 4) * (W / 4);
      const float* X_ptr = X;
      float* Y_ptr = Y;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
          AvgPoolNeon4x4p0s0Plane(H, W, X_ptr, Y_ptr);
          X_ptr += X_stride;
          Y_ptr += Y_stride;
        }
      }
    #else
      (void)N;
      (void)C;
      (void)H;
      (void)W;
      (void)X;
      (void)Y;
    #endif
    */
}

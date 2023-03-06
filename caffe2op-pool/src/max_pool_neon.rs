crate::ix!();

/// Vectorizes 2x2p0s0 average pooling for ARM NEON
#[cfg(target_feature = "neon")]
#[inline] pub fn max_pool_neon2x2p0s_0plane(
    inputH: i32,
    inputW: i32,
    input:  *const f32,
    output: *mut f32)  {

    todo!();
    /*
        constexpr int kKernelHeight = 2;
      constexpr int kKernelWidth = 2;

      // Handle portion that can be unrolled by 4
      constexpr int kUnroll = 4;
      constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
      constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

      if (inputW % kLoadCols == 0) {
        for (int h = 0; h < inputH; h += kKernelHeight) {
          float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
          const float* curInput = input + h * inputW;

          for (int w = 0; w < inputW; w += kLoadCols) {
            float32x2_t hmax_0, hmax_1, hmax_2, hmax_3;
            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
              hmax_0 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
            }
            curInput += kLoadSizeFloat;
            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
              hmax_1 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
            }
            curInput += kLoadSizeFloat;
            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
              hmax_2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
            }
            curInput += kLoadSizeFloat;
            {
              float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
              float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
              float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
              hmax_3 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
            }
            curInput += kLoadSizeFloat;

            float32x4_t out_0 = vcombine_f32(hmax_0, hmax_1);
            float32x4_t out_1 = vcombine_f32(hmax_2, hmax_3);
            vst1q_f32_aligned(&outputRow[w / kKernelWidth + 0], out_0);
            vst1q_f32_aligned(&outputRow[w / kKernelWidth + 4], out_1);
          }
        }
      } else {
        // Not unrolled
        for (int h = 0; h < inputH; h += kKernelHeight) {
          const float* inputRow = input + h * inputW;
          float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

          for (int w = 0; w < inputW; w += kKernelWidth * 2) {
            const float* curInput = inputRow + w;
            float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
            float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
            float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
            float32x2_t hmax = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
            vst1_f32(&outputRow[w / kKernelWidth], hmax);
          }
        }
      }
    */
}

#[inline] pub fn run_neon_max_pool2x2p0s0NCHW(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    x: *const f32,
    y: *mut f32)  
{
    todo!();
    /*
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      const int X_stride = H * W;
      const int Y_stride = (H / 2) * (W / 2);
      const float* X_ptr = X;
      float* Y_ptr = Y;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
          MaxPoolNeon2x2p0s0Plane(H, W, X_ptr, Y_ptr);
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

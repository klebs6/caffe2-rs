crate::ix!();

#[cfg(target_feature = "neon")]
#[inline] pub fn run_neon_prelu(
    out:   *mut f32,
    input: *const f32,
    size:  i32,
    w:     f32)  {
    
    todo!();
    /*
        float32x4_t vZero = vdupq_n_f32(0.0f);
      float32x4_t vW = vdupq_n_f32(w);

      constexpr int kVecSizeInFloat = sizeof(float32x4_t) / sizeof(float);

      if (size < kVecSizeInFloat) {
        for (int i = 0; i < size; ++i) {
          float v = in[i];
          out[i] = v > 0 ? v : v * w;
        }

        return;
      }

      // We want to load aligned from the input, but assume the output is unaligned
      int prologue = kVecSizeInFloat -
          // remainder in floats
          (((uintptr_t)in) % (sizeof(float32x4_t))) / sizeof(float);

      int i = 0;

      // Prologue loop
      for (; i < prologue; ++i) {
        float v = in[i];
        out[i] = v > 0 ? v : v * w;
      }

      // The loop is manually unrolled by 6; seems to be the limit for
      // armv7 to avoid register spills
      constexpr int kUnroll = 6;
      constexpr int kFloatsPerLoop = kUnroll * kVecSizeInFloat;

      int remainder = size - prologue;
      int vectorizable = prologue + (remainder / kFloatsPerLoop) * kFloatsPerLoop;

      for (; i < vectorizable; i += kFloatsPerLoop) {
        float32x4_t v0 = vld1q_f32_aligned(in + i + 0);
        float32x4_t v1 = vld1q_f32_aligned(in + i + 4);
        float32x4_t v2 = vld1q_f32_aligned(in + i + 8);
        float32x4_t v3 = vld1q_f32_aligned(in + i + 12);
        float32x4_t v4 = vld1q_f32_aligned(in + i + 16);
        float32x4_t v5 = vld1q_f32_aligned(in + i + 20);

        uint32x4_t gz0 = vcgtq_f32(v0, vZero);
        uint32x4_t gz1 = vcgtq_f32(v1, vZero);
        uint32x4_t gz2 = vcgtq_f32(v2, vZero);
        uint32x4_t gz3 = vcgtq_f32(v3, vZero);
        uint32x4_t gz4 = vcgtq_f32(v4, vZero);
        uint32x4_t gz5 = vcgtq_f32(v5, vZero);

        float32x4_t v0neg = vmulq_f32(v0, vW);
        float32x4_t v1neg = vmulq_f32(v1, vW);
        float32x4_t v2neg = vmulq_f32(v2, vW);
        float32x4_t v3neg = vmulq_f32(v3, vW);
        float32x4_t v4neg = vmulq_f32(v4, vW);
        float32x4_t v5neg = vmulq_f32(v5, vW);

        // v0 > 0 ? v0 : v0 * w
        v0 = vbslq_f32(gz0, v0, v0neg);
        v1 = vbslq_f32(gz1, v1, v1neg);
        v2 = vbslq_f32(gz2, v2, v2neg);
        v3 = vbslq_f32(gz3, v3, v3neg);
        v4 = vbslq_f32(gz4, v4, v4neg);
        v5 = vbslq_f32(gz5, v5, v5neg);

        vst1q_f32(out + i + 0, v0);
        vst1q_f32(out + i + 4, v1);
        vst1q_f32(out + i + 8, v2);
        vst1q_f32(out + i + 12, v3);
        vst1q_f32(out + i + 16, v4);
        vst1q_f32(out + i + 20, v5);
      }

      for (; i < size; ++i) {
        float v = in[i];
        out[i] = v > 0 ? v : v * w;
      }
    */
}

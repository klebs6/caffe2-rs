crate::ix!();

#[inline] pub fn reinterleave_rows<const kStrideW: i32>(
    src:         *const f32,
    bias:        *const f32,
    c:           i32,
    h:           i32,
    dst:         *mut f32,
    outputC:     i32,
    output_h:    i32,
    output_w:    i32,
    inputW:      i32,
    kernel_w:    i32,
    stride_w:    i32,
    adjH:        i32) 
{
    todo!();
    /*
        // Each row in src is of the form:
      // [w mod stride_w == 0 elements]...[w mod stride_w == stride_w - 1
      // elements]
      // We need to re-interleave the values and write them in the output
      int colBlockSize = inputW + kernel_w / kStrideW;
      int noAdjOutputW = (inputW - 1) * kStrideW + kernel_w;

      int point = c * output_h + h;
      src += point * colBlockSize * kStrideW;
      dst += point * output_w;

      float b = bias ? bias[c] : 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      float32x4_t biasV = vdupq_n_f32(b);
    #endif

      int w = 0;
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float)) * 2;
      int limit = ((inputW - 1) / kUnroll) * kUnroll;

      for (; w < limit; w += kUnroll) {
        // We need to interleave in terms of kStrideW units
        float32x4_t v0[kStrideW];
        float32x4_t v1[kStrideW];

        for (int i = 0; i < kStrideW; ++i) {
          v0[i] = vld1q_f32(src + i * colBlockSize);
          v1[i] = vld1q_f32(src + i * colBlockSize + 4);
        }

        // add per-channel bias
        for (int i = 0; i < kStrideW; ++i) {
          v0[i] = vaddq_f32(v0[i], biasV);
          v1[i] = vaddq_f32(v1[i], biasV);
        }

        // Write interleaved into the output
        StoreInterleaved<float, kStrideW>::store(dst + 0 * kStrideW, v0);
        StoreInterleaved<float, kStrideW>::store(dst + 4 * kStrideW, v1);

        src += kUnroll;
        dst += kUnroll * kStrideW;
      }
    #endif

      // Handle non-vectorizable remainder
      for (; w < inputW - 1; ++w) {
        float v[kStrideW];

        for (int i = 0; i < kStrideW; ++i) {
          v[i] = src[i * colBlockSize];
        }

        // add per-channel bias
        for (int i = 0; i < kStrideW; ++i) {
          v[i] += b;
        }

        // Write interleaved into the output
        StoreInterleaved<float, kStrideW>::store(dst, v);

        src += 1;
        dst += kStrideW;
      }

      // We have handled 0 .. (inputW - 1) * stride inclusive so far.
      // Handle the remainder
      int outputPoint = (inputW - 1) * kStrideW;
      int block = 0;

      // Output width may include adjustment into which we don't
      // write; ignore it
      while (outputPoint < noAdjOutputW) {
        float v = src[block * colBlockSize];
        dst[0] = v + b;
        ++outputPoint;
        dst += 1;

        ++block;
        if (block >= kStrideW) {
          block = 0;
          src += 1;
        }
      }

      // Remainder of the buffer comprised of just the `adj` must have
      // bias added
      for (; outputPoint < output_w; ++outputPoint) {
        dst[0] = b;
        dst += 1;
      }
    */
}


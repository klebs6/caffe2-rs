crate::ix!();

//use to be struct SumMultiple with fn sumInto
#[cfg(target_feature = "neon")]
pub fn sum_multiple<const N: i32>(acc: *mut f32, to_sum: *mut *mut f32, size: libc::size_t) {
    match N {
        1 => sum_multiple1(acc, to_sum, size),
        2 => sum_multiple2(acc, to_sum, size),
        3 => sum_multiple3(acc, to_sum, size),
    }
}

#[cfg(target_feature = "neon")]
pub fn sum_multiple1(acc: *mut f32, to_sum: *mut *mut f32, size: libc::size_t) {

    todo!();

    /*
    constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float));
    int limit = (size / kUnroll) * kUnroll;

    auto toSum0 = toSum[0];

    size_t i = 0;
    for (; i < limit; i += kUnroll) {
      float32x4_t v0 = vld1q_f32_aligned(acc + i);
      float32x4_t v1 = vld1q_f32_aligned(toSum0 + i);

      v0 = vaddq_f32(v0, v1);

      vst1q_f32_aligned(acc + i, v0);
    }

    for (; i < size; ++i) {
      float v0 = acc[i];
      float v1 = toSum0[i];

      v0 += v1;

      acc[i] = v0;
    }
    */
}

#[cfg(target_feature = "neon")]
pub fn sum_multiple2(acc: *mut f32, to_sum: *mut *mut f32, size: libc::size_t) {

    todo!();
    /*
    constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float));
    int limit = (size / kUnroll) * kUnroll;

    auto toSum0 = toSum[0];
    auto toSum1 = toSum[1];

    size_t i = 0;
    for (; i < limit; i += kUnroll) {
      float32x4_t v0 = vld1q_f32_aligned(acc + i);
      float32x4_t v1 = vld1q_f32_aligned(toSum0 + i);
      float32x4_t v2 = vld1q_f32_aligned(toSum1 + i);

      v0 = vaddq_f32(v0, v1);
      v0 = vaddq_f32(v0, v2);

      vst1q_f32_aligned(acc + i, v0);
    }

    for (; i < size; ++i) {
      float v0 = acc[i];
      float v1 = toSum0[i];
      float v2 = toSum1[i];

      v0 += v1;
      v0 += v2;

      acc[i] = v0;
    }
    */
}

#[cfg(target_feature = "neon")]
pub fn sum_multiple3(acc: *mut f32, to_sum: *mut *mut f32, size: libc::size_t) {

    todo!();
    /*
    constexpr int kUnroll = (sizeof(float32x4_t) / sizeof(float));
    int limit = (size / kUnroll) * kUnroll;

    auto toSum0 = toSum[0];
    auto toSum1 = toSum[1];
    auto toSum2 = toSum[2];

    size_t i = 0;
    for (; i < limit; i += kUnroll) {
      float32x4_t v0 = vld1q_f32_aligned(acc + i);
      float32x4_t v1 = vld1q_f32_aligned(toSum0 + i);
      float32x4_t v2 = vld1q_f32_aligned(toSum1 + i);
      float32x4_t v3 = vld1q_f32_aligned(toSum2 + i);

      v0 = vaddq_f32(v0, v1);
      v2 = vaddq_f32(v2, v3);
      v0 = vaddq_f32(v0, v2);

      vst1q_f32_aligned(acc + i, v0);
    }

    for (; i < size; ++i) {
      float v0 = acc[i];
      float v1 = toSum0[i];
      float v2 = toSum1[i];
      float v3 = toSum2[i];

      v0 += v1;
      v2 += v3;
      v0 += v2;

      acc[i] = v0;
    }
    */
}

/// Performs acc[i] += sum_j toSum_j[i] pointwise
pub fn sum_into(acc: *mut f32, to_sum: &Vec<*mut f32>, size: libc::size_t) {

    todo!();
    /*
    #if defined(__ARM_NEON__) || defined(__ARM_NEON)
      if (toSum.size() == 1) {
        SumMultiple<1>::sumInto(acc, toSum.data(), size);
        return;
      } else if (toSum.size() == 2) {
        SumMultiple<2>::sumInto(acc, toSum.data(), size);
        return;
      } else if (toSum.size() == 3) {
        SumMultiple<3>::sumInto(acc, toSum.data(), size);
        return;
      }
    #endif

      // Otherwise, use fallback implementation
      EigenVectorArrayMap<float> accT(acc, size);

      for (auto p : toSum) {
        accT += ConstEigenVectorArrayMap<float>(p, size);
      }
    */
}

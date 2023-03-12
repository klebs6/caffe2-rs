crate::ix!();

#[inline] pub fn is_little_endian() -> bool {
    
    todo!();
    /*
        constexpr std::int32_t kValue = 1;
      return reinterpret_cast<const std::uint8_t*>(&kValue)[0] == 1;
    */
}

pub type ConvertFnType<T> = fn(
    dst: *mut f32, 
    src: *const T, 
    N:   libc::size_t) -> ();


#[inline] pub fn compress_uniform_simplified_(
    x:         *const f32,
    n:         i32,
    xmin:      f32,
    xmax:      f32,
    xq:        *mut f32,
    bit_rate:  i32) -> f32 
{
    todo!();
    /*
        xmin = static_cast<at::Half>(xmin);
      float data_range = xmax - xmin;
      float qmax = (1 << bit_rate) - 1;
      float scale = data_range == 0
          ? 1.0
          : static_cast<float>(static_cast<at::Half>(data_range / qmax));
      float inverse_scale = 1.0f / scale;

      float norm = 0.0f;
      constexpr int VLEN = 8;
      int i = 0;

    #ifdef __AVX__
      // vectorized loop
      __m256 norm_v = _mm256_setzero_ps();
      for (; i < N / VLEN * VLEN; i += VLEN) {
        __m256 X_v = _mm256_loadu_ps(X + i);
        // Affine
        __m256 Xq_v = _mm256_mul_ps(
            _mm256_sub_ps(X_v, _mm256_set1_ps(xmin)),
            _mm256_set1_ps(inverse_scale));
        // Round
        // Use _MM_FROUND_CUR_DIRECTION to match the behavior with the remainder
        // code. In most cases, the rounding mode is round-to-nearest-even.
        Xq_v = _mm256_round_ps(Xq_v, _MM_FROUND_CUR_DIRECTION);
        // Clip
        Xq_v = _mm256_max_ps(
            _mm256_setzero_ps(), _mm256_min_ps(Xq_v, _mm256_set1_ps(qmax)));
        // Inverse affine
        Xq_v = _mm256_add_ps(
            _mm256_mul_ps(Xq_v, _mm256_set1_ps(scale)), _mm256_set1_ps(xmin));
        __m256 err_v = _mm256_sub_ps(X_v, Xq_v);
        norm_v = _mm256_add_ps(_mm256_mul_ps(err_v, err_v), norm_v);
      }
      alignas(64) float temp[VLEN];
      _mm256_store_ps(temp, norm_v);
      for (int j = 0; j < VLEN; ++j) {
        norm += temp[j];
      }
    #endif // __AVX__

      // remainder loop
      for (; i < N; i++) {
        Xq[i] = std::max(
            0.0f, std::min<float>(nearbyint((X[i] - xmin) * inverse_scale), qmax));
        Xq[i] = Xq[i] * scale + xmin;
        norm += (X[i] - Xq[i]) * (X[i] - Xq[i]);
      }

      return std::sqrt(norm);
    */
}

#[inline] pub fn convertfp_32fp32(dst: *mut f32, src: *const f32, n: usize)  {
    
    todo!();
    /*
        memcpy(dst, src, sizeof(float) * N);
    */
}

#[inline] pub fn convertfp_16fp32(dst: *mut f32, src: *const f16, n: usize)  {
    
    todo!();
    /*
        for (size_t i = 0; i < N; i++) {
        dst[i] = src[i];
      }
    */
}

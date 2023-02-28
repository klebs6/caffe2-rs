crate::ix!();

#[inline] pub fn reluavx2<T>(
    n:          i32,
    zero_point: i32,
    x:          *const T,
    y:          *mut T)  {

    todo!();
    /*
    
    */
}

#[inline] pub fn reluavx2_u8(
    n:          i32,
    zero_point: i32,
    x:          *const u8,
    y:          *mut u8)  {
    
    todo!();
    /*
        constexpr int kVLen = 32;
      const int n = N / kVLen * kVLen;
      const int r = N % kVLen;
      const __m256i zero_v = _mm256_set1_epi8(static_cast<uint8_t>(zero_point));
      for (int i = 0; i < n; i += kVLen) {
        __m256i cur_v = _mm256_max_epu8(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + i)), zero_v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(Y + i), cur_v);
      }
      for (int i = 0; i < r; ++i) {
        Y[n + i] = std::max(X[n + i], static_cast<uint8_t>(zero_point));
      }
    */
}

#[inline] pub fn reluavx2_u16(
    n:          i32,
    zero_point: i32,
    x:          *const u16,
    y:          *mut u16)  {
    
    todo!();
    /*
        constexpr int kVLen = 16;
      const int n = N / kVLen * kVLen;
      const int r = N % kVLen;
      const __m256i zero_v = _mm256_set1_epi16(static_cast<uint16_t>(zero_point));
      for (int i = 0; i < n; i += kVLen) {
        __m256i cur_v = _mm256_max_epu16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + i)), zero_v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(Y + i), cur_v);
      }
      for (int i = 0; i < r; ++i) {
        Y[n + i] = std::max(X[n + i], static_cast<uint16_t>(zero_point));
      }
    */
}

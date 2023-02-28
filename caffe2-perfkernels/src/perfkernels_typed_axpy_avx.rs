crate::ix!();

#[inline] pub fn typed_axpy__avx_f_16c(
    n: i32,
    a: f32,
    x: *const f32,
    y: *mut f32)  
{
    todo!();
    /*
        int current = 0;
      const int bound = (N % 8) ? N - 8 : N;
      __m256 mma = _mm256_set1_ps(a);
      for (; current < bound; current += 8) {
        _mm256_storeu_ps(
            y + current,
            _mm256_add_ps(
                _mm256_mul_ps(mma, _mm256_loadu_ps(x + current)),
                _mm256_loadu_ps(y + current)));
      }

      if (bound != N) {
        while (current < N) {
          y[current] += x[current] * a;
          ++current;
        }
      }
    */
}

#[inline] pub fn typed_axpy_halffloat__avx_f_16c(
    n: i32,
    a: f32,
    x: *const f16,
    y: *mut f32)
{
    todo!();
    /*
        // if x does not start at the 16 byte boundary, we will process the first few.
      // before we get to a real one.
      while ((reinterpret_cast<unsigned long>(x) % 16) && N) {
        *(y++) += _cvtsh_ss((*(x++)).x) * a;
        --N;
      }

      // From now on we can do vectorized additions using __m256, which is 8 floats,
      // so we will vectorize every 8 element and then resort to cvtsh_ss.
      __m256 mma = _mm256_set1_ps(a);
      int current = 0;
      const int bound = (N % 8) ? N - 8 : N;

      for (; current < bound; current += 8) {
        __m128i mmx_16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + current));
        __m256 mmx_32 = _mm256_cvtph_ps(mmx_16);
        __m256 mmy_in = _mm256_loadu_ps(y + current);
        __m256 mmmul = _mm256_mul_ps(mmx_32, mma);
        __m256 mmy_out = _mm256_add_ps(mmmul, mmy_in);
        _mm256_storeu_ps(y + current, mmy_out);
      }

      if (bound != N) {
        while (current < N) {
          y[current] += _cvtsh_ss(x[current].x) * a;
          ++current;
        }
      }
    */
}

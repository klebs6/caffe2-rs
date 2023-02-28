crate::ix!();

/// version without prefetching
#[inline] pub fn adagrad_update__avx2_fma(
    n:              i32,
    w:              *const f32,
    g:              *const f32,
    h:              *const f32,
    nw:             *mut f32,
    nh:             *mut f32,
    epsilon:        f32,
    decay:          f32,
    lr:             f32,
    weight_decay:   Option<f32>)  
{
    let weight_decay: f32 = weight_decay.unwrap_or(0.0);

    todo!();
    /*
        constexpr size_t kSize = 8;
      auto i = 0;
      for (; i + kSize <= N; i += kSize) {
        __m256 gi = _mm256_loadu_ps(g + i);
        __m256 hi = _mm256_loadu_ps(h + i);
        __m256 wi = _mm256_loadu_ps(w + i);
        gi = _mm256_fmadd_ps(_mm256_set1_ps(weight_decay), wi, gi);

        __m256 nhi = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(decay), hi), _mm256_mul_ps(gi, gi));
        _mm256_storeu_ps(nh + i, nhi);
        __m256 vtmp = _mm256_div_ps(
            _mm256_mul_ps(_mm256_set1_ps(lr), gi),
            _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
        _mm256_storeu_ps(nw + i, _mm256_add_ps(wi, vtmp));
      }

      for (; i < N; ++i) {
        float gi = std::fma(weight_decay, w[i], g[i]);
        float hi = nh[i] = decay * h[i] + gi * gi;
        nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
      }
    */
}

//{w_n, h_n, nw_n, nh_n} prefetch ptr
#[inline] pub fn adagrad_update_prefetch__avx2_fma(
    n:             i32,
    w:             *const f32,
    w_n:           *const f32,
    g:             *const f32,
    h:             *const f32,
    h_n:           *const f32,
    nw:            *mut f32,
    nw_n:          *mut f32,
    nh:            *mut f32,
    nh_n:          *mut f32,
    epsilon:       f32,
    lr:            f32,
    weight_decay:  Option<f32>)  
{
    let weight_decay: f32 = weight_decay.unwrap_or(0.0);

    todo!();
    /*
        internal::adagrad_update_prefetch_inlined(
          N, w, w_n, g, h, h_n, nw, nw_n, nh, nh_n, epsilon, lr, weight_decay);
    */
}

/**
  | Compute adagrad sparse, assumes embedding and
  | momentum are at::Half
  |
  | {w_n, h_n, nw_n, nh_n} prefetch ptr
  */
#[inline] pub fn adagrad_fp16_update_prefetch__avx2_fma(
    n:             i32,
    w:             *const f16,
    w_n:           *const f16,
    g:             *const f32,
    h:             *const f16,
    h_n:           *const f16,
    nw:            *mut f16,
    nw_n:          *mut f16,
    nh:            *mut f16,
    nh_n:          *mut f16,
    epsilon:       f32,
    lr:            f32,
    weight_decay:  Option<f32>)  
{
    let weight_decay: f32 = weight_decay.unwrap_or(0.0);

    todo!();
    /*
        constexpr int kSize = 8;
      auto i = 0;
      for (; i + kSize <= N; i += kSize) {
        _mm_prefetch(reinterpret_cast<const char*>(&w_n[i]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&h_n[i]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&nw_n[i]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&nh_n[i]), _MM_HINT_T0);

        // only convert momentum and embedding, gradient is fp32
        __m256 gi = _mm256_loadu_ps(g + i);
        __m128i hhi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(h + i));
        __m256 hi = _mm256_cvtph_ps(hhi);
        __m128i whi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        __m256 wi = _mm256_cvtph_ps(whi);
        gi = _mm256_fmadd_ps(_mm256_set1_ps(weight_decay), wi, gi);

        __m256 nhi = _mm256_add_ps(hi, _mm256_mul_ps(gi, gi));
        __m128i nhhi = _mm256_cvtps_ph(nhi, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(nh + i), nhhi);

        __m256 vtmp = _mm256_div_ps(
            _mm256_mul_ps(_mm256_set1_ps(lr), gi),
            _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
        __m256 nwi = _mm256_add_ps(wi, vtmp);
        __m128i nhwi = _mm256_cvtps_ph(nwi, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(nw + i), nhwi);
      }

      for (; i < N; ++i) {
        float gi = std::fma(
            weight_decay,
            _cvtsh_ss(reinterpret_cast<const unsigned short*>(w)[i]),
            g[i]);
        float nhi =
            _cvtsh_ss(reinterpret_cast<const unsigned short*>(h)[i]) + gi * gi;
        reinterpret_cast<unsigned short*>(nh)[i] = _cvtss_sh(nhi, 0);
        float nwi = _cvtsh_ss(reinterpret_cast<const unsigned short*>(w)[i]) +
            lr * gi / (std::sqrt(nhi) + epsilon);
        reinterpret_cast<unsigned short*>(nw)[i] = _cvtss_sh(nwi, 0);
      }
    */
}

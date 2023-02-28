// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8clamp/sse2.c]

pub fn pytorch_u8clamp_ukernel_sse2(
    n:      usize,
    x:      *const u8,
    y:      *mut u8,
    params: [PyTorchQnnpU8ClampingParams; 1])  {
    
    todo!();
        /*
            assert(n != 0);

      if
        PYTORCH_QNNP_LIKELY(n >= 8) {
          const __m128i voutput_max =
              _mm_load_si128((const __m128i*)&params->sse2.output_max);
          const __m128i voutput_min =
              _mm_load_si128((const __m128i*)&params->sse2.output_min);
          for (; n >= 64; n -= 64) {
            const __m128i vx0 = _mm_loadu_si128((const __m128i*)x);
            const __m128i vx1 = _mm_loadu_si128((const __m128i*)x + 1);
            const __m128i vx2 = _mm_loadu_si128((const __m128i*)x + 2);
            const __m128i vx3 = _mm_loadu_si128((const __m128i*)x + 3);
            x += 64;

            const __m128i vy0 =
                _mm_min_epu8(_mm_max_epu8(vx0, voutput_min), voutput_max);
            const __m128i vy1 =
                _mm_min_epu8(_mm_max_epu8(vx1, voutput_min), voutput_max);
            const __m128i vy2 =
                _mm_min_epu8(_mm_max_epu8(vx2, voutput_min), voutput_max);
            const __m128i vy3 =
                _mm_min_epu8(_mm_max_epu8(vx3, voutput_min), voutput_max);

            __builtin_prefetch(x + 640);

            _mm_storeu_si128((__m128i*)y, vy0);
            _mm_storeu_si128((__m128i*)y + 1, vy1);
            _mm_storeu_si128((__m128i*)y + 2, vy2);
            _mm_storeu_si128((__m128i*)y + 3, vy3);
            y += 64;
          }
          for (; n >= 8; n -= 8) {
            __m128i vout = _mm_loadl_epi64((const __m128i*)x);
            x += 8;
            vout = _mm_min_epu8(vout, voutput_max);
            vout = _mm_max_epu8(vout, voutput_min);
            _mm_storel_epi64((__m128i*)y, vout);
            y += 8;
          }
          if (n != 0) {
            const usize n_increment = n - 8;
            x = (const u8*)((uintptr_t)x + n_increment);
            y = (u8*)((uintptr_t)y + n_increment);

            __m128i vout = _mm_loadl_epi64((const __m128i*)x);
            vout = _mm_min_epu8(vout, voutput_max);
            vout = _mm_max_epu8(vout, voutput_min);
            _mm_storel_epi64((__m128i*)y, vout);
          }
        }
      else {
        const u32 voutput_max = params->sse2.output_max[0];
        const u32 voutput_min = params->sse2.output_min[0];
        do {
          u32 vout = *x++;
          vout = vout > voutput_max ? voutput_max : vout;
          vout = vout < voutput_min ? voutput_min : vout;
          *y++ = (u8)vout;
        } while (--n != 0);
      }
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8rmax/sse2.c]

pub fn pytorch_u8rmax_ukernel_sse2(
    n: usize,
    x: *const u8) -> u8 {
    
    todo!();
        /*
            assert(n != 0);

      if
        PYTORCH_QNNP_LIKELY(n >= 16) {
          __m128i vmax = _mm_setzero_si128();
          do {
            const __m128i vx = _mm_loadu_si128((const __m128i*)x);
            x += 16;
            vmax = _mm_max_epu8(vmax, vx);
            n -= 16;
          } while (n >= 16);
          if (n != 0) {
            const usize x_increment = n - 16;
            x = (const u8*)((uintptr_t)x + x_increment);
            const __m128i vx = _mm_loadu_si128((const __m128i*)x);
            vmax = _mm_max_epu8(vmax, vx);
          }
          vmax = _mm_max_epu8(vmax, _mm_unpackhi_epi64(vmax, vmax));
          vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
          vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
          vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
          return (u8)_mm_cvtsi128_si32(vmax);
        }
      else {
        u8 vmax = 0;
        do {
          const u8 vx = *x++;
          vmax = vx > vmax ? vx : vmax;
        } while (--n != 0);
        return vmax;
      }
        */
}



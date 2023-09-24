// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/x4-sse2.c]

pub fn pytorch_qnnp_x8zip_x4_sse2(
        n:      usize,
        input:  *const void,
        output: *mut void)  {
    
    todo!();
        /*
            const u8* x = input;
      const u8* y = x + n;
      const u8* z = y + n;
      const u8* w = z + n;
      u8* o = output;

      if (n >= 16) {
        do {
          const __m128i vx = _mm_loadu_si128((const __m128i*)x);
          x += 16;
          const __m128i vy = _mm_loadu_si128((const __m128i*)y);
          y += 16;
          const __m128i vz = _mm_loadu_si128((const __m128i*)z);
          z += 16;
          const __m128i vw = _mm_loadu_si128((const __m128i*)w);
          w += 16;
          const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
          const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
          const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
          const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
          const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
          const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
          const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
          const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
          _mm_storeu_si128((__m128i*)o, vxyzw0);
          _mm_storeu_si128((__m128i*)o + 1, vxyzw1);
          _mm_storeu_si128((__m128i*)o + 2, vxyzw2);
          _mm_storeu_si128((__m128i*)o + 3, vxyzw3);
          o = (void*)((uintptr_t)o + 64);
          n -= 16;
        } while (n >= 16);
        if (n != 0) {
          const usize address_increment = n - 16;
          const __m128i vx =
              _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment));
          const __m128i vy =
              _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment));
          const __m128i vz =
              _mm_loadu_si128((const __m128i*)((uintptr_t)z + address_increment));
          const __m128i vw =
              _mm_loadu_si128((const __m128i*)((uintptr_t)w + address_increment));
          const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
          const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
          const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
          const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
          const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
          const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
          const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
          const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);
          o = (void*)((uintptr_t)o + address_increment * 4);
          _mm_storeu_si128((__m128i*)o, vxyzw0);
          _mm_storeu_si128((__m128i*)o + 1, vxyzw1);
          _mm_storeu_si128((__m128i*)o + 2, vxyzw2);
          _mm_storeu_si128((__m128i*)o + 3, vxyzw3);
        }
      } else {
        do {
          const u8 vx = *x++;
          const u8 vy = *y++;
          const u8 vz = *z++;
          const u8 vw = *w++;
          o[0] = vx;
          o[1] = vy;
          o[2] = vz;
          o[3] = vw;
          o += 4;
        } while (--n != 0);
      }
        */
}

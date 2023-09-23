// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip/xm-sse2.c]

pub fn pytorch_qnnp_x8zip_xm_sse2(
    n:      usize,
    m:      usize,
    input:  *const void,
    output: *mut void)  {
    
    todo!();
        /*
            const u8* w = input;
      const usize input_increment = n * 3;
      const usize output_increment = 4 - m * n;
      const u8* last_input = w + n * (m - 1);
      void* last_output = (void*)((uintptr_t)output + (m - 4));

      if (n >= 8) {
        for (usize i = 0; i < m; i += 4) {
          usize k = n;
          w = (const u8*)((uintptr_t)w + input_increment);
          if (w >= last_input) {
            w = last_input;
          }
          const u8* z = (const u8*)((uintptr_t)w - n);
          const u8* y = (const u8*)((uintptr_t)z - n);
          const u8* x = (const u8*)((uintptr_t)y - n);
          while (k >= 16) {
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
            __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
            __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
            __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
            __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw2);
            output = (void*)((uintptr_t)output + m);
            vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw2);
            output = (void*)((uintptr_t)output + m);
            vxyzw2 = _mm_unpackhi_epi64(vxyzw2, vxyzw2);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw2);
            output = (void*)((uintptr_t)output + m);
            vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw2);
            output = (void*)((uintptr_t)output + m);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw3);
            output = (void*)((uintptr_t)output + m);
            vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw3);
            output = (void*)((uintptr_t)output + m);
            vxyzw3 = _mm_unpackhi_epi64(vxyzw3, vxyzw3);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw3);
            output = (void*)((uintptr_t)output + m);
            vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw3);
            output = (void*)((uintptr_t)output + m);
            k -= 16;
          };
          if (k >= 8) {
            const __m128i vx = _mm_loadl_epi64((const __m128i*)x);
            x += 8;
            const __m128i vy = _mm_loadl_epi64((const __m128i*)y);
            y += 8;
            const __m128i vz = _mm_loadl_epi64((const __m128i*)z);
            z += 8;
            const __m128i vw = _mm_loadl_epi64((const __m128i*)w);
            w += 8;
            const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
            const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
            __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
            __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);
            vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
            output = (void*)((uintptr_t)output + m);

            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
            *((u32*)output) = _mm_cvtsi128_si32(vxyzw1);
            output = (void*)((uintptr_t)output + m);
            k -= 8;
          }
          if (k != 0) {
            const usize address_decrement = 8 - k;
            x -= address_decrement;
            y -= address_decrement;
            z -= address_decrement;
            w -= address_decrement;
            const __m128i vshift = _mm_cvtsi32_si128(8 * address_decrement);

            const __m128i vx =
                _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)x), vshift);
            const __m128i vy =
                _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)y), vshift);
            const __m128i vz =
                _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)z), vshift);
            const __m128i vw =
                _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)w), vshift);
            w += 8;
            const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
            const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
            __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
            __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

            if (k & 4) {
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = vxyzw1;
            }

            if (k & 2) {
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
              vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
            }
            if (k & 1) {
              *((u32*)output) = _mm_cvtsi128_si32(vxyzw0);
              output = (void*)((uintptr_t)output + m);
            }
          }
          output = (void*)((uintptr_t)output + output_increment);
          if (output > last_output) {
            output = last_output;
          }
        }
      } else {
        const u8* i = input;
        u8* o = output;
        usize k = n;
        do {
          usize l = m;
          const u8* ii = i++;
          do {
            *o++ = *ii;
            ii += n;
          } while (--l != 0);
        } while (--k != 0);
      }
        */
}

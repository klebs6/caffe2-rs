// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8maxpool/16x9p8q-sse2.c]

pub fn pytorch_u8maxpool_ukernel_16x9p8q_sse2(
    n:                usize,
    ks:               usize,
    kc:               usize,
    input:            *const *const u8,
    output:           *mut u8,
    input_increment:  usize,
    output_increment: usize,
    params:           [PyTorchQnnpU8ClampingParams; 1])  {

    todo!();
        /*
            assert(n != 0);
      assert(ks != 0);
      assert(kc >= 16);

      const __m128i voutput_max =
          _mm_load_si128((const __m128i*)params->sse2.output_max);
      const __m128i voutput_min =
          _mm_load_si128((const __m128i*)params->sse2.output_min);

      do {
        u8* o = output;
        {
          const u8* i0 = *input++;
          const u8* i1 = *input++;
          const u8* i2 = *input++;
          const u8* i3 = *input++;
          const u8* i4 = *input++;
          const u8* i5 = *input++;
          const u8* i6 = *input++;
          const u8* i7 = *input++;
          const u8* i8 = *input++;
          if (ks < 2) {
            i1 = i0;
          }
          if (ks <= 2) {
            i2 = i0;
          }
          if (ks < 4) {
            i3 = i0;
          }
          if (ks <= 4) {
            i4 = i0;
          }
          if (ks < 6) {
            i5 = i0;
          }
          if (ks <= 6) {
            i6 = i0;
          }
          if (ks < 8) {
            i7 = i0;
          }
          if (ks <= 8) {
            i8 = i0;
          }

          usize k = kc;
          while (k >= 16) {
            const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
            i0 += 16;
            const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
            i1 += 16;
            const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
            i2 += 16;
            const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
            i3 += 16;
            const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
            i4 += 16;
            const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
            i5 += 16;
            const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
            i6 += 16;
            const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
            i7 += 16;
            const __m128i vi8 = _mm_loadu_si128((const __m128i*)i8);
            i8 += 16;

            const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
            const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
            const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
            const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

            const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
            const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
            const __m128i vmax = _mm_max_epu8(vmax2345, vmax01678);
            const __m128i vout =
                _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

            _mm_storeu_si128((__m128i*)o, vout);
            o += 16;

            k -= 16;
          }
          if (k != 0) {
            const usize address_increment = k - 16;
            i0 = (const u8*)((uintptr_t)i0 + address_increment);
            i1 = (const u8*)((uintptr_t)i1 + address_increment);
            i2 = (const u8*)((uintptr_t)i2 + address_increment);
            i3 = (const u8*)((uintptr_t)i3 + address_increment);
            i4 = (const u8*)((uintptr_t)i4 + address_increment);
            i5 = (const u8*)((uintptr_t)i5 + address_increment);
            i6 = (const u8*)((uintptr_t)i6 + address_increment);
            i7 = (const u8*)((uintptr_t)i7 + address_increment);
            i8 = (const u8*)((uintptr_t)i8 + address_increment);
            o = (u8*)((uintptr_t)o + address_increment);

            const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
            const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
            const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
            const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
            const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
            const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
            const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
            const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
            const __m128i vi8 = _mm_loadu_si128((const __m128i*)i8);

            const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
            const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
            const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
            const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

            const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
            const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
            const __m128i vmax = _mm_max_epu8(vmax2345, vmax01678);
            const __m128i vout =
                _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

            _mm_storeu_si128((__m128i*)o, vout);
            o += 16;
          }
        }

        for (ptrdiff_t m = (ptrdiff_t)ks - 9; m > 0; m -= 8) {
          const u8* i0 = *input++;
          const u8* i1 = *input++;
          const u8* i2 = *input++;
          const u8* i3 = *input++;
          const u8* i4 = *input++;
          const u8* i5 = *input++;
          const u8* i6 = *input++;
          const u8* i7 = *input++;
          if (m < 2) {
            i1 = i0;
          }
          if (m <= 2) {
            i2 = i0;
          }
          if (m < 4) {
            i3 = i0;
          }
          if (m <= 4) {
            i4 = i0;
          }
          if (m < 6) {
            i5 = i0;
          }
          if (m <= 6) {
            i6 = i0;
          }
          if (m < 8) {
            i7 = i0;
          }

          o = output;
          usize k = kc;
          while (k >= 16) {
            const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
            i0 += 16;
            const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
            i1 += 16;
            const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
            i2 += 16;
            const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
            i3 += 16;
            const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
            i4 += 16;
            const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
            i5 += 16;
            const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
            i6 += 16;
            const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
            i7 += 16;
            const __m128i vo = _mm_loadu_si128((const __m128i*)o);

            const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
            const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
            const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
            const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

            const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
            const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
            const __m128i vmax = _mm_max_epu8(vmax2345, vmax0167);
            const __m128i vout =
                _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

            _mm_storeu_si128((__m128i*)o, vout);
            o += 16;

            k -= 16;
          }
          if (k != 0) {
            const usize address_increment = k - 16;
            i0 = (const u8*)((uintptr_t)i0 + address_increment);
            i1 = (const u8*)((uintptr_t)i1 + address_increment);
            i2 = (const u8*)((uintptr_t)i2 + address_increment);
            i3 = (const u8*)((uintptr_t)i3 + address_increment);
            i4 = (const u8*)((uintptr_t)i4 + address_increment);
            i5 = (const u8*)((uintptr_t)i5 + address_increment);
            i6 = (const u8*)((uintptr_t)i6 + address_increment);
            i7 = (const u8*)((uintptr_t)i7 + address_increment);
            o = (u8*)((uintptr_t)o + address_increment);

            const __m128i vi0 = _mm_loadu_si128((const __m128i*)i0);
            const __m128i vi1 = _mm_loadu_si128((const __m128i*)i1);
            const __m128i vi2 = _mm_loadu_si128((const __m128i*)i2);
            const __m128i vi3 = _mm_loadu_si128((const __m128i*)i3);
            const __m128i vi4 = _mm_loadu_si128((const __m128i*)i4);
            const __m128i vi5 = _mm_loadu_si128((const __m128i*)i5);
            const __m128i vi6 = _mm_loadu_si128((const __m128i*)i6);
            const __m128i vi7 = _mm_loadu_si128((const __m128i*)i7);
            const __m128i vo = _mm_loadu_si128((const __m128i*)o);

            const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
            const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
            const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
            const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

            const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
            const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
            const __m128i vmax = _mm_max_epu8(vmax2345, vmax0167);
            const __m128i vout =
                _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

            _mm_storeu_si128((__m128i*)o, vout);
            o += 16;
          }
        }
        input = (const u8**)((uintptr_t)input + input_increment);
        output = (u8*)((uintptr_t)o + output_increment);
      } while (--n != 0);
        */
}

// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool/up8x7-sse2.c]

pub fn pytorch_q8gavgpool_ukernel_up8x7_sse2(
        m:                   usize,
        n:                   usize,
        input:               *const u8,
        input_stride:        usize,
        zero:                *const u8,
        output:              *mut u8,
        quantization_params: [PyTorchQnnpAvgPoolQuantizationParams; 1])  {
    
    todo!();
        /*
            assert(m >= 1);
      assert(m <= 7);
      assert(n >= 8);

      const u8* i0 = input;
      const u8* i1 = i0 + input_stride;
      if (m < 2) {
        i1 = zero;
      }
      const u8* i2 = i1 + input_stride;
      if (m <= 2) {
        i2 = zero;
      }
      const u8* i3 = i2 + input_stride;
      if (m < 4) {
        i3 = zero;
      }
      const u8* i4 = i3 + input_stride;
      if (m <= 4) {
        i4 = zero;
      }
      const u8* i5 = i4 + input_stride;
      if (m < 6) {
        i5 = zero;
      }
      const u8* i6 = i5 + input_stride;
      if (m <= 6) {
        i6 = zero;
      }
      const __m128i vbias =
          _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
      const __m128i vzero = _mm_setzero_si128();

      const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

      do {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0);
        i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1);
        i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2);
        i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3);
        i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4);
        i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5);
        i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6);
        i6 += 8;

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

        __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
        __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

        const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
        const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

        const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
        const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

        __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
        vout = _mm_adds_epi16(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_zero_point));
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_min_epu8(
            vout,
            _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
        vout = _mm_max_epu8(
            vout,
            _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

        _mm_storel_epi64((__m128i*)output, vout);
        output += 8;

        n -= 8;
      } while (n >= 8);
      if (n != 0) {
        const usize address_decrement = 8 - n;
        i0 = (const u8*)((uintptr_t)i0 - address_decrement);
        i1 = (const u8*)((uintptr_t)i1 - address_decrement);
        i2 = (const u8*)((uintptr_t)i2 - address_decrement);
        i3 = (const u8*)((uintptr_t)i3 - address_decrement);
        i4 = (const u8*)((uintptr_t)i4 - address_decrement);
        i5 = (const u8*)((uintptr_t)i5 - address_decrement);
        i6 = (const u8*)((uintptr_t)i6 - address_decrement);
        const __m128i vi_shift = _mm_cvtsi32_si128(8 * address_decrement);

        const __m128i vi0 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i0), vi_shift);
        const __m128i vi1 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i1), vi_shift);
        const __m128i vi2 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i2), vi_shift);
        const __m128i vi3 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i3), vi_shift);
        const __m128i vi4 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i4), vi_shift);
        const __m128i vi5 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i5), vi_shift);
        const __m128i vi6 =
            _mm_srl_epi64(_mm_loadl_epi64((const __m128i*)i6), vi_shift);

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

        __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vxi0, vzero));
        __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vxi0, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi1, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi1, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi2, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi2, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi3, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi3, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi4, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi4, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi5, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi5, vzero));
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi6, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi6, vzero));

        const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
        const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

        const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
        const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

        __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
        vout = _mm_adds_epi16(
            vout,
            _mm_load_si128(
                (const __m128i*)quantization_params->sse2.output_zero_point));
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_min_epu8(
            vout,
            _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
        vout = _mm_max_epu8(
            vout,
            _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

        if (n & 4) {
          *((u32*)output) = (u32)_mm_cvtsi128_si32(vout);
          output += 4;
          vout = _mm_srli_epi64(vout, 32);
        }
        if (n & 2) {
          *((u16*)output) = (u16)_mm_extract_epi16(vout, 0);
          output += 2;
          vout = _mm_srli_epi32(vout, 16);
        }
        if (n & 1) {
          *((u8*)output) = (u8)_mm_cvtsi128_si32(vout);
        }
      }
        */
}



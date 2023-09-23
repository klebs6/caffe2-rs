// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool/up8xm-sse2.c]

pub fn pytorch_q8avgpool_ukernel_up8xm_sse2(
    n:                   usize,
    ks:                  usize,
    kc:                  usize,
    input:               *const *const u8,
    zero:                *const u8,
    output:              *mut u8,
    input_increment:     usize,
    output_increment:    usize,
    quantization_params: [PyTorchQnnpAvgPoolQuantizationParams; 1])  {

    todo!();
    /*
            assert(n != 0);
      assert(ks != 0);
      assert(kc < 8);

      const __m128i vbias =
          _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
      const __m128i vzero = _mm_setzero_si128();
      const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

      do {
        const u8** next_input =
            (const u8**)((uintptr_t)input + input_increment);
        __m128i vacc_lo = vbias;
        __m128i vacc_hi = vbias;

        usize m = ks;
        do {
          const u8* i = *input++;
          i += kc;
          __m128i vi = _mm_setzero_si128();
          if (kc & 1) {
            i -= 1;
            vi = _mm_cvtsi32_si128((int)(u32)*i);
          }
          if (kc & 2) {
            vi = _mm_slli_epi32(vi, 16);
            i -= 2;
            vi = _mm_insert_epi16(vi, *((const u16*)i), 0);
          }
          if (kc & 4) {
            i -= 4;
            vi = _mm_unpacklo_epi32(
                _mm_cvtsi32_si128((int)*((const u32*)i)), vi);
          }

          const __m128i vxi = _mm_unpacklo_epi8(vi, vzero);
          vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi, vzero));
          vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi, vzero));
        } while (--m != 0);
        input = next_input;

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

        if (kc & 4) {
          *((u32*)output) = (u32)_mm_cvtsi128_si32(vout);
          output += 4;
          vout = _mm_srli_epi64(vout, 32);
        }
        if (kc & 2) {
          *((u16*)output) = (u16)_mm_extract_epi16(vout, 0);
          output += 2;
          vout = _mm_srli_epi32(vout, 16);
        }
        if (kc & 1) {
          *((u8*)output) = (u8)_mm_cvtsi128_si32(vout);
          output += 1;
        }
        output = (u8*)((uintptr_t)output + output_increment);
      } while (--n != 0);
        */
}

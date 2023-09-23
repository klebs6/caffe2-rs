// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/u8maxpool/sub16-sse2.c]

pub fn pytorch_u8maxpool_ukernel_sub16_sse2(
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
      assert(kc != 0);
      assert(kc < 16);

      const __m128i voutput_max =
          _mm_load_si128((const __m128i*)params->sse2.output_max);
      const __m128i voutput_min =
          _mm_load_si128((const __m128i*)params->sse2.output_min);

      do {
        __m128i vmax = _mm_setzero_si128();

        usize m = ks;
        do {
          const u8* i = *input++;
          i += kc;
          __m128i vi = vmax;
          if (kc & 1) {
            i -= 1;
            vi = _mm_cvtsi32_si128(*i);
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
          if (kc & 8) {
            i -= 8;
            vi = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)i), vi);
          }
          vmax = _mm_max_epu8(vmax, vi);
        } while (--m != 0);
        input = (const u8**)((uintptr_t)input + input_increment);
        __m128i vout = _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);

        if (kc & 8) {
          _mm_storel_epi64((__m128i*)output, vout);
          output += 8;
          vout = _mm_unpackhi_epi64(vout, vout);
        }
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

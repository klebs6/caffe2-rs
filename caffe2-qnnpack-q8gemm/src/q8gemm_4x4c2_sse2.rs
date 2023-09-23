// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm/4x4c2-sse2.c]

pub fn pytorch_q8gemm_ukernel_4x4c2_sse2(
        mr:                   usize,
        nr:                   usize,
        k:                    usize,
        a:                    *const u8,
        a_stride:             usize,
        w:                    *const void,
        c:                    *mut u8,
        c_stride:             usize,
        output_channel_index: usize,
        quantization_params:  [PyTorchQnnpConvQuantizationParams; 1])  {
    
    todo!();
        /*
            __m128i vacc0x0123 = _mm_loadu_si128((const __m128i*)w);
      __m128i vacc1x0123 = vacc0x0123;
      __m128i vacc2x0123 = vacc0x0123;
      __m128i vacc3x0123 = vacc0x0123;
      w = (const void*)((uintptr_t)w + 16);

      const u8* a0 = a;
      const u8* a1 = (const u8*)((uintptr_t)a0 + a_stride);
      if (mr < 2) {
        a1 = a0;
      }
      const u8* a2 = (const u8*)((uintptr_t)a1 + a_stride);
      if (mr <= 2) {
        a2 = a1;
      }
      const u8* a3 = (const u8*)((uintptr_t)a2 + a_stride);
      if (mr != 4) {
        a3 = a2;
      }

      const __m128i va_zero_point = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.input_zero_point);
      const i16 vb_zero_point_0 =
        (i16)(u16)quantization_params->sse2.kernel_zero_points[
        output_channel_index];
      const i16 vb_zero_point_1 =
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 1];
      const i16 vb_zero_point_2 =
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 2];
      const i16 vb_zero_point_3 =
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 3];

      __m128i vb_zero_point = _mm_set_epi16(vb_zero_point_3,
                                            vb_zero_point_3,
                                            vb_zero_point_2,
                                            vb_zero_point_2,
                                            vb_zero_point_1,
                                            vb_zero_point_1,
                                            vb_zero_point_0,
                                            vb_zero_point_0
                                            );
      const __m128i vzero = _mm_setzero_si128();
      for (; k >= 8; k -= 8) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
        const __m128i vxa0 =
            sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
        const __m128i vxa1 =
            sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*)a2);
        const __m128i vxa2 =
            sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
        a2 += 8;
        const __m128i va3 = _mm_loadl_epi64((const __m128i*)a3);
        const __m128i vxa3 =
            sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);
        a3 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
        const __m128i vxb0 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

        const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
        const __m128i vxb1 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

        const __m128i vb2 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

        const __m128i vb3 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
        const __m128i vxb3 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
        w = (const void*)((uintptr_t)w + 32);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
      }
      if (k != 0) {
        const usize a_predecrement = 8 - k;
        const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

        const __m128i va0 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
        const __m128i vxa0 =
            sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
        const __m128i va1 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
        const __m128i vxa1 =
            sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
        const __m128i va2 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a2 - a_predecrement)), va_shift);
        const __m128i vxa2 =
            sub_zero_point(_mm_unpacklo_epi8(va2, vzero), va_zero_point);
        const __m128i va3 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a3 - a_predecrement)), va_shift);
        const __m128i vxa3 =
            sub_zero_point(_mm_unpacklo_epi8(va3, vzero), va_zero_point);

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*)w);
        const __m128i vxb0 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);

        vacc0x0123 = _mm_add_epi32(
            vacc0x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa0, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc1x0123 = _mm_add_epi32(
            vacc1x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa1, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc2x0123 = _mm_add_epi32(
            vacc2x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa2, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));
        vacc3x0123 = _mm_add_epi32(
            vacc3x0123,
            _mm_madd_epi16(_mm_shuffle_epi32(vxa3, _MM_SHUFFLE(0, 0, 0, 0)), vxb0));

        if (k > 2) {
          const __m128i vb1 = _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 8));
          const __m128i vxb1 =
              _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);

          vacc0x0123 = _mm_add_epi32(
              vacc0x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
          vacc1x0123 = _mm_add_epi32(
              vacc1x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
          vacc2x0123 = _mm_add_epi32(
              vacc2x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));
          vacc3x0123 = _mm_add_epi32(
              vacc3x0123,
              _mm_madd_epi16(
                  _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(1, 1, 1, 1)), vxb1));

          if (k > 4) {
            const __m128i vb2 =
                _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 16));
            const __m128i vxb2 =
                _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);

            vacc0x0123 = _mm_add_epi32(
                vacc0x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
            vacc1x0123 = _mm_add_epi32(
                vacc1x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
            vacc2x0123 = _mm_add_epi32(
                vacc2x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));
            vacc3x0123 = _mm_add_epi32(
                vacc3x0123,
                _mm_madd_epi16(
                    _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(2, 2, 2, 2)), vxb2));

            if (k > 6) {
              const __m128i vb3 =
                  _mm_loadl_epi64((const __m128i*)((uintptr_t)w + 24));
              const __m128i vxb3 =
                  _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);

              vacc0x0123 = _mm_add_epi32(
                  vacc0x0123,
                  _mm_madd_epi16(
                      _mm_shuffle_epi32(vxa0, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
              vacc1x0123 = _mm_add_epi32(
                  vacc1x0123,
                  _mm_madd_epi16(
                      _mm_shuffle_epi32(vxa1, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
              vacc2x0123 = _mm_add_epi32(
                  vacc2x0123,
                  _mm_madd_epi16(
                      _mm_shuffle_epi32(vxa2, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
              vacc3x0123 = _mm_add_epi32(
                  vacc3x0123,
                  _mm_madd_epi16(
                      _mm_shuffle_epi32(vxa3, _MM_SHUFFLE(3, 3, 3, 3)), vxb3));
            }
          }
        }
      }

      const __m128 vmultiplier =
          _mm_loadu_ps(&quantization_params->sse2.requantization_scales[output_channel_index]);

      vacc0x0123 = _mm_cvtps_epi32(
                    _mm_mul_ps(
                      _mm_cvtepi32_ps(vacc0x0123),
                      vmultiplier
                      )
                    );
      vacc1x0123 = _mm_cvtps_epi32(
                    _mm_mul_ps(
                      _mm_cvtepi32_ps(vacc1x0123),
                      vmultiplier
                      )
                    );
      vacc2x0123 = _mm_cvtps_epi32(
                    _mm_mul_ps(
                      _mm_cvtepi32_ps(vacc2x0123),
                      vmultiplier
                      )
                    );
      vacc3x0123 = _mm_cvtps_epi32(
                    _mm_mul_ps(
                      _mm_cvtepi32_ps(vacc3x0123),
                      vmultiplier
                      )
                    );

      const __m128i voutput_zero_point = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.output_zero_point);
      const __m128i vacc01x0123 = _mm_adds_epi16(
          _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
      const __m128i vacc23x0123 = _mm_adds_epi16(
          _mm_packs_epi32(vacc2x0123, vacc3x0123), voutput_zero_point);
      __m128i vout = _mm_packus_epi16(vacc01x0123, vacc23x0123);
      vout = _mm_min_epu8(
          vout,
          _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
      vout = _mm_max_epu8(
          vout,
          _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

      u8* c0 = c;
      u8* c1 = (u8*)((uintptr_t)c0 + c_stride);
      if (mr < 2) {
        c1 = c0;
      }
      u8* c2 = (u8*)((uintptr_t)c1 + c_stride);
      if (mr <= 2) {
        c2 = c1;
      }
      u8* c3 = (u8*)((uintptr_t)c2 + c_stride);
      if (mr != 4) {
        c3 = c2;
      }
      if (nr == 4) {
        *((u32*)c0) = (u32)_mm_cvtsi128_si32(vout);
        *((u32*)c1) = (u32)_mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));
        *((u32*)c2) =
            (u32)_mm_cvtsi128_si32(_mm_unpackhi_epi32(vout, vout));
        *((u32*)c3) = (u32)_mm_cvtsi128_si32(_mm_srli_si128(vout, 12));
      } else {
        if (nr >= 2) {
          *((u16*)c0) = (u16)_mm_extract_epi16(vout, 0);
          c0 += 2;
          *((u16*)c1) = (u16)_mm_extract_epi16(vout, 2);
          c1 += 2;
          *((u16*)c2) = (u16)_mm_extract_epi16(vout, 4);
          c2 += 2;
          *((u16*)c3) = (u16)_mm_extract_epi16(vout, 6);
          c3 += 2;
          vout = _mm_srli_epi32(vout, 16);
          nr -= 2;
        }
        if (nr != 0) {
          *((u8*)c0) = (u8)_mm_cvtsi128_si32(vout);
          *((u8*)c1) = (u8)_mm_extract_epi16(vout, 2);
          *((u8*)c2) = (u8)_mm_extract_epi16(vout, 4);
          *((u8*)c3) = (u8)_mm_extract_epi16(vout, 6);
        }
      }
        */
}

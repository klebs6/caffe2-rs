// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm/2x4c8-sse2.c]

#[inline] pub fn pytorch_sse_reduce4_i32(
        x: __m128i,
        y: __m128i,
        z: __m128i,
        w: __m128i) -> __m128i {
    
    todo!();
        /*
            #if defined(__SSSE3__) && !defined(__ANDROID__)
      /* xxyy = ( y2 + y3, y0 + y1, x2 + x3, x0 + x1 ) */
      const __m128i xxyy = _mm_hadd_epi32(x, y);
      /* zzww = ( w2 + w3, w0 + w1, z2 + z3, z0 + z1 ) */
      const __m128i zzww = _mm_hadd_epi32(z, w);
      /* xyzw = ( w0 + w1 + w2 + w3, y0 + y1 + y2 + y3, z0 + z1 + z2 + z3, x0 + x1 +
       * x2 + x3 ) */
      return _mm_hadd_epi32(xxyy, zzww);
    #else
      /* xzxz = ( z1 + z3, x1 + x3, z0 + z2, x0 + x2 ) */
      const __m128i xzxz =
          _mm_add_epi32(_mm_unpacklo_epi32(x, z), _mm_unpackhi_epi32(x, z));
      /* ywyw = ( w1 + w3, y1 + y3, w0 + w2, y0 + y2 ) */
      const __m128i ywyw =
          _mm_add_epi32(_mm_unpacklo_epi32(y, w), _mm_unpackhi_epi32(y, w));
      /* xyzw = ( w0 + w2 + w1 + w3, y0 + y2 + y1 + y3, z0 + z2 + z1 + z3, x0 + x2 +
       * x1 + x3 ) */
      return _mm_add_epi32(
          _mm_unpacklo_epi32(xzxz, ywyw), _mm_unpackhi_epi32(xzxz, ywyw));
    #endif
        */
}

pub fn pytorch_q8gemm_ukernel_2x4c8_sse2(
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
            __m128i vacc00 = _mm_cvtsi32_si128((int)((const i32*)w)[0]);
      __m128i vacc01 = _mm_cvtsi32_si128((int)((const i32*)w)[1]);
      __m128i vacc02 = _mm_cvtsi32_si128((int)((const i32*)w)[2]);
      __m128i vacc03 = _mm_cvtsi32_si128((int)((const i32*)w)[3]);
      __m128i vacc10 = vacc00;
      __m128i vacc11 = vacc01;
      __m128i vacc12 = vacc02;
      __m128i vacc13 = vacc03;
      w = (const void*)((uintptr_t)w + 16);

      const u8* a0 = a;
      const u8* a1 = (const u8*)((uintptr_t)a0 + a_stride);
      if (mr != 2) {
        a1 = a0;
      }

      const u8* b0 = w;
      const u8* b1 = b0 + 8;
      if (nr < 2) {
        b1 = b0;
      }
      const u8* b2 = b1 + 8;
      if (nr <= 2) {
        b2 = b1;
      }
      const u8* b3 = b2 + 8;
      if (nr != 4) {
        b3 = b2;
      }
      const usize b_stride = nr * 8;

      const __m128i va_zero_point = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.input_zero_point);
      const __m128i vb_zero_point_0 = _mm_set1_epi16(
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index]);
      // Assumes kernel_zero_point allocated memory is always multiple of nr=4.
      const __m128i vb_zero_point_1 = _mm_set1_epi16(
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 1]);
      const __m128i vb_zero_point_2 = _mm_set1_epi16(
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 2]);
      const __m128i vb_zero_point_3 = _mm_set1_epi16(
          (i16)(u16)quantization_params->sse2.kernel_zero_points[
            output_channel_index + 3]);
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

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*)b0);
        const __m128i vxb0 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point_0);
        b0 += b_stride;
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*)b1);
        const __m128i vxb1 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point_1);
        b1 += b_stride;
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*)b2);
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point_2);
        b2 += b_stride;
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*)b3);
        const __m128i vxb3 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point_3);
        b3 += b_stride;

        vacc00 = _mm_add_epi32(vacc00, _mm_madd_epi16(vxa0, vxb0));
        vacc01 = _mm_add_epi32(vacc01, _mm_madd_epi16(vxa0, vxb1));
        vacc02 = _mm_add_epi32(vacc02, _mm_madd_epi16(vxa0, vxb2));
        vacc03 = _mm_add_epi32(vacc03, _mm_madd_epi16(vxa0, vxb3));
        vacc10 = _mm_add_epi32(vacc10, _mm_madd_epi16(vxa1, vxb0));
        vacc11 = _mm_add_epi32(vacc11, _mm_madd_epi16(vxa1, vxb1));
        vacc12 = _mm_add_epi32(vacc12, _mm_madd_epi16(vxa1, vxb2));
        vacc13 = _mm_add_epi32(vacc13, _mm_madd_epi16(vxa1, vxb3));
      }
      if (k != 0) {
        const usize a_predecrement = 8 - k;
        const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

        const __m128i va_zero_point_partial = _mm_unpacklo_epi8(
            _mm_srl_epi64(_mm_packus_epi16(va_zero_point, va_zero_point), va_shift),
            vzero);

        const __m128i va0 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);
        const __m128i vxa0 =
            sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point_partial);
        const __m128i va1 = _mm_srl_epi64(
            _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
        const __m128i vxa1 =
            sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point_partial);

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*)b0);
        const __m128i vxb0 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point_0);
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*)b1);
        const __m128i vxb1 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point_1);
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*)b2);
        const __m128i vxb2 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point_2);
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*)b3);
        const __m128i vxb3 =
            _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point_3);

        vacc00 = _mm_add_epi32(vacc00, _mm_madd_epi16(vxa0, vxb0));
        vacc01 = _mm_add_epi32(vacc01, _mm_madd_epi16(vxa0, vxb1));
        vacc02 = _mm_add_epi32(vacc02, _mm_madd_epi16(vxa0, vxb2));
        vacc03 = _mm_add_epi32(vacc03, _mm_madd_epi16(vxa0, vxb3));
        vacc10 = _mm_add_epi32(vacc10, _mm_madd_epi16(vxa1, vxb0));
        vacc11 = _mm_add_epi32(vacc11, _mm_madd_epi16(vxa1, vxb1));
        vacc12 = _mm_add_epi32(vacc12, _mm_madd_epi16(vxa1, vxb2));
        vacc13 = _mm_add_epi32(vacc13, _mm_madd_epi16(vxa1, vxb3));
      }

      __m128i vacc0x0123 = pytorch_sse_reduce4_i32(vacc00, vacc01, vacc02, vacc03);
      __m128i vacc1x0123 = pytorch_sse_reduce4_i32(vacc10, vacc11, vacc12, vacc13);

      const __m128 vmultiplier =
          _mm_loadu_ps(&quantization_params->sse2.requantization_scales
              [output_channel_index]);

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

      const __m128i voutput_zero_point = _mm_load_si128(
          (const __m128i*)quantization_params->sse2.output_zero_point);
      const __m128i vacc01x0123 = _mm_adds_epi16(
          _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
      __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);
      vout = _mm_min_epu8(
          vout,
          _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
      vout = _mm_max_epu8(
          vout,
          _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

      u8* c0 = c;
      u8* c1 = (u8*)((uintptr_t)c0 + c_stride);
      if (mr != 2) {
        c1 = c0;
      }
      if (nr == 4) {
        *((u32*)c0) = (u32)_mm_cvtsi128_si32(vout);
        *((u32*)c1) = (u32)_mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));
      } else {
        if (nr >= 2) {
          *((u16*)c0) = (u16)_mm_extract_epi16(vout, 0);
          c0 += 2;
          *((u16*)c1) = (u16)_mm_extract_epi16(vout, 2);
          c1 += 2;
          vout = _mm_srli_epi32(vout, 16);
          nr -= 2;
        }
        if (nr != 0) {
          *((u8*)c0) = (u8)_mm_cvtsi128_si32(vout);
          *((u8*)c1) = (u8)_mm_extract_epi16(vout, 2);
        }
      }
        */
}

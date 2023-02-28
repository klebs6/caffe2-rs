crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h]

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn cvtbf16_fp32(
        a:  &__m256i,
        o1: &mut __m256,
        o2: &mut __m256)  {
    
    todo!();
        /*
            __m128i lo = _mm256_extractf128_si256(a, 0);
      __m128i hi = _mm256_extractf128_si256(a, 1);
      o1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
      o2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16));
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn cvtfp32_bf16(
        a: &__m256,
        b: &__m256) -> __m256i {
    
    todo!();
        /*
            __m256i lo = _mm256_castps_si256(a);
      __m256i hi = _mm256_castps_si256(b);
      __m256i nan = _mm256_set1_epi32(0xffff);
      __m256i mask_lo = _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_ORD_Q));
      __m256i mask_hi = _mm256_castps_si256(_mm256_cmp_ps(b, b, _CMP_ORD_Q));
      __m256i ones = _mm256_set1_epi32(0x1);
      __m256i vec_bias = _mm256_set1_epi32(0x7fff);
      // u32 lsb = (input >> 16) & 1;
      auto t_lo = _mm256_and_si256(_mm256_srli_epi32(lo, 16), ones);
      auto t_hi = _mm256_and_si256(_mm256_srli_epi32(hi, 16), ones);
      // u32 rounding_bias = 0x7fff + lsb;
      t_lo = _mm256_add_epi32(t_lo, vec_bias);
      t_hi = _mm256_add_epi32(t_hi, vec_bias);
      // input += rounding_bias;
      t_lo = _mm256_add_epi32(t_lo, lo);
      t_hi = _mm256_add_epi32(t_hi, hi);
      // input = input >> 16;
      t_lo = _mm256_srli_epi32(t_lo, 16);
      t_hi = _mm256_srli_epi32(t_hi, 16);
      // Check NaN before converting back to bf16
      t_lo = _mm256_blendv_epi8(nan, t_lo, mask_lo);
      t_hi = _mm256_blendv_epi8(nan, t_hi, mask_hi);

      t_lo = _mm256_packus_epi32(t_lo, t_hi);      // t_hi[4-7] t_lo[4-7] t_hi[0-4] t_lo[0-4]
      return _mm256_permute4x64_epi64(t_lo, 0xd8); // 11        01        10        00
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn merge_compare_result(
        a: &__m256,
        b: &__m256) -> __m256i {
    
    todo!();
        /*
            __m256i lo = _mm256_castps_si256(a);
      __m256i hi = _mm256_castps_si256(b);
      lo = _mm256_srli_epi32(lo, 16);
      hi = _mm256_srli_epi32(hi, 16);
      auto out = _mm256_packus_epi32(lo, hi);
      return _mm256_permute4x64_epi64(out, 0xd8);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[derive(Default)]
pub struct VectorizedBFloat16 {
    values: __m256i,
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
pub mod vectorized_bfloat16 {

    pub type value_type = u16;
    pub type size_type  = i32;
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
impl VectorizedBFloat16 {

    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 16;
        */
    }
    
    pub fn new(v: __m256i) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: bf16) -> Self {
    
        todo!();
        /*


            value_type uw = val.x;
        values = _mm256_set1_epi16(uw);
        */
    }
    
    pub fn new(
        val1:  bf16,
        val2:  bf16,
        val3:  bf16,
        val4:  bf16,
        val5:  bf16,
        val6:  bf16,
        val7:  bf16,
        val8:  bf16,
        val9:  bf16,
        val10: bf16,
        val11: bf16,
        val12: bf16,
        val13: bf16,
        val14: bf16,
        val15: bf16,
        val16: bf16) -> Self {
    
        todo!();
        /*


            values = _mm256_setr_epi16(
            val1.x, val2.x, val3.x, val4.x, val5.x, val6.x, val7.x, val8.x,
            val9.x, val10.x, val11.x, val12.x, val13.x, val14.x, val15.x, val16.x);
        */
    }
    
    pub fn operator_m256i(&self) -> __m256i {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn zero_mask(&self) -> i32 {
        
        todo!();
        /*
            // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
        __m256i cmp = _mm256_cmpeq_epi16(values, _mm256_set1_epi16(0));
        return _mm256_movemask_epi8(cmp);
        */
    }
    
    pub fn loadu(ptr: *const void) -> Vectorized<bf16> {
        
        todo!();
        /*
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i16) -> Vectorized<bf16> {
        
        todo!();
        /*
            __at_align32__ i16 tmp_values[size()];
        memcpy(tmp_values, ptr, count * sizeof(i16));
        return loadu(tmp_values);
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i32)  {
        
        todo!();
        /*
            if (count == size()) {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
        } else if (count > 0) {
          __at_align32__ i16 tmp_values[size()];
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(i16));
        }
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<bf16>,
        b: &Vectorized<bf16>) -> Vectorized<bf16> {
    
        todo!();
        /*
            __at_align32__ i16 tmp_values[size()];
        a.store(tmp_values);
        if (mask & 0x01)
          tmp_values[0] = _mm256_extract_epi16(b.values, 0);
        if (mask & 0x02)
          tmp_values[1] = _mm256_extract_epi16(b.values, 1);
        if (mask & 0x04)
          tmp_values[2] = _mm256_extract_epi16(b.values, 2);
        if (mask & 0x08)
          tmp_values[3] = _mm256_extract_epi16(b.values, 3);
        if (mask & 0x10)
          tmp_values[4] = _mm256_extract_epi16(b.values, 4);
        if (mask & 0x20)
          tmp_values[5] = _mm256_extract_epi16(b.values, 5);
        if (mask & 0x40)
          tmp_values[6] = _mm256_extract_epi16(b.values, 6);
        if (mask & 0x80)
          tmp_values[7] = _mm256_extract_epi16(b.values, 7);
        if (mask & 0x100)
          tmp_values[8] = _mm256_extract_epi16(b.values, 8);
        if (mask & 0x200)
          tmp_values[9] = _mm256_extract_epi16(b.values, 9);
        if (mask & 0x400)
          tmp_values[10] = _mm256_extract_epi16(b.values, 10);
        if (mask & 0x800)
          tmp_values[11] = _mm256_extract_epi16(b.values, 11);
        if (mask & 0x1000)
          tmp_values[12] = _mm256_extract_epi16(b.values, 12);
        if (mask & 0x2000)
          tmp_values[13] = _mm256_extract_epi16(b.values, 13);
        if (mask & 0x4000)
          tmp_values[14] = _mm256_extract_epi16(b.values, 14);
        if (mask & 0x8000)
          tmp_values[15] = _mm256_extract_epi16(b.values, 15);
        return loadu(tmp_values);
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<bf16>,
        b:    &Vectorized<bf16>,
        mask: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return _mm256_blendv_epi8(a.values, b.values, mask.values);
        */
    }

    pub fn arange(
        base: bf16,
        step: Step) -> Vectorized<bf16> {

        let base: bf16 = base.unwrap_or(0.0);
        let step: Step = step.unwrap_or(1);

        todo!();
        /*
            return Vectorized<BFloat16>(
          base,             base +      step, base +  2 * step, base +  3 * step,
          base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
          base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
          base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
        */
    }
    
    pub fn set(
        a:     &Vectorized<bf16>,
        b:     &Vectorized<bf16>,
        count: i64) -> Vectorized<bf16> {
        
        todo!();
        /*
            switch (count) {
          case 0:
            return a;
          case 1:
            return blend<1>(a, b);
          case 2:
            return blend<3>(a, b);
          case 3:
            return blend<7>(a, b);
          case 4:
            return blend<15>(a, b);
          case 5:
            return blend<31>(a, b);
          case 6:
            return blend<63>(a, b);
          case 7:
            return blend<127>(a, b);
          case 8:
            return blend<255>(a, b);
          case 9:
            return blend<511>(a, b);
          case 10:
            return blend<1023>(a, b);
          case 11:
            return blend<2047>(a, b);
          case 12:
            return blend<4095>(a, b);
          case 13:
            return blend<8191>(a, b);
          case 14:
            return blend<16383>(a, b);
          case 15:
            return blend<32767>(a, b);
        }
        return b;
        */
    }
    
    pub fn map(&self, vop: fn(x: __m256) -> __m256) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        const auto o1 = vop(lo);
        const auto o2 = vop(hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn abs(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        const auto mask = _mm256_set1_ps(-0.f);
        const auto o1 = _mm256_andnot_ps(mask, lo);
        const auto o2 = _mm256_andnot_ps(mask, hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn angle(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto angle_lambda = [](__m256 values) {
          const auto zero_vec = _mm256_set1_ps(0.f);
          const auto nan_vec = _mm256_set1_ps(NAN);
          const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
          const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
          const auto pi = _mm256_set1_ps(pi<float>);

          const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
          auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
          angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
          return angle;
        };
        auto o1 = angle_lambda(lo);
        auto o2 = angle_lambda(hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn real(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn imag(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return _mm256_set1_epi16(0);
        */
    }
    
    pub fn conj(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn acos(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_acosf8_u10);
        */
    }
    
    pub fn asin(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_asinf8_u10);
        */
    }
    
    pub fn atan(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_atanf8_u10);
        */
    }
    
    pub fn atan2(&self, b: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        __m256 b1, b2;
        cvtbf16_fp32(values, lo, hi);
        cvtbf16_fp32(b.values, b1, b2);
        auto o1 = Sleef_atan2f8_u10(lo, b1);
        auto o2 = Sleef_atan2f8_u10(hi, b2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn copysign(&self, sign: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            // copy sign bit (0x8000) from sign and remaining bits from values
        __m256i mask_value = _mm256_set1_epi32(~0x80008000);
        __m256i mask_signbit = _mm256_set1_epi32(0x80008000);
        return Vectorized<BFloat16>(
          _mm256_or_si256(
            _mm256_and_si256(values, mask_value),
            _mm256_and_si256(sign, mask_signbit)));
        */
    }
    
    pub fn erf(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_erff8_u10);
        */
    }
    
    pub fn erfc(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_erfcf8_u15);
        */
    }
    
    pub fn erfinv(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        __at_align32__ float tmp1[size() / 2], tmp2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
        for (i64 i = 0; i < size() / 2; i++) {
          tmp1[i] = calc_erfinv(tmp1[i]);
          tmp2[i] = calc_erfinv(tmp2[i]);
        }
        auto o1 = _mm256_loadu_ps(tmp1);
        auto o2 = _mm256_loadu_ps(tmp2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn exp(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_expf8_u10);
        */
    }
    
    pub fn expm1(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_expm1f8_u10);
        */
    }
    
    pub fn fmod(&self, q: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 x_lo, x_hi;
        cvtbf16_fp32(values, x_lo, x_hi);
        __m256 q_lo, q_hi;
        cvtbf16_fp32(q.values, q_lo, q_hi);
        auto o1 = Sleef_fmodf8(x_lo, q_lo);
        auto o2 = Sleef_fmodf8(x_hi, q_hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        __m256 b1, b2;
        cvtbf16_fp32(values, lo, hi);
        cvtbf16_fp32(b.values, b1, b2);
        auto o1 = Sleef_hypotf8_u05(lo, b1);
        auto o2 = Sleef_hypotf8_u05(hi, b2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn i0(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        __at_align32__ float tmp1[size() / 2], tmp2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
        for (i64 i = 0; i < size() / 2; i++) {
          tmp1[i] = calc_i0(tmp1[i]);
          tmp2[i] = calc_i0(tmp2[i]);
        }
        auto o1 = _mm256_loadu_ps(tmp1);
        auto o2 = _mm256_loadu_ps(tmp2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn i0e(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        constexpr auto sz = size();
        __at_align32__ float tmp1[sz / 2], tmp2[sz / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

        for (auto i = decltype(sz){0}; i < sz / 2; i++) {
          tmp1[i] = calc_i0e(tmp1[i]);
          tmp2[i] = calc_i0e(tmp2[i]);
        }
        const auto o1 = _mm256_loadu_ps(tmp1);
        const auto o2 = _mm256_loadu_ps(tmp2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        __m256 xlo, xhi;
        cvtbf16_fp32(values, lo, hi);
        cvtbf16_fp32(x.values, xlo, xhi);
        __at_align32__ float tmp1[size() / 2], tmp2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
        __at_align32__ float tmpx1[size() / 2], tmpx2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
        for (i64 i = 0; i < size() / 2; ++i) {
          tmp1[i] = calc_igamma(tmp1[i], tmpx1[i]);
          tmp2[i] = calc_igamma(tmp2[i], tmpx2[i]);
        }
        auto o1 = _mm256_loadu_ps(tmp1);
        auto o2 = _mm256_loadu_ps(tmp2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        __m256 xlo, xhi;
        cvtbf16_fp32(values, lo, hi);
        cvtbf16_fp32(x.values, xlo, xhi);
        __at_align32__ float tmp1[size() / 2], tmp2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
        __at_align32__ float tmpx1[size() / 2], tmpx2[size() / 2];
        _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
        _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
        for (i64 i = 0; i < size() / 2; ++i) {
          tmp1[i] = calc_igammac(tmp1[i], tmpx1[i]);
          tmp2[i] = calc_igammac(tmp2[i], tmpx2[i]);
        }
        auto o1 = _mm256_loadu_ps(tmp1);
        auto o2 = _mm256_loadu_ps(tmp2);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn log(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_logf8_u10);
        */
    }
    
    pub fn log2(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_log2f8_u10);
        */
    }
    
    pub fn log10(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_log10f8_u10);
        */
    }
    
    pub fn log1p(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_log1pf8_u10);
        */
    }
    
    pub fn frac(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn sin(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_sinf8_u10);
        */
    }
    
    pub fn sinh(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_sinhf8_u10);
        */
    }
    
    pub fn cos(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_cosf8_u10);
        */
    }
    
    pub fn cosh(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_coshf8_u10);
        */
    }
    
    pub fn ceil(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto o1 = _mm256_ceil_ps(lo);
        auto o2 = _mm256_ceil_ps(hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn floor(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto o1 = _mm256_floor_ps(lo);
        auto o2 = _mm256_floor_ps(hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn neg(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto mask = _mm256_set1_ps(-0.f);
        auto o1 = _mm256_xor_ps(mask, lo);
        auto o2 = _mm256_xor_ps(mask, hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn round(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn tan(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_tanf8_u10);
        */
    }
    
    pub fn tanh(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_tanhf8_u10);
        */
    }
    
    pub fn trunc(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn lgamma(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return map(Sleef_lgammaf8_u10);
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto o1 = _mm256_sqrt_ps(lo);
        auto o2 = _mm256_sqrt_ps(hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn reciprocal(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto ones = _mm256_set1_ps(1);
        auto o1 = _mm256_div_ps(ones, lo);
        auto o2 = _mm256_div_ps(ones, hi);
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        cvtbf16_fp32(values, lo, hi);
        auto ones = _mm256_set1_ps(1);
        auto o1 = _mm256_div_ps(ones, _mm256_sqrt_ps(lo));
        auto o2 = _mm256_div_ps(ones, _mm256_sqrt_ps(hi));
        return cvtfp32_bf16(o1, o2);
        */
    }
    
    pub fn pow(&self, b: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            __m256 lo, hi;
        __m256 b1, b2;
        cvtbf16_fp32(values, lo, hi);
        cvtbf16_fp32(b.values, b1, b2);
        auto o1 = Sleef_powf8_u10(lo, b1);
        auto o2 = Sleef_powf8_u10(hi, b2);
        return cvtfp32_bf16(o1, o2);
        */
    }
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn bfloat16_binary_op_as_fp32<Op>(
        a:  &Vectorized<bf16>,
        b:  &Vectorized<bf16>,
        op: Op) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 b_lo, b_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(b), b_lo, b_hi);
      auto o1 = op(a_lo, b_lo);
      auto o2 = op(a_hi, b_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn bfloat16_compare_as_fp32<Op>(
        a:  &Vectorized<bf16>,
        b:  &Vectorized<bf16>,
        op: Op) -> Vectorized<bf16> {

    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 b_lo, b_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(b), b_lo, b_hi);
      auto o1 = op(a_lo, b_lo);
      auto o2 = op(a_hi, b_hi);
      return merge_compare_result(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
lazy_static!{
    /*
    Vectorized<BFloat16>  operator>(const Vectorized<BFloat16>& other)  {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
      });
    }
    Vectorized<BFloat16> inline Vectorized<BFloat16>::operator<(const Vectorized<BFloat16>& other) const {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
      });
    }
    Vectorized<BFloat16> inline Vectorized<BFloat16>::operator>=(const Vectorized<BFloat16>& other) const {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
      });
    }
    Vectorized<BFloat16> inline Vectorized<BFloat16>::operator<=(const Vectorized<BFloat16>& other) const {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_LE_OQ);
      });
    }
    Vectorized<BFloat16> inline Vectorized<BFloat16>::operator==(const Vectorized<BFloat16>& other) const {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
      });
    }
    Vectorized<BFloat16> inline Vectorized<BFloat16>::operator!=(const Vectorized<BFloat16>& other) const {
      return bfloat16_compare_as_fp32(*this, other, [](__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ);
      });
    }
    */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
lazy_static!{
    /*
    Vectorized<BFloat16> inline operator+(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return bfloat16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
    }
    Vectorized<BFloat16> inline operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return bfloat16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
    }
    Vectorized<BFloat16> inline operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return bfloat16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
    }
    Vectorized<BFloat16> inline operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return bfloat16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
    }

    Vectorized<BFloat16> inline operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return _mm256_and_si256(a, b);
    }
    Vectorized<BFloat16> inline operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return _mm256_or_si256(a, b);
    }
    Vectorized<BFloat16> inline operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
      return _mm256_xor_si256(a, b);
    }
    */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
impl VectorizedBfloat16 {
    
    pub fn eq(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this == other) & Vectorized<BFloat16>(1.0f);
        */
    }
    
    pub fn ne(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this != other) & Vectorized<BFloat16>(1.0f);
        */
    }
    
    pub fn gt(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this > other) & Vectorized<BFloat16>(1.0f);
        */
    }
    
    pub fn ge(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this >= other) & Vectorized<BFloat16>(1.0f);
        */
    }
    
    pub fn lt(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this < other) & Vectorized<BFloat16>(1.0f);
        */
    }
    
    pub fn le(&self, other: &Vectorized<bf16>) -> Vectorized<bf16> {
        
        todo!();
        /*
            return (*this <= other) & Vectorized<BFloat16>(1.0f);
        */
    }

    // frac. Implement this here so we can use
    // subtraction
    //
    pub fn frac(&self) -> Vectorized<bf16> {
        
        todo!();
        /*
            return *this - this->trunc();
        */
    }
}

/**
  | Implements the IEEE 754 201X `maximum`
  | operation, which propagates NaN if either input
  | is a NaN.
  |
  */
#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn maximum(
        a: &Vectorized<bf16>,
        b: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 b_lo, b_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(b), b_lo, b_hi);
      auto max_lo = _mm256_max_ps(a_lo, b_lo);
      auto max_hi = _mm256_max_ps(a_hi, b_hi);
      auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
      auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      auto o1 = _mm256_or_ps(max_lo, nan_lo);
      auto o2 = _mm256_or_ps(max_hi, nan_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

/**
  | Implements the IEEE 754 201X `minimum`
  | operation, which propagates NaN if either input
  | is a NaN.
  |
  */
#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn minimum(
        a: &Vectorized<bf16>,
        b: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 b_lo, b_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(b), b_lo, b_hi);
      auto min_lo = _mm256_min_ps(a_lo, b_lo);
      auto min_hi = _mm256_min_ps(a_hi, b_hi);
      auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
      auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      auto o1 = _mm256_or_ps(min_lo, nan_lo);
      auto o2 = _mm256_or_ps(min_hi, nan_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn clamp(
        a:   &Vectorized<bf16>,
        min: &Vectorized<bf16>,
        max: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 min_lo, min_hi;
      __m256 max_lo, max_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(min), min_lo, min_hi);
      cvtbf16_fp32(__m256i(max), max_lo, max_hi);
      auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
      auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
      return cvtfp32_bf16(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn clamp_max(
        a:   &Vectorized<bf16>,
        max: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 max_lo, max_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(max), max_lo, max_hi);
      auto o1 = _mm256_min_ps(max_lo, a_lo);
      auto o2 = _mm256_min_ps(max_hi, a_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn clamp_min(
        a:   &Vectorized<bf16>,
        min: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 min_lo, min_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(min), min_lo, min_hi);
      auto o1 = _mm256_max_ps(min_lo, a_lo);
      auto o2 = _mm256_max_ps(min_hi, a_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn convert(
        src: *const bf16,
        dst: *mut bf16,
        n:   i64)  {
    
    todo!();
        /*
            i64 i;
    #pragma unroll
      for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
        auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
      }
    #pragma unroll
      for (; i < n; i++) {
        dst[i] = src[i];
      }
        */
}

#[cfg(all(target_feature = "avx2",not(target_os = "windows")))]
#[inline] pub fn fmadd(
        a: &Vectorized<bf16>,
        b: &Vectorized<bf16>,
        c: &Vectorized<bf16>) -> Vectorized<bf16> {
    
    todo!();
        /*
            __m256 a_lo, a_hi;
      __m256 b_lo, b_hi;
      __m256 c_lo, c_hi;
      cvtbf16_fp32(__m256i(a), a_lo, a_hi);
      cvtbf16_fp32(__m256i(b), b_lo, b_hi);
      cvtbf16_fp32(__m256i(c), c_lo, c_hi);
      auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
      auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
      return cvtfp32_bf16(o1, o2);
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_float.h]

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub struct VectorizedFloat {
    values: __m256,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_float {

    pub type ValueType = f32;
    pub type SizeType  = i32;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedFloat {

    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 8;
        */
    }
    
    pub fn new(v: __m256) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: f32) -> Self {
    
        todo!();
        /*


            values = _mm256_set1_ps(val);
        */
    }
    
    pub fn new(
        val1: f32,
        val2: f32,
        val3: f32,
        val4: f32,
        val5: f32,
        val6: f32,
        val7: f32,
        val8: f32) -> Self {
    
        todo!();
        /*


            values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
        */
    }
    
    pub fn operator_m256(&self) -> __m256 {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
    
        todo!();
        /*
            return _mm256_blend_ps(a.values, b.values, mask);
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<f32>,
        b:    &Vectorized<f32>,
        mask: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_blendv_ps(a.values, b.values, mask.values);
        */
    }
    
    pub fn arange<step_t>(
        base: f32,
        step: Step) -> Vectorized<f32> {
        let base: f32 = base.unwrap_or(0.0);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            return Vectorized<float>(
          base,            base +     step, base + 2 * step, base + 3 * step,
          base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
        */
    }
    
    pub fn set(
        a:     &Vectorized<f32>,
        b:     &Vectorized<f32>,
        count: i64) -> Vectorized<f32> {
        let count: i64 = count.unwrap_or(size);

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
        }
        return b;
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i64) -> Vectorized<f32> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size())
          return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
        __at_align32__ float tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < size(); ++i) {
          tmp_values[i] = 0.0;
        }
        memcpy(
            tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
        return _mm256_loadu_ps(tmp_values);
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i64)  {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size()) {
          _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
        } else if (count > 0) {
          float tmp_values[size()];
          _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(float));
        }
        */
    }
    
    pub fn zero_mask(&self) -> i32 {
        
        todo!();
        /*
            // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
        __m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
        return _mm256_movemask_ps(cmp);
        */
    }
    
    pub fn isnan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
        */
    }
    
    pub fn map(&self, f: fn(_0: f32) -> f32) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        store(tmp);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn abs(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            auto mask = _mm256_set1_ps(-0.f);
        return _mm256_andnot_ps(mask, values);
        */
    }
    
    pub fn angle(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            const auto zero_vec = _mm256_set1_ps(0.f);
        const auto nan_vec = _mm256_set1_ps(NAN);
        const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
        const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
        const auto pi = _mm256_set1_ps(pi<float>);

        const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
        auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
        angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
        return angle;
        */
    }
    
    pub fn real(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn imag(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_set1_ps(0);
        */
    }
    
    pub fn conj(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn acos(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_acosf8_u10(values));
        */
    }
    
    pub fn asin(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_asinf8_u10(values));
        */
    }
    
    pub fn atan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_atanf8_u10(values));
        */
    }
    
    pub fn atan2(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_atan2f8_u10(values, b));
        */
    }
    
    pub fn copysign(&self, sign: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_copysignf8(values, sign));
        */
    }
    
    pub fn erf(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_erff8_u10(values));
        */
    }
    
    pub fn erfc(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_erfcf8_u15(values));
        */
    }
    
    pub fn erfinv(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_erfinv);
        */
    }
    
    pub fn exp(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_expf8_u10(values));
        */
    }
    
    pub fn expm1(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_expm1f8_u10(values));
        */
    }
    
    pub fn fmod(&self, q: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_fmodf8(values, q));
        */
    }
    
    pub fn log(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_logf8_u10(values));
        */
    }
    
    pub fn log2(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_log2f8_u10(values));
        */
    }
    
    pub fn log10(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_log10f8_u10(values));
        */
    }
    
    pub fn log1p(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_log1pf8_u10(values));
        */
    }
    
    pub fn sin(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_sinf8_u10(values));
        */
    }
    
    pub fn sinh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_sinhf8_u10(values));
        */
    }
    
    pub fn cos(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_cosf8_u10(values));
        */
    }
    
    pub fn cosh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_coshf8_u10(values));
        */
    }
    
    pub fn ceil(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_ceil_ps(values);
        */
    }
    
    pub fn floor(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_floor_ps(values);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_hypotf8_u05(values, b));
        */
    }
    
    pub fn i0(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_i0);
        */
    }
    
    pub fn i0e(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return map(calc_i0e);
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            __at_align32__ float tmp[size()];
        __at_align32__ float tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn neg(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_nextafterf8(values, b));
        */
    }
    
    pub fn round(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn tan(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_tanf8_u10(values));
        */
    }
    
    pub fn tanh(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_tanhf8_u10(values));
        */
    }
    
    pub fn trunc(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn lgamma(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_lgammaf8_u10(values));
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_sqrt_ps(values);
        */
    }
    
    pub fn reciprocal(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_div_ps(_mm256_set1_ps(1), values);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
        */
    }
    
    pub fn pow(&self, b: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return Vectorized<float>(Sleef_powf8_u10(values, b));
        */
    }
}

/// Comparison using the _CMP_**_OQ predicate.
///   `O`: get false if an operand is NaN
///   `Q`: do not raise if an operand is NaN
///
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialEq for VectorizedFloat {

    fn eq(&self, other: &Self) -> bool {
        _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ)
    }

    fn ne(&self, other: &Self) -> bool {
        _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialOrd for VectorizedFloat {

    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {

        todo!();

        /*
        return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Add for Vectorized<f32> {

    type Output = Self;

    fn add(self, x: Self) -> Self::Output {
        _mm256_add_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Sub for Vectorized<f32> {

    type Output = Self;

    fn sub(self, x: Self) -> Self::Output {
        _mm256_sub_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Mul for Vectorized<f32> {

    type Output = Self;

    fn mul(self, x: Self) -> Self::Output {
        _mm256_mul_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Div for Vectorized<f32> {

    type Output = Self;

    fn div(self, x: Self) -> Self::Output {
        _mm256_div_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedFloat {

    /**
      | frac. Implement this here so we can use
      | subtraction
      |
      */
    pub fn frac(&self) -> Vectorized<f32> {
        
        todo!();
        /*
            return *this - this->trunc();
        */
    }
}

/// Implements the IEEE 754 201X `maximum`
/// operation, which propagates NaN if either input
/// is a NaN.
///
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn maximum(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            Vectorized<float> max = _mm256_max_ps(a, b);
      Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      return _mm256_or_ps(max, isnan);
        */
}

/// Implements the IEEE 754 201X `minimum`
/// operation, which propagates NaN if either input
/// is a NaN.
///
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn minimum(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            Vectorized<float> min = _mm256_min_ps(a, b);
      Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      return _mm256_or_ps(min, isnan);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp(
        a:   &Vectorized<f32>,
        min: &Vectorized<f32>,
        max: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return _mm256_min_ps(max, _mm256_max_ps(min, a));
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp_max(
        a:   &Vectorized<f32>,
        max: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return _mm256_min_ps(max, a);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp_min(
        a:   &Vectorized<f32>,
        min: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return _mm256_max_ps(min, a);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitAnd<&Vectorized<f32>> for &Vectorized<f32> {

    type Output = Vectorized<f32>;
    
    fn bitand(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            return _mm256_and_ps(a, b);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitOr<&Vectorized<f32>> for &Vectorized<f32> {

    type Output = Vectorized<f32>;
    
    fn bitor(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            return _mm256_or_ps(a, b);
        */
    }
}


#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitXor<&Vectorized<f32>> for &Vectorized<f32> {
    type Output = Vectorized<f32>;

    
    fn bitxor(self, other: &Vectorized<f32>) -> Self::Output {
        todo!();
        /*
            return _mm256_xor_ps(a, b);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedFloat {
    
    pub fn eq(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this == other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn ne(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this != other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn gt(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this > other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn ge(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this >= other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn lt(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this < other) & Vectorized<float>(1.0f);
        */
    }
    
    pub fn le(&self, other: &Vectorized<f32>) -> Vectorized<f32> {
        
        todo!();
        /*
            return (*this <= other) & Vectorized<float>(1.0f);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn convert(
        src: *const f32,
        dst: *mut f32,
        n:   i64)  {
    
    todo!();
        /*
            i64 i;
    #pragma unroll
      for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
        _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
      }
    #pragma unroll
      for (; i < n; i++) {
        dst[i] = src[i];
      }
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
#[inline] pub fn fmadd(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>,
        c: &Vectorized<f32>) -> Vectorized<f32> {
    
    todo!();
        /*
            return _mm256_fmadd_ps(a, b, c);
        */
}

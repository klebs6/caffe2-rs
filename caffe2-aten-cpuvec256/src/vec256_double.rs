crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_double.h]

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub struct VectorizedDouble {
    values: __m256d,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_double {

    pub type ValueType = f64;
    pub type SizeType  = i32;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedDouble {
    
    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn new(v: __m256d) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: f64) -> Self {
    
        todo!();
        /*


            values = _mm256_set1_pd(val);
        */
    }
    
    pub fn new(
        val1: f64,
        val2: f64,
        val3: f64,
        val4: f64) -> Self {
    
        todo!();
        /*


            values = _mm256_setr_pd(val1, val2, val3, val4);
        */
    }
    
    pub fn operator_m_256d(&self) -> __m256d {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<f64>,
        b: &Vectorized<f64>) -> Vectorized<f64> {
    
        todo!();
        /*
            return _mm256_blend_pd(a.values, b.values, mask);
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<f64>,
        b:    &Vectorized<f64>,
        mask: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_blendv_pd(a.values, b.values, mask.values);
        */
    }
    
    pub fn arange<step_t>(
        base: f64,
        step: Step) -> Vectorized<f64> {
        let base: f64 = base.unwrap_or(0.0);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
        */
    }
    
    pub fn set(
        a:     &Vectorized<f64>,
        b:     &Vectorized<f64>,
        count: i64) -> Vectorized<f64> {
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
        }
        return b;
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i64) -> Vectorized<f64> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size())
          return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

        __at_align32__ double tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < size(); ++i) {
          tmp_values[i] = 0.0;
        }
        memcpy(
            tmp_values,
            reinterpret_cast<const double*>(ptr),
            count * sizeof(double));
        return _mm256_load_pd(tmp_values);
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i32)  {
        let count: i32 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size()) {
          _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
        } else if (count > 0) {
          double tmp_values[size()];
          _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(double));
        }
        */
    }
    
    pub fn zero_mask(&self) -> i32 {
        
        todo!();
        /*
            // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
        __m256d cmp = _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
        return _mm256_movemask_pd(cmp);
        */
    }
    
    pub fn isnan(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_UNORD_Q);
        */
    }
    
    pub fn map(&self, f: fn(_0: f64) -> f64) -> Vectorized<f64> {
        
        todo!();
        /*
            __at_align32__ double tmp[size()];
        store(tmp);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn abs(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            auto mask = _mm256_set1_pd(-0.f);
        return _mm256_andnot_pd(mask, values);
        */
    }
    
    pub fn angle(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            const auto zero_vec = _mm256_set1_pd(0.f);
        const auto nan_vec = _mm256_set1_pd(NAN);
        const auto not_nan_mask = _mm256_cmp_pd(values, values, _CMP_EQ_OQ);
        const auto nan_mask = _mm256_cmp_pd(not_nan_mask, zero_vec, _CMP_EQ_OQ);
        const auto pi = _mm256_set1_pd(pi<double>);

        const auto neg_mask = _mm256_cmp_pd(values, zero_vec, _CMP_LT_OQ);
        auto angle = _mm256_blendv_pd(zero_vec, pi, neg_mask);
        angle = _mm256_blendv_pd(angle, nan_vec, nan_mask);
        return angle;
        */
    }
    
    pub fn real(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn imag(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_set1_pd(0);
        */
    }
    
    pub fn conj(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return *this;
        */
    }
    
    pub fn acos(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_acosd4_u10(values));
        */
    }
    
    pub fn asin(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_asind4_u10(values));
        */
    }
    
    pub fn atan(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_atand4_u10(values));
        */
    }
    
    pub fn atan2(&self, b: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_atan2d4_u10(values, b));
        */
    }
    
    pub fn copysign(&self, sign: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_copysignd4(values, sign));
        */
    }
    
    pub fn erf(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_erfd4_u10(values));
        */
    }
    
    pub fn erfc(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_erfcd4_u15(values));
        */
    }
    
    pub fn erfinv(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return map(calc_erfinv);
        */
    }
    
    pub fn exp(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_expd4_u10(values));
        */
    }
    
    pub fn expm1(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_expm1d4_u10(values));
        */
    }
    
    pub fn fmod(&self, q: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_fmodd4(values, q));
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_hypotd4_u05(values, b));
        */
    }
    
    pub fn i0(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return map(calc_i0);
        */
    }
    
    pub fn i0e(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return map(calc_i0e);
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            __at_align32__ double tmp[size()];
        __at_align32__ double tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            __at_align32__ double tmp[size()];
        __at_align32__ double tmp_x[size()];
        store(tmp);
        x.store(tmp_x);
        for (i64 i = 0; i < size(); i++) {
          tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn log(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_logd4_u10(values));
        */
    }
    
    pub fn log2(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_log2d4_u10(values));
        */
    }
    
    pub fn log10(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_log10d4_u10(values));
        */
    }
    
    pub fn log1p(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_log1pd4_u10(values));
        */
    }
    
    pub fn sin(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_sind4_u10(values));
        */
    }
    
    pub fn sinh(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_sinhd4_u10(values));
        */
    }
    
    pub fn cos(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_cosd4_u10(values));
        */
    }
    
    pub fn cosh(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_coshd4_u10(values));
        */
    }
    
    pub fn ceil(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_ceil_pd(values);
        */
    }
    
    pub fn floor(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_floor_pd(values);
        */
    }
    
    pub fn neg(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_nextafterd4(values, b));
        */
    }
    
    pub fn round(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn tan(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_tand4_u10(values));
        */
    }
    
    pub fn tanh(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_tanhd4_u10(values));
        */
    }
    
    pub fn trunc(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn lgamma(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_lgammad4_u10(values));
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_sqrt_pd(values);
        */
    }
    
    pub fn reciprocal(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_div_pd(_mm256_set1_pd(1), values);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<f64> {
        
        todo!();
        /*
            return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
        */
    }
    
    pub fn pow(&self, b: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return Vectorized<double>(Sleef_powd4_u10(values, b));
        */
    }
}

/// Comparison using the _CMP_**_OQ predicate.
///   `O`: get false if an operand is NaN
///   `Q`: do not raise if an operand is NaN
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialEq for VectorizedDouble {

    fn eq(&self, other: &Self) -> bool {

        todo!();

        /*
        _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ)
        */
    }

    fn ne(&self, other: &Self) -> bool {

        todo!();

        /*
        _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ)
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialOrd for VectorizedDouble {

    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering> {

        todo!();

        /*
        _mm256_cmp_pd(values, other.values, _CMP_LT_OQ)
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Add for VectorizedDouble {

    type Output = Self;

    fn add(self, x: Self) -> Self::Output {
        _mm256_add_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Sub for VectorizedDouble {

    type Output = Self;

    fn sub(self, x: Self) -> Self::Output {
        _mm256_sub_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Mul for VectorizedDouble {

    type Output = Self;

    fn mul(self, x: Self) -> Self::Output {
        _mm256_mul_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Div for VectorizedDouble {

    type Output = Self;

    fn div(self, x: Self) -> Self::Output {
        _mm256_div_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedDouble {
    
    /**
      | frac. Implement this here so we can use
      | subtraction.
      |
      */
    pub fn frac(&self) -> Vectorized<f64> {
        
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
        a: &Vectorized<f64>,
        b: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            Vectorized<double> max = _mm256_max_pd(a, b);
      Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      return _mm256_or_pd(max, isnan);
        */
}

/// Implements the IEEE 754 201X `minimum`
/// operation, which propagates NaN if either
/// input is a NaN.
///
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn minimum(
    a: &Vectorized<f64>,
    b: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            Vectorized<double> min = _mm256_min_pd(a, b);
      Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
      // Exploit the fact that all-ones is a NaN.
      return _mm256_or_pd(min, isnan);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp(
        a:   &Vectorized<f64>,
        min: &Vectorized<f64>,
        max: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            return _mm256_min_pd(max, _mm256_max_pd(min, a));
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp_min(
        a:   &Vectorized<f64>,
        min: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            return _mm256_max_pd(min, a);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn clamp_max(
        a:   &Vectorized<f64>,
        max: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            return _mm256_min_pd(max, a);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitAnd<&Vectorized<f64>> for &Vectorized<f64> {
    type Output = Vectorized<f64>;
    
    fn bitand(self, other: &Vectorized<f64>) -> Self::Output {
        todo!();
        /*
            return _mm256_and_pd(a, b);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitOr<&Vectorized<f64>> for &Vectorized<f64> {
    type Output = Vectorized<f64>;

    
    fn bitor(self, other: &Vectorized<f64>) -> Self::Output {
        todo!();
        /*
            return _mm256_or_pd(a, b);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitXor<&Vectorized<f64>> for &Vectorized<f64> {
    type Output = Vectorized<f64>;

    
    fn bitxor(self, other: &Vectorized<f64>) -> Self::Output {
        todo!();
        /*
            return _mm256_xor_pd(a, b);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedDouble {
    
    pub fn eq(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this == other) & Vectorized<double>(1.0);
        */
    }
    
    pub fn ne(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this != other) & Vectorized<double>(1.0);
        */
    }
    
    pub fn gt(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this > other) & Vectorized<double>(1.0);
        */
    }
    
    pub fn ge(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this >= other) & Vectorized<double>(1.0);
        */
    }
    
    pub fn lt(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this < other) & Vectorized<double>(1.0);
        */
    }
    
    pub fn le(&self, other: &Vectorized<f64>) -> Vectorized<f64> {
        
        todo!();
        /*
            return (*this <= other) & Vectorized<double>(1.0);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn convert(
        src: *const f64,
        dst: *mut f64,
        n:   i64)  {
    
    todo!();
        /*
            i64 i;
    #pragma unroll
      for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
        _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
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
        a: &Vectorized<f64>,
        b: &Vectorized<f64>,
        c: &Vectorized<f64>) -> Vectorized<f64> {
    
    todo!();
        /*
            return _mm256_fmadd_pd(a, b, c);
        */
}

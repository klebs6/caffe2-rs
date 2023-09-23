//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_complex_double.h]

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[derive(Default)]
pub struct VectorizedComplexDouble {
    values: __m256d,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_complex_double {

    use super::*;

    pub type ValueType = Complex<f64>;
    pub type SizeType  = i32;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedComplexDouble {

    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 2;
        */
    }
    
    pub fn new(v: __m256d) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: Complex<f64>) -> Self {
    
        todo!();
        /*


            double real_value = val.real();
        double imag_value = val.imag();
        values = _mm256_setr_pd(real_value, imag_value,
                                real_value, imag_value);
        */
    }
    
    pub fn new(
        val1: Complex<f64>,
        val2: Complex<f64>) -> Self {
    
        todo!();
        /*


            values = _mm256_setr_pd(val1.real(), val1.imag(),
                                val2.real(), val2.imag());
        */
    }
    
    pub fn operator_m_256d(&self) -> __m256d {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<Complex<f64>>,
        b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
    
        todo!();
        /*
            // convert complex<V> index mask to V index mask: xy -> xxyy
        switch (mask) {
          case 0:
            return a;
          case 1:
            return _mm256_blend_pd(a.values, b.values, 0x03);
          case 2:
            return _mm256_blend_pd(a.values, b.values, 0x0c);
        }
        return b;
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<Complex<f64>>,
        b:    &Vectorized<Complex<f64>>,
        mask: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            // convert complex<V> index mask to V index mask: xy -> xxyy
        auto mask_ = _mm256_unpacklo_pd(mask.values, mask.values);
        return _mm256_blendv_pd(a.values, b.values, mask_);
        */
    }
    
    pub fn arange<step_t>(
        base: Complex<f64>,
        step: Step) -> Vectorized<Complex<f64>> {
        let base: Complex<f64> = base.unwrap_or(0.);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            return Vectorized<complex<double>>(base,
                                            base + step);
        */
    }
    
    pub fn set(
        a:     &Vectorized<Complex<f64>>,
        b:     &Vectorized<Complex<f64>>,
        count: i64) -> Vectorized<Complex<f64>> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            switch (count) {
          case 0:
            return a;
          case 1:
            return blend<1>(a, b);
        }
        return b;
        */
    }
    
    pub fn loadu(
        ptr:   *const void,
        count: i64) -> Vectorized<Complex<f64>> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size())
          return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

        __at_align32__ double tmp_values[2*size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < 2*size(); ++i) {
          tmp_values[i] = 0.0;
        }
        memcpy(
            tmp_values,
            reinterpret_cast<const double*>(ptr),
            count * sizeof(complex<double>));
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
          double tmp_values[2*size()];
          _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(complex<double>));
        }
        */
    }
    
    pub fn map(&self, f: fn(_0: &Complex<f64>) -> Complex<f64>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            __at_align32__ complex<double> tmp[size()];
        store(tmp);
        for (int i = 0; i < size(); i++) {
          tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn abs_2(&self) -> __m256d {
        
        todo!();
        /*
            auto val_2 = _mm256_mul_pd(values, values);     // a*a     b*b
        return _mm256_hadd_pd(val_2, val_2);            // a*a+b*b a*a+b*b
        */
    }
    
    pub fn abs(&self) -> __m256d {
        
        todo!();
        /*
            return _mm256_sqrt_pd(abs_2_());                // abs     abs
        */
    }
    
    pub fn abs(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                         0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
        return _mm256_and_pd(abs_(), real_mask);        // abs     0
        */
    }
    
    pub fn angle(&self) -> __m256d {
        
        todo!();
        /*
            //angle = atan2(b/a)
        auto b_a = _mm256_permute_pd(values, 0x05);     // b        a
        return Sleef_atan2d4_u10(values, b_a);          // 90-angle angle
        */
    }
    
    pub fn angle(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                         0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
        auto angle = _mm256_permute_pd(angle_(), 0x05); // angle    90-angle
        return _mm256_and_pd(angle, real_mask);         // angle    0
        */
    }
    
    pub fn sgn(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            auto abs = abs_();
        auto zero = _mm256_setzero_pd();
        auto mask = _mm256_cmp_pd(abs, zero, _CMP_EQ_OQ);
        auto abs_val = Vectorized(abs);

        auto div = values / abs_val.values;       // x / abs(x)

        return blendv(div, zero, mask);
        */
    }
    
    pub fn real(&self) -> __m256d {
        
        todo!();
        /*
            const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                         0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
        return _mm256_and_pd(values, real_mask);
        */
    }
    
    pub fn real(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return real_();
        */
    }
    
    pub fn imag(&self) -> __m256d {
        
        todo!();
        /*
            const __m256d imag_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                         0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
        return _mm256_and_pd(values, imag_mask);
        */
    }
    
    pub fn imag(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return _mm256_permute_pd(imag_(), 0x05);           //b        a
        */
    }
    
    pub fn conj(&self) -> __m256d {
        
        todo!();
        /*
            const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        return _mm256_xor_pd(values, sign_mask);           // a       -b
        */
    }
    
    pub fn conj(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return conj_();
        */
    }
    
    pub fn log(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            // Most trigonomic ops use the log() op to improve complex number performance.
        return map(log);
        */
    }
    
    pub fn log2(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            const __m256d log2_ = _mm256_set1_pd(log(2));
        return _mm256_div_pd(log(), log2_);
        */
    }
    
    pub fn log10(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            const __m256d log10_ = _mm256_set1_pd(log(10));
        return _mm256_div_pd(log(), log10_);
        */
    }
    
    pub fn log1p(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn asin(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            // asin(x)
        // = -i*ln(iz + sqrt(1 -z^2))
        // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
        // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
        const __m256d one = _mm256_set1_pd(1);

        auto conj = conj_();
        auto b_a = _mm256_permute_pd(conj, 0x05);                         //-b        a
        auto ab = _mm256_mul_pd(conj, b_a);                               //-ab       -ab
        auto im = _mm256_add_pd(ab, ab);                                  //-2ab      -2ab

        auto val_2 = _mm256_mul_pd(values, values);                       // a*a      b*b
        auto re = _mm256_hsub_pd(val_2, _mm256_permute_pd(val_2, 0x05));  // a*a-b*b  b*b-a*a
        re = _mm256_sub_pd(one, re);

        auto root = Vectorized(_mm256_blend_pd(re, im, 0x0A)).sqrt();         //sqrt(re + i*im)
        auto ln = Vectorized(_mm256_add_pd(b_a, root)).log();                 //ln(iz + sqrt())
        return Vectorized(_mm256_permute_pd(ln.values, 0x05)).conj();         //-i*ln()
        */
    }
    
    pub fn acos(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            // acos(x) = pi/2 - asin(x)
        constexpr auto pi_2d = pi<double> / 2;
        const __m256d pi_2 = _mm256_setr_pd(pi_2d, 0.0, pi_2d, 0.0);
        return _mm256_sub_pd(pi_2, asin());
        */
    }
    
    pub fn atan2(&self, b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn erf(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn erfc(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn exp(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            //exp(a + bi)
        // = exp(a)*(cos(b) + sin(b)i)
        auto exp = Sleef_expd4_u10(values);                               //exp(a)           exp(b)
        exp = _mm256_blend_pd(exp, _mm256_permute_pd(exp, 0x05), 0x0A);   //exp(a)           exp(a)

        auto sin_cos = Sleef_sincosd4_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
        auto cos_sin = _mm256_blend_pd(_mm256_permute_pd(sin_cos.y, 0x05),
                                       sin_cos.x, 0x0A);                  //cos(b)           sin(b)
        return _mm256_mul_pd(exp, cos_sin);
        */
    }
    
    pub fn expm1(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn sin(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(sin);
        */
    }
    
    pub fn sinh(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(sinh);
        */
    }
    
    pub fn cos(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(cos);
        */
    }
    
    pub fn cosh(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(cosh);
        */
    }
    
    pub fn ceil(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return _mm256_ceil_pd(values);
        */
    }
    
    pub fn floor(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return _mm256_floor_pd(values);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn neg(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            auto zero = _mm256_setzero_pd();
        return _mm256_sub_pd(zero, values);
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn round(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn tan(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(tan);
        */
    }
    
    pub fn tanh(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(tanh);
        */
    }
    
    pub fn trunc(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return map(sqrt);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            return sqrt().reciprocal();
        */
    }
    
    pub fn pow(&self, exp: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            __at_align32__ complex<double> x_tmp[size()];
        __at_align32__ complex<double> y_tmp[size()];
        store(x_tmp);
        exp.store(y_tmp);
        for (int i = 0; i < size(); i++) {
          x_tmp[i] = pow(x_tmp[i], y_tmp[i]);
        }
        return loadu(x_tmp);
        */
    }
}

/// Comparison using the _CMP_**_OQ predicate.
///   `O`: get false if an operand is NaN
///   `Q`: do not raise if an operand is NaN
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialEq for VectorizedComplexDouble {

    fn eq(&self, other: Self) -> bool {

        todo!();

        /*
        _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ)
        */
    }

    fn ne(&self, other: Self) -> bool {

        todo!();

        /*
        _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ)
        */
    }

    /*
       Vectorized<complex<double>> operator<(const Vectorized<complex<double>>& other) const {
           TORCH_CHECK(false, "not supported for complex numbers");
       }
       */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Add<VectorizedComplexDouble> for VectorizedComplexDouble {

    type Output = Self;

    fn add(self, rhs: Rhs) -> Self::Output {
        _mm256_add_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Sub<VectorizedComplexDouble> for VectorizedComplexDouble {

    type Output = Self;

    fn sub(self, rhs: Rhs) -> Self::Output {
        _mm256_sub_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Mul<VectorizedComplexDouble> for VectorizedComplexDouble {

    type Output = Self;

    fn mul(self, rhs: Rhs) -> Self::Output {

        todo!();

        /*
        //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
        const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        auto ac_bd = _mm256_mul_pd(a, b);         //ac       bd

        auto d_c = _mm256_permute_pd(b, 0x05);    //d        c
        d_c = _mm256_xor_pd(sign_mask, d_c);      //d       -c
        auto ad_bc = _mm256_mul_pd(a, d_c);       //ad      -bc

        auto ret = _mm256_hsub_pd(ac_bd, ad_bc);  //ac - bd  ad + bc
        return ret;
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Div<VectorizedComplexDouble> for VectorizedComplexDouble {

    type Output = Self;

    fn div(self, rhs: Rhs) -> Self::Output {

        todo!();

        /*
          //re + im*i = (a + bi)  / (c + di)
          //re = (ac + bd)/abs_2()
          //im = (bc - ad)/abs_2()
          const __m256d sign_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
          auto ac_bd = _mm256_mul_pd(a, b);         //ac       bd

          auto d_c = _mm256_permute_pd(b, 0x05);    //d        c
          d_c = _mm256_xor_pd(sign_mask, d_c);      //-d       c
          auto ad_bc = _mm256_mul_pd(a, d_c);       //-ad      bc

          auto re_im = _mm256_hadd_pd(ac_bd, ad_bc);//ac + bd  bc - ad
          return _mm256_div_pd(re_im, b.abs_2_());
        */
    }
}


#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedComplexDouble {
    
    /// reciprocal. Implement this here so we can
    /// use multiplication.
    ///
    pub fn reciprocal(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            //re + im*i = (a + bi)  / (c + di)
      //re = (ac + bd)/abs_2() = c/abs_2()
      //im = (bc - ad)/abs_2() = d/abs_2()
      const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
      auto c_d = _mm256_xor_pd(sign_mask, values);    //c       -d
      return _mm256_div_pd(c_d, abs_2_());
        */
    }
    
    pub fn atan(&self) -> Vectorized<Complex<f64>> {
        
        todo!();
        /*
            // atan(x) = i/2 * ln((i + z)/(i - z))
      const __m256d i = _mm256_setr_pd(0.0, 1.0, 0.0, 1.0);
      const Vectorized i_half = _mm256_setr_pd(0.0, 0.5, 0.0, 0.5);

      auto sum = Vectorized(_mm256_add_pd(i, values));                      // a        1+b
      auto sub = Vectorized(_mm256_sub_pd(i, values));                      // -a       1-b
      auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
      return i_half*ln;                                                 // i/2*ln()
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn maximum(
        a: &Vectorized<Complex<f64>>,
        b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
    
    todo!();
        /*
            auto abs_a = a.abs_2_();
      auto abs_b = b.abs_2_();
      auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_LT_OQ);
      auto max = _mm256_blendv_pd(a, b, mask);
      // Exploit the fact that all-ones is a NaN.
      auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
      return _mm256_or_pd(max, isnan);
        */
}


#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn minimum(
        a: &Vectorized<Complex<f64>>,
        b: &Vectorized<Complex<f64>>) -> Vectorized<Complex<f64>> {
    
    todo!();
        /*
            auto abs_a = a.abs_2_();
      auto abs_b = b.abs_2_();
      auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_GT_OQ);
      auto min = _mm256_blendv_pd(a, b, mask);
      // Exploit the fact that all-ones is a NaN.
      auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
      return _mm256_or_pd(min, isnan);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitAnd for VectorizedComplexDouble {
    type Output = Self;

    fn bitand(self, x: Self) -> Self::Output {
        _mm256_and_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitOr for VectorizedComplexDouble {

    type Output = Self;

    fn bitor(self, x: Self) -> Self::Output {
        _mm256_or_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitXor for VectorizedComplexDouble {

    type Output = Self;

    fn bitxor(self, x: Self) -> Self::Output {
        _mm256_xor_pd(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialEq for VectorizedComplexDouble {

    fn eq(self, other: Self) -> bool {

        todo!();

        /*
        (*this == other) & Vectorized<complex<double>>(_mm256_set1_pd(1.0))
        */
    }

    fn ne(self, other: Self) -> bool {

        todo!();

        /*
        (*this != other) & Vectorized<complex<double>>(_mm256_set1_pd(1.0))
        */
    }
}

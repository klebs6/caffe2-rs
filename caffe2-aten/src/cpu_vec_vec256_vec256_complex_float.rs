//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_complex_float.h]

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub struct VectorizedComplexFloat {
    values: __m256,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_complex_float {

    use super::*;

    pub type ValueType = Complex<f32>;
    pub type SizeType  = i32;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedComplexFloat {
    
    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn new(v: __m256) -> Self {
    
        todo!();
        /*
        : values(v),

        
        */
    }
    
    pub fn new(val: Complex<f32>) -> Self {
    
        todo!();
        /*


            float real_value = val.real();
        float imag_value = val.imag();
        values = _mm256_setr_ps(real_value, imag_value,
                                real_value, imag_value,
                                real_value, imag_value,
                                real_value, imag_value
                                );
        */
    }
    
    pub fn new(
        val1: Complex<f32>,
        val2: Complex<f32>,
        val3: Complex<f32>,
        val4: Complex<f32>) -> Self {
    
        todo!();
        /*


            values = _mm256_setr_ps(val1.real(), val1.imag(),
                                val2.real(), val2.imag(),
                                val3.real(), val3.imag(),
                                val4.real(), val4.imag()
                                );
        */
    }
    
    pub fn operator_m256(&self) -> __m256 {
        
        todo!();
        /*
            return values;
        */
    }
    
    pub fn blend<const mask: i64>(
        a: &Vectorized<Complex<f32>>,
        b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
    
        todo!();
        /*
            // convert complex<V> index mask to V index mask: xy -> xxyy
        switch (mask) {
          case 0:
            return a;
          case 1:
            return _mm256_blend_ps(a.values, b.values, 0x03); //b0000 0001 = b0000 0011
          case 2:
            return _mm256_blend_ps(a.values, b.values, 0x0C); //b0000 0010 = b0000 1100
          case 3:
            return _mm256_blend_ps(a.values, b.values, 0x0F); //b0000 0011 = b0000 1111
          case 4:
            return _mm256_blend_ps(a.values, b.values, 0x30); //b0000 0100 = b0011 0000
          case 5:
            return _mm256_blend_ps(a.values, b.values, 0x33); //b0000 0101 = b0011 0011
          case 6:
            return _mm256_blend_ps(a.values, b.values, 0x3C); //b0000 0110 = b0011 1100
          case 7:
            return _mm256_blend_ps(a.values, b.values, 0x3F); //b0000 0111 = b0011 1111
          case 8:
            return _mm256_blend_ps(a.values, b.values, 0xC0); //b0000 1000 = b1100 0000
          case 9:
            return _mm256_blend_ps(a.values, b.values, 0xC3); //b0000 1001 = b1100 0011
          case 10:
            return _mm256_blend_ps(a.values, b.values, 0xCC); //b0000 1010 = b1100 1100
          case 11:
            return _mm256_blend_ps(a.values, b.values, 0xCF); //b0000 1011 = b1100 1111
          case 12:
            return _mm256_blend_ps(a.values, b.values, 0xF0); //b0000 1100 = b1111 0000
          case 13:
            return _mm256_blend_ps(a.values, b.values, 0xF3); //b0000 1101 = b1111 0011
          case 14:
            return _mm256_blend_ps(a.values, b.values, 0xFC); //b0000 1110 = b1111 1100
        }
        return b;
        */
    }
    
    pub fn blendv(
        a:    &Vectorized<Complex<f32>>,
        b:    &Vectorized<Complex<f32>>,
        mask: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            // convert complex<V> index mask to V index mask: xy -> xxyy
        auto mask_ = _mm256_unpacklo_ps(mask.values, mask.values);
        return _mm256_blendv_ps(a.values, b.values, mask_);
        */
    }
    
    pub fn arange<step_t>(
        base: Complex<f32>,
        step: Step) -> Vectorized<Complex<f32>> {
        let base: Complex<f32> = base.unwrap_or(0.0);
        let step: Step = step.unwrap_or(1);
        todo!();
        /*
            return Vectorized<complex<float>>(base,
                                            base + step,
                                            base + complex<float>(2)*step,
                                            base + complex<float>(3)*step);
        */
    }
    
    pub fn set(
        a:     &Vectorized<Complex<f32>>,
        b:     &Vectorized<Complex<f32>>,
        count: i64) -> Vectorized<Complex<f32>> {
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
        count: i64) -> Vectorized<Complex<f32>> {
        let count: i64 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size())
          return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));

        __at_align32__ float tmp_values[2*size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (auto i = 0; i < 2*size(); ++i) {
          tmp_values[i] = 0.0;
        }
        memcpy(
            tmp_values,
            reinterpret_cast<const float*>(ptr),
            count * sizeof(complex<float>));
        return _mm256_load_ps(tmp_values);
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut void,
        count: i32)  {
        let count: i32 = count.unwrap_or(size);

        todo!();
        /*
            if (count == size()) {
          _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
        } else if (count > 0) {
          float tmp_values[2*size()];
          _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
          memcpy(ptr, tmp_values, count * sizeof(complex<float>));
        }
        */
    }
    
    pub fn map(&self, f: fn(_0: &Complex<f32>) -> Complex<f32>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            __at_align32__ complex<float> tmp[size()];
        store(tmp);
        for (int i = 0; i < size(); i++) {
          tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
        */
    }
    
    pub fn abs_2(&self) -> __m256 {
        
        todo!();
        /*
            auto val_2 = _mm256_mul_ps(values, values);     // a*a     b*b
        auto ret = _mm256_hadd_ps(val_2, val_2);        // a*a+b*b a*a+b*b
        return _mm256_permute_ps(ret, 0xD8);
        */
    }
    
    pub fn abs(&self) -> __m256 {
        
        todo!();
        /*
            return _mm256_sqrt_ps(abs_2_());                // abs     abs
        */
    }
    
    pub fn abs(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
        return _mm256_and_ps(abs_(), real_mask);        // abs     0
        */
    }
    
    pub fn angle(&self) -> __m256 {
        
        todo!();
        /*
            //angle = atan2(b/a)
        auto b_a = _mm256_permute_ps(values, 0xB1);     // b        a
        return Sleef_atan2f8_u10(values, b_a);          // 90-angle angle
        */
    }
    
    pub fn angle(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
        auto angle = _mm256_permute_ps(angle_(), 0xB1); // angle    90-angle
        return _mm256_and_ps(angle, real_mask);         // angle    0
        */
    }
    
    pub fn sgn(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            auto abs = abs_();
        auto zero = _mm256_setzero_ps();
        auto mask = _mm256_cmp_ps(abs, zero, _CMP_EQ_OQ);
        auto abs_val = Vectorized(abs);

        auto div = values / abs_val.values;       // x / abs(x)

        return _mm256_blendv_ps(div, zero, mask);
        */
    }
    
    pub fn real(&self) -> __m256 {
        
        todo!();
        /*
            const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
        return _mm256_and_ps(values, real_mask);
        */
    }
    
    pub fn real(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return real_();
        */
    }
    
    pub fn imag(&self) -> __m256 {
        
        todo!();
        /*
            const __m256 imag_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                       0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
        return _mm256_and_ps(values, imag_mask);
        */
    }
    
    pub fn imag(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return _mm256_permute_ps(imag_(), 0xB1);        //b        a
        */
    }
    
    pub fn conj(&self) -> __m256 {
        
        todo!();
        /*
            const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        return _mm256_xor_ps(values, sign_mask);        // a       -b
        */
    }
    
    pub fn conj(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return conj_();
        */
    }
    
    pub fn log(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            // Most trigonomic ops use the log() op to improve complex number performance.
        return map(log);
        */
    }
    
    pub fn log2(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            const __m256 log2_ = _mm256_set1_ps(log(2));
        return _mm256_div_ps(log(), log2_);
        */
    }
    
    pub fn log10(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            const __m256 log10_ = _mm256_set1_ps(log(10));
        return _mm256_div_ps(log(), log10_);
        */
    }
    
    pub fn log1p(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn asin(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            // asin(x)
        // = -i*ln(iz + sqrt(1 -z^2))
        // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
        // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
        const __m256 one = _mm256_set1_ps(1);

        auto conj = conj_();
        auto b_a = _mm256_permute_ps(conj, 0xB1);                         //-b        a
        auto ab = _mm256_mul_ps(conj, b_a);                               //-ab       -ab
        auto im = _mm256_add_ps(ab, ab);                                  //-2ab      -2ab

        auto val_2 = _mm256_mul_ps(values, values);                       // a*a      b*b
        auto re = _mm256_hsub_ps(val_2, _mm256_permute_ps(val_2, 0xB1));  // a*a-b*b  b*b-a*a
        re = _mm256_permute_ps(re, 0xD8);
        re = _mm256_sub_ps(one, re);

        auto root = Vectorized(_mm256_blend_ps(re, im, 0xAA)).sqrt();         //sqrt(re + i*im)
        auto ln = Vectorized(_mm256_add_ps(b_a, root)).log();                 //ln(iz + sqrt())
        return Vectorized(_mm256_permute_ps(ln.values, 0xB1)).conj();         //-i*ln()
        */
    }
    
    pub fn acos(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(acos);
        */
    }
    
    pub fn atan2(&self, b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn erf(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn erfc(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn exp(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            //exp(a + bi)
        // = exp(a)*(cos(b) + sin(b)i)
        auto exp = Sleef_expf8_u10(values);                               //exp(a)           exp(b)
        exp = _mm256_blend_ps(exp, _mm256_permute_ps(exp, 0xB1), 0xAA);   //exp(a)           exp(a)

        auto sin_cos = Sleef_sincosf8_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
        auto cos_sin = _mm256_blend_ps(_mm256_permute_ps(sin_cos.y, 0xB1),
                                       sin_cos.x, 0xAA);                  //cos(b)           sin(b)
        return _mm256_mul_ps(exp, cos_sin);
        */
    }
    
    pub fn expm1(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn sin(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(sin);
        */
    }
    
    pub fn sinh(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(sinh);
        */
    }
    
    pub fn cos(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(cos);
        */
    }
    
    pub fn cosh(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(cosh);
        */
    }
    
    pub fn ceil(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return _mm256_ceil_ps(values);
        */
    }
    
    pub fn floor(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return _mm256_floor_ps(values);
        */
    }
    
    pub fn hypot(&self, b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn igamma(&self, x: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn igammac(&self, x: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn neg(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            auto zero = _mm256_setzero_ps();
        return _mm256_sub_ps(zero, values);
        */
    }
    
    pub fn nextafter(&self, b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            AT_ERROR("not supported for complex numbers");
        */
    }
    
    pub fn round(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn tan(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(tan);
        */
    }
    
    pub fn tanh(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(tanh);
        */
    }
    
    pub fn trunc(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        */
    }
    
    pub fn sqrt(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return map(sqrt);
        */
    }
    
    pub fn rsqrt(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            return sqrt().reciprocal();
        */
    }
    
    pub fn pow(&self, exp: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            __at_align32__ complex<float> x_tmp[size()];
        __at_align32__ complex<float> y_tmp[size()];
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
impl PartialEq for VectorizedComplexFloat {

    fn eq(self, x: Self) -> bool {

        todo!();

        /*
        _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ)
        */
    }

    fn ne(self, x: Self) -> bool {

        todo!();

        /*
        _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ)
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Add for VectorizedComplexFloat {

    type Output = Self;

    fn add(self, x: Self) -> Self::Output {
        _mm256_add_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Sub for VectorizedComplexFloat {

    type Output = Self;

    fn sub(self, x: Self) -> Self::Output {
        _mm256_sub_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Mul for VectorizedComplexFloat {

    type Output = Self;

    fn mul(self, x: Self) -> Self::Output {

        todo!();

        /*
          //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
          const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
          auto ac_bd = _mm256_mul_ps(a, b);         //ac       bd

          auto d_c = _mm256_permute_ps(b, 0xB1);    //d        c
          d_c = _mm256_xor_ps(sign_mask, d_c);      //d       -c
          auto ad_bc = _mm256_mul_ps(a, d_c);       //ad      -bc

          auto ret = _mm256_hsub_ps(ac_bd, ad_bc);  //ac - bd  ad + bc
          ret = _mm256_permute_ps(ret, 0xD8);
          return ret;
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Div for VectorizedComplexFloat {

    type Output = Self;

    fn div(self, x: Self) -> Self::Output {

        todo!();

        /*
          //re + im*i = (a + bi)  / (c + di)
          //re = (ac + bd)/abs_2()
          //im = (bc - ad)/abs_2()
          const __m256 sign_mask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
          auto ac_bd = _mm256_mul_ps(a, b);         //ac       bd

          auto d_c = _mm256_permute_ps(b, 0xB1);    //d        c
          d_c = _mm256_xor_ps(sign_mask, d_c);      //-d       c
          auto ad_bc = _mm256_mul_ps(a, d_c);       //-ad      bc

          auto re_im = _mm256_hadd_ps(ac_bd, ad_bc);//ac + bd  bc - ad
          re_im = _mm256_permute_ps(re_im, 0xD8);
          return _mm256_div_ps(re_im, b.abs_2_());
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedComplexFloat {
    
    /**
      | reciprocal. Implement this here so
      | we can use multiplication.
      |
      */
    pub fn reciprocal(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            //re + im*i = (a + bi)  / (c + di)
      //re = (ac + bd)/abs_2() = c/abs_2()
      //im = (bc - ad)/abs_2() = d/abs_2()
      const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
      auto c_d = _mm256_xor_ps(sign_mask, values);    //c       -d
      return _mm256_div_ps(c_d, abs_2_());
        */
    }
    
    pub fn atan(&self) -> Vectorized<Complex<f32>> {
        
        todo!();
        /*
            // atan(x) = i/2 * ln((i + z)/(i - z))
      const __m256 i = _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
      const Vectorized i_half = _mm256_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

      auto sum = Vectorized(_mm256_add_ps(i, values));                      // a        1+b
      auto sub = Vectorized(_mm256_sub_ps(i, values));                      // -a       1-b
      auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
      return i_half*ln;                                                 // i/2*ln()
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn maximum(
        a: &Vectorized<Complex<f32>>,
        b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
    
    todo!();
        /*
            auto abs_a = a.abs_2_();
      auto abs_b = b.abs_2_();
      auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
      auto max = _mm256_blendv_ps(a, b, mask);
      // Exploit the fact that all-ones is a NaN.
      auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
      return _mm256_or_ps(max, isnan);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn minimum(
        a: &Vectorized<Complex<f32>>,
        b: &Vectorized<Complex<f32>>) -> Vectorized<Complex<f32>> {
    
    todo!();
        /*
            auto abs_a = a.abs_2_();
      auto abs_b = b.abs_2_();
      auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
      auto min = _mm256_blendv_ps(a, b, mask);
      // Exploit the fact that all-ones is a NaN.
      auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
      return _mm256_or_ps(min, isnan);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitAnd for VectorizedComplexFloat {

    type Output = Self;

    fn bitand(self, x: Self) -> Self::Output {
        _mm256_and_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitOr for VectorizedComplexFloat {

    type Output = Self;

    fn bitor(self, x: Self) -> Self::Output {
        _mm256_or_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl BitXor for VectorizedComplexFloat {

    type Output = Self;

    fn bitxor(self, x: Self) -> Self::Output {
        _mm256_xor_ps(a, b)
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl PartialEq for VectorizedComplexFloat {

    fn eq(self, x: Self) -> bool {
        todo!();

        /*
            (*this == other) & Vectorized<complex<float>>(_mm256_set1_ps(1.0f))
        */
    }

    fn ne(self, x: Self) -> bool {

        todo!();

        /*
          return (*this != other) & Vectorized<complex<float>>(_mm256_set1_ps(1.0f));
        */
    }
}

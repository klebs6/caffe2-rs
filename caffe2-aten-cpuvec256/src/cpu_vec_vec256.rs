/*!
  | DO NOT DEFINE STATIC DATA IN THIS HEADER!
  | 
  | See Note [Do not compile initializers
  | with AVX]
  |
  | Note [Acceptable use of anonymous namespace in
  | header]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Yes you saw right, this is an anonymous
  | namespace in a header.  This header, and all of
  | its subheaders, REQUIRE their code to be
  | entirely inlined into the compilation unit that
  | uses them.
  |
  | It's important that these functions have
  | internal linkage so that kernels for different
  | architectures don't get combined during
  | linking.
  |
  | It's sufficient to label functions "static",
  | but class methods must be an unnamed namespace
  | to have internal linkage (since static means
  | something different in the context of classes).
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256.h]

impl fmt::Display for &mut std::io::BufWriter {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << val.val_;
         return stream;
        */
    }
}

impl fmt::Display for &mut std::io::BufWriter {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << static_cast<int>(val.val_);
         return stream;
        */
    }
}

impl fmt::Display for &mut std::io::BufWriter {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << static_cast<unsigned int>(val.val_);
         return stream;
        */
    }
}

impl fmt::Display for &mut std::io::BufWriter {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            T buf[Vectorized<T>::size()];
      vec.store(buf);
      stream << "vec[";
      for (int i = 0; i != Vectorized<T>::size(); i++) {
        if (i != 0) {
          stream << ", ";
        }
        stream << buf[i];
      }
      stream << "]";
      return stream;
        */
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn cast(src: &Vectorized<f64>) -> Vectorized<f32> {
    
    todo!();
        /*
            return _mm256_castpd_ps(src);
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn cast(src: &Vectorized<f32>) -> Vectorized<f64> {
    
    todo!();
        /*
            return _mm256_castps_pd(src);
        */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[macro_export] macro_rules! define_float_int_cast {
    ($int_t:ident, $float_t:ident, $float_ch:ident) => {
        /*
        
        template<>                                                         
        inline  Vectorized<int_t> cast<int_t, float_t>(const Vectorized<float_t>& src) {   
          return _mm256_castp ## float_ch ## _si256(src);                  
        }                                                                  
        template<>                                                         
        inline Vectorized<float_t> cast<float_t, int_t>(const Vectorized<int_t>& src) {   
          return _mm256_castsi256_p ## float_ch (src);                     
        }

        DEFINE_FLOAT_INT_CAST(i64, double, d)
        DEFINE_FLOAT_INT_CAST(i32, double, d)
        DEFINE_FLOAT_INT_CAST(i16, double, d)
        DEFINE_FLOAT_INT_CAST(i64, float, s)
        DEFINE_FLOAT_INT_CAST(i32, float, s)
        DEFINE_FLOAT_INT_CAST(i16, float, s)
        */
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
lazy_static!{
    /*
    template<i64 scale = 1>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
    inline gather(const double* base_addr, const Vectorized<i64>& vindex) {
      return _mm256_i64gather_pd(base_addr, vindex, scale);
    }

    template<i64 scale = 1>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
    inline gather(const float* base_addr, const Vectorized<i32>& vindex) {
      return _mm256_i32gather_ps(base_addr, vindex, scale);
    }
    */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
lazy_static!{
    /*
    template<i64 scale = 1>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
    inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                       const Vectorized<i64>& vindex, const Vectorized<double>& mask) {
      return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
    }

    template<i64 scale = 1>
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
    inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                       const Vectorized<i32>& vindex, const Vectorized<float>& mask) {
      return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
    }
    */
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | Only works for inputs in the range: [-2^51, 2^51]
  |
  | From: https://stackoverflow.com/a/41148578
  |
  */
#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn convert_to_int_of_same_size(src: &Vectorized<f64>) -> Vectorized<i64> {
    
    todo!();
        /*
            auto x = _mm256_add_pd(src, _mm256_set1_pd(0x0018000000000000));
      return _mm256_sub_epi64(
          _mm256_castpd_si256(x),
          _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
      );
        */
}

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn convert_to_int_of_same_size(src: &Vectorized<f32>) -> Vectorized<i32> {
    
    todo!();
        /*
            return _mm256_cvttps_epi32(src);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn interleave2(
        a: &Vectorized<f64>,
        b: &Vectorized<f64>) -> (Vectorized<f64>,Vectorized<f64>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, a1, a3, a3}
      //   b = {b0, b1, b2, b3}

      // swap lanes:
      //   a_swapped = {a0, a1, b0, b1}
      //   b_swapped = {a2, a3, b2, b3}
      auto a_swapped = _mm256_permute2f128_pd(a, b, 0b0100000);  // 0, 2.   4 bits apart
      auto b_swapped = _mm256_permute2f128_pd(a, b, 0b0110001);  // 1, 3.   4 bits apart

      // group cols crossing lanes:
      //   return {a0, b0, a1, b1}
      //          {a2, b2, a3, b3}
      return make_pair(_mm256_permute4x64_pd(a_swapped, 0b11011000),  // 0, 2, 1, 3
                            _mm256_permute4x64_pd(b_swapped, 0b11011000)); // 0, 2, 1, 3
        */
}

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn interleave2(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> (Vectorized<f32>,Vectorized<f32>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
      //   b = {b0, b1, b2, b3, b4, b5, b6, b7}

      // swap lanes:
      //   a_swapped = {a0, a1, a2, a3, b0, b1, b2, b3}
      //   b_swapped = {a4, a5, a6, a7, b4, b5, b6, b7}
      // TODO: can we support caching this?
      auto a_swapped = _mm256_permute2f128_ps(a, b, 0b0100000);  // 0, 2.   4 bits apart
      auto b_swapped = _mm256_permute2f128_ps(a, b, 0b0110001);  // 1, 3.   4 bits apart

      // group cols crossing lanes:
      //   return {a0, b0, a1, b1, a2, b2, a3, b3}
      //          {a4, b4, a5, b5, a6, b6, a7, b7}
      const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      return make_pair(_mm256_permutevar8x32_ps(a_swapped, group_ctrl),
                            _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
        */
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn deinterleave2(
        a: &Vectorized<f64>,
        b: &Vectorized<f64>) -> (Vectorized<f64>,Vectorized<f64>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, b0, a1, b1}
      //   b = {a2, b2, a3, b3}

      // group cols crossing lanes:
      //   a_grouped = {a0, a1, b0, b1}
      //   b_grouped = {a2, a3, b2, b3}
      auto a_grouped = _mm256_permute4x64_pd(a, 0b11011000);  // 0, 2, 1, 3
      auto b_grouped = _mm256_permute4x64_pd(b, 0b11011000);  // 0, 2, 1, 3

      // swap lanes:
      //   return {a0, a1, a2, a3}
      //          {b0, b1, b2, b3}
      return make_pair(_mm256_permute2f128_pd(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                            _mm256_permute2f128_pd(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
        */
}



#[cfg(target_feature = "avx2")]
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline] pub fn deinterleave2(
        a: &Vectorized<f32>,
        b: &Vectorized<f32>) -> (Vectorized<f32>,Vectorized<f32>) {
    
    todo!();
        /*
            // inputs:
      //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
      //   b = {a4, b4, a5, b5, a6, b6, a7, b7}

      // group cols crossing lanes:
      //   a_grouped = {a0, a1, a2, a3, b0, b1, b2, b3}
      //   b_grouped = {a4, a5, a6, a7, b4, b5, b6, b7}
      // TODO: can we support caching this?
      const __m256i group_ctrl = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
      auto a_grouped = _mm256_permutevar8x32_ps(a, group_ctrl);
      auto b_grouped = _mm256_permutevar8x32_ps(b, group_ctrl);

      // swap lanes:
      //   return {a0, a1, a2, a3, a4, a5, a6, a7}
      //          {b0, b1, b2, b3, b4, b5, b6, b7}
      return make_pair(_mm256_permute2f128_ps(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                            _mm256_permute2f128_ps(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
        */
}

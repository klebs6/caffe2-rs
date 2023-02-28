/*!
  | This file defines Vectorized<> for the
  | quantized types.
  |
  | Currently, we simply use these classes as
  | efficient converters between the quantized
  | types and Vectorized<float>, usually in
  | bandwidth-bound cases where doing the
  | arithmetic in full-precision is acceptable
  | (e.g. elementwise operators).
  |
  | Conversions are as follows:
  |
  |  Vectorized<qint8> -> 4x Vectorized<float>
  |  Vectorized<quint8> -> 4x Vectorized<float>
  |  Vectorized<qint32> -> 1x Vectorized<float>
  |
  | The size of the returned float vector is
  | specified by the special constexpr function
  | float_num_vecs. The type of the value returned
  | from dequantize (and expected as an argument to
  | quantize) is specified by
  | float_vec_return_type.
  |
  | When writing kernels with these vectors, it is
  | expected that floating- point operations will
  | be carried out in a loop over
  | Vectorized<T>::float_num_vecs iterations.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cpu/vec/vec256/vec256_qint.h]

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[derive(Default)]
pub struct Vectorizedqi {

    /**
      | __attribute__((aligned(64)));
      |
      */
    vals: Align64<__m256i>,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Vectorizedqi {
    
    pub fn new(v: __m256i) -> Self {
    
        todo!();
        /*
        : vals(v),

        
        */
    }
    
    pub fn operator_m_256i(&self) -> __m256i {
        
        todo!();
        /*
            return vals;
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
pub fn pack_saturate_and_clamp<T>(
        first:   __m256i,
        second:  __m256i,
        min_val: T,
        max_val: T) -> __m256i {
    
    todo!();
        /*
        
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
pub fn pack_saturate_and_clamp_i32(
        first:   __m256i,
        second:  __m256i,
        min_val: i32,
        max_val: i32) -> __m256i {
    
    todo!();
        /*
            // This function is for linkage only, will not be used
      AT_ERROR("pack_saturate_and_clamp<i32> is not supported");
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
pub fn pack_saturate_and_clamp_i8(
    first:   __m256i,
    second:  __m256i,
    min_val: i8,
    max_val: i8) -> __m256i {
    
    todo!();
        /*
            __m256i packed_and_sat = _mm256_packs_epi16(first, second);
      return _mm256_max_epi8(
          _mm256_set1_epi8(min_val),
          _mm256_min_epi8(packed_and_sat, _mm256_set1_epi8(max_val)));
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
pub fn pack_saturate_and_clamp_u8(
        first:   __m256i,
        second:  __m256i,
        min_val: u8,
        max_val: u8) -> __m256i {
    
    todo!();
        /*
            __m256i packed_and_sat = _mm256_packus_epi16(first, second);
      return _mm256_max_epu8(
          _mm256_set1_epi8(min_val),
          _mm256_min_epu8(packed_and_sat, _mm256_set1_epi8(max_val)));
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[inline(always)] pub fn quantize_avx2<T>(
    src:           *const f32,
    dst:           *mut T::underlying,
    len:           i32,
    inverse_scale: f32,
    zero_point:    i64)  {

    todo!();
        /*
            #if defined(target_feature = "avx2")
      constexpr int VLEN = 8;
      constexpr auto min_val = numeric_limits<typename T::underlying>::min();
      constexpr auto max_val = numeric_limits<typename T::underlying>::max();
      const __m256i min_v = _mm256_set1_epi32(min_val);
      const __m256i max_v = _mm256_set1_epi32(max_val);
      // This is the largest int32 value < int32_max exactly representable in float
      constexpr i32 int32_float_max_val =
          i32::max - 127;
      int i = 0;
      __m256 inverse_scale_v = _mm256_set1_ps(inverse_scale);
      // clang-format off
      static const __m256i shuffle_mask_v = _mm256_set_epi8(
          0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff,
          0x0c, 0x08, 0x04, 0x00,
          0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff,
          0x0c, 0x08, 0x04, 0x00);
      // clang-format on
      __m256i permute_mask_v =
          _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
      __m256i permute_mask_l8_v =
          _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
      int len_aligned = len / (VLEN * 4) * (VLEN * 4);
      for (; i < len_aligned; i += 4 * VLEN) {
        // x
        __m256 x_vals = _mm256_load_ps(src + i);
        __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
        // If the floating point value is greater than int32_max,
        // _mm256_cvtps_epi32 converts them to -ve. Clip at int32_float_max_val to
        // Clip at int32_float_max_val to avoid this.
        x_transformed_v =
            _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
        // y
        __m256 y_vals = _mm256_load_ps(src + i + VLEN);
        __m256 y_transformed_v = _mm256_mul_ps(y_vals, inverse_scale_v);
        y_transformed_v =
            _mm256_min_ps(y_transformed_v, _mm256_set1_ps(int32_float_max_val));
        // z
        __m256 z_vals = _mm256_load_ps(src + i + 2 * VLEN);
        __m256 z_transformed_v = _mm256_mul_ps(z_vals, inverse_scale_v);
        z_transformed_v =
            _mm256_min_ps(z_transformed_v, _mm256_set1_ps(int32_float_max_val));
        // w
        __m256 w_vals = _mm256_load_ps(src + i + 3 * VLEN);
        __m256 w_transformed_v = _mm256_mul_ps(w_vals, inverse_scale_v);
        w_transformed_v =
            _mm256_min_ps(w_transformed_v, _mm256_set1_ps(int32_float_max_val));

        __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
        __m256i y_rounded_v = _mm256_cvtps_epi32(y_transformed_v);
        __m256i z_rounded_v = _mm256_cvtps_epi32(z_transformed_v);
        __m256i w_rounded_v = _mm256_cvtps_epi32(w_transformed_v);

        // add zero point
        x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
        y_rounded_v = _mm256_add_epi32(y_rounded_v, _mm256_set1_epi32(zero_point));
        z_rounded_v = _mm256_add_epi32(z_rounded_v, _mm256_set1_epi32(zero_point));
        w_rounded_v = _mm256_add_epi32(w_rounded_v, _mm256_set1_epi32(zero_point));

        __m256i xy_packed_v = _mm256_packs_epi32(x_rounded_v, y_rounded_v);
        __m256i zw_packed_v = _mm256_packs_epi32(z_rounded_v, w_rounded_v);
        __m256i xyzw_clamped_v = pack_saturate_and_clamp<typename T::underlying>(
            xy_packed_v, zw_packed_v, min_val, max_val);

        xyzw_clamped_v =
            _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), xyzw_clamped_v);
      }

      // Additional 8-lane AVX2 version to take advantage when len is smaller
      // based on fbgemm::QuantizeAvx2 (https://github.com/pytorch/FBGEMM)
      for (; i < len / VLEN * VLEN; i += VLEN) {
        __m256 x_vals = _mm256_load_ps(src + i);
        __m256 x_transformed_v = _mm256_mul_ps(x_vals, inverse_scale_v);
        x_transformed_v =
            _mm256_min_ps(x_transformed_v, _mm256_set1_ps(int32_float_max_val));
        __m256i x_rounded_v = _mm256_cvtps_epi32(x_transformed_v);
        x_rounded_v = _mm256_add_epi32(x_rounded_v, _mm256_set1_epi32(zero_point));
        __m256i x_clipped_v =
            _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, x_rounded_v));

        x_clipped_v = _mm256_shuffle_epi8(x_clipped_v, shuffle_mask_v);
        x_clipped_v = _mm256_permutevar8x32_epi32(x_clipped_v, permute_mask_l8_v);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(dst + i),
            _mm256_castsi256_si128(x_clipped_v));
      }

      for (; i < len; ++i) {
        float transformed = src[i] * inverse_scale;

        // Not exactly the same behavior as the vectorized code.
        // The vectorized code above always rounds to even in halfway cases
        // (https://software.intel.com/en-us/node/523819), but nearbyint
        // does the same only when the current rounding mode is FE_TONEAREST.
        // However, in practice, this should not be a problem because most cases
        // use the default rounding mode FE_TONEAREST.
        // Note that we cannot implement the same behavior as the vectorized code
        // using round because it does rounding away from zero in halfway
        // cases.
        transformed = zero_point + nearbyint(transformed);
        float clipped =
            min(max(transformed, float(min_val)), float(max_val));
        dst[i] = clipped;
      }
    #else
      quantize_vec<T>(
          1.0f / inverse_scale, zero_point, src, reinterpret_cast<T*>(dst), len);
    #endif
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub struct VectorizedQint32 {
    base: Vectorizedqi,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_qint32 {

    use super::*;

    pub type SizeType           = i32;
    pub type FloatVecReturnType = Array<Vectorized<f32>,1>;
    pub type IntVecReturnType   = Array<Vectorized<qint32>,1>;
    pub type ValueType          = qint32_underlying;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedQint32 {
    
    pub fn size() -> SizeType {
        
        todo!();
        /*
            return 8;
        */
    }
    
    pub fn float_num_vecs() -> i32 {
        
        todo!();
        /*
            return 1;
        */
    }
    
    pub fn int_num_vecs() -> i32 {
        
        todo!();
        /*
            return 1;
        */
    }
    
    pub fn new(vals: __m256i) -> Self {
    
        todo!();
        /*
            vals = vals_;
        */
    }

    /// Broadcast constructor
    pub fn new(val: &qint32) -> Self {
    
        todo!();
        /*


            value_type uw = val.val_;
            vals = _mm256_set1_epi32(uw);
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut c_void,
        count: i32)  {

        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            if (count != size()) {
            memcpy(ptr, &vals, count * sizeof(value_type));
          } else {
            _mm256_storeu_si256((__m256i*)ptr, vals);
          }
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<qint32> {
        
        todo!();
        /*
            return Vectorized<qint32>(ptr);
        */
    }
    
    pub fn dequantize(&self, 
        scale:           Vectorized<f32>,
        zero_point:      Vectorized<f32>,
        scale_zp_premul: Vectorized<f32>) -> FloatVecReturnType {
        
        todo!();
        /*
            __m256 float_vals = _mm256_cvtepi32_ps(vals);
    #if defined(target_feature = "avx2")
          return {vec::fmadd(scale, Vectorized<float>(float_vals), scale_zp_premul)};
    #else
          return {scale * (Vectorized<float>(float_vals) - zero_point)};
    #endif
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<qint32> {
        
        todo!();
        /*
            Vectorized<qint32> retval;
          auto rhs_data = (__m256)rhs[0];
          quantize_vec<qint32, /*precision=*/32>(
              scale, zero_point, (float*)&rhs_data, (qint32*)&retval.vals, 8);
          return retval;
        */
    }
    
    pub fn maximum(&self, b: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_max_epi32(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i32, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(int_vals.data()), vals);
          array<i32, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(b_vals.data()), b.vals);
          array<i32, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = max<i32>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn minimum(&self, b: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epi32(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i32, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i32, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<i32, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<i32>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<qint32>,
        q_six:      Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epi32(
              _mm256_max_epi32(vals, zero_point.vals), q_six.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i32, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i32, size()> zero_point_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
          array<i32,size()> q_six_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
          array<i32, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<i32>(
                max<i32>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<qint32>) -> IntVecReturnType {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return {_mm256_sub_epi32(vals, b)};
    #else
          array<i32, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i32, size()> b_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<i32, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = int_vals[i] - b_vals[i];
          }
          return {_mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals))};
    #endif
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<qint32> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          __m256 multiplier_v = _mm256_set1_ps(multiplier);
          __m256i zero_point_v = _mm256_set1_epi32(zero_point);

          __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier_v);
          __m256i rounded = _mm256_cvtps_epi32(scaled);
          return _mm256_add_epi32(rounded, zero_point_v);
    #else
          array<i32,size()> inp_vals;
          inp[0].store(inp_vals.data());
          array<i32, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] =
                nearbyint(static_cast<float>(inp_vals[i]) * multiplier) +
                zero_point;
          }
          return loadu(result_vals.data());
    #endif
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            for (usize i = 0; i < 8; ++i) {
              cout << ((i32*)&vals)[i] << " ";
            }
            cout << endl;
        */
    }
 
    /// Load from memory constructor
    ///
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            vals = _mm256_loadu_si256((const __m256i*)ptr);
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Mul<&Vectorized<qint32>> for &Vectorized<qint32> {

    type Output = Vectorized<qint32>;
    
    fn mul(self, other: &Vectorized<qint32>) -> Self::Output {
        todo!();
        /*
            #ifdef target_feature = "avx2"
      return _mm256_mullo_epi32(a, b);
    #else
      // Pray the compiler can autovectorize this
      array<i32, decay_t<decltype(a)>::size()> a_vals;
      array<i32, decay_t<decltype(b)>::size()> b_vals;
      a.store(a_vals.data());
      b.store(b_vals.data());
      array<i32, decay_t<decltype(a)>::size()> result_vals;
      for (usize i = 0; i < decay_t<decltype(a)>::size(); ++i) {
        result_vals[i] = a_vals[i] * b_vals[i];
      }
      return Vectorized<qint32>::loadu(result_vals.data());
    #endif
        */
    }
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl Add<&Vectorized<qint32>> for &Vectorized<qint32> {

    type Output = Vectorized<qint32>;
    
    fn add(self, other: &&Vectorized<qint32>) -> Self::Output {
        todo!();
        /*
            #ifdef target_feature = "avx2"
      return _mm256_add_epi32(a, b);
    #else
      // Pray the compiler can autovectorize this
      array<i32, decay_t<decltype(a)>::size()> a_vals;
      array<i32, decay_t<decltype(b)>::size()> b_vals;
      a.store(a_vals.data());
      b.store(b_vals.data());
      array<i32, decay_t<decltype(a)>::size()> result_vals;
      for (usize i = 0; i < decay_t<decltype(a)>::size(); ++i) {
        result_vals[i] = a_vals[i] + b_vals[i];
      }
      return Vectorized<qint32>::loadu(result_vals.data());
    #endif
        */
    }
}

/**
  | Convert values from int32 back to int8/uint8
  |
  */
#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[cfg(target_feature = "avx2")]
pub fn requantize_avx2<T>(
        inp:        &Array<Vectorized<qint32>,4>,
        multiplier: __m256,
        zp:         __m256i) -> __m256i {

    todo!();
        /*
            static_assert(
          is_same<T, i8>::value || is_same<T, u8>::value,
          "Only i8/u8 are supported");
      constexpr auto min_val = T::min;
      constexpr auto max_val = T::max;
      __m256i permute_mask_v =
          _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
      __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[0]), multiplier);
      __m256 y_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[1]), multiplier);
      __m256 z_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[2]), multiplier);
      __m256 w_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(inp[3]), multiplier);

      __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
      __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
      __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
      __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

      /* Add zero point */
      __m256i x_v = _mm256_add_epi32(x_rounded_v, zp);
      __m256i y_v = _mm256_add_epi32(y_rounded_v, zp);
      __m256i z_v = _mm256_add_epi32(z_rounded_v, zp);
      __m256i w_v = _mm256_add_epi32(w_rounded_v, zp);

      /* Pack to i16 and saturate */
      __m256i xy_packed_v = _mm256_packs_epi32(x_v, y_v);
      __m256i zw_packed_v = _mm256_packs_epi32(z_v, w_v);

      __m256i xyzw_clamped_v =
          pack_saturate_and_clamp<T>(xy_packed_v, zw_packed_v, min_val, max_val);

      /*
       * xyzw_clamped_v has results in the following layout so we need to
       * permute: x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
       */
      xyzw_clamped_v = _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);
      return xyzw_clamped_v;
        */
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
#[derive(Default)]
pub struct VectorizedQint8 {
    base: Vectorizedqi,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_qint8 {

    use super::*;

    pub type FloatVecReturnType = Array<Vectorized<f32>,4>;
    pub type IntVecReturnType   = Array<Vectorized<qint32>,4>;
    pub type ValueType          = qint8::underlying;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedQint8 {
    
    pub fn size() -> i32 {
        
        todo!();
        /*
            return 32;
        */
    }
    
    pub fn float_num_vecs() -> i32 {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn int_num_vecs() -> i32 {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn new(vals: __m256i) -> Self {
    
        todo!();
        /*


            vals = vals_;
        */
    }

    /**
      | Broadcast constructor
      |
      */
    pub fn new(val: &qint8) -> Self {
    
        todo!();
        /*


            value_type uw = val.val_;
            vals = _mm256_set1_epi8(uw);
        */
    }

    /**
      | This is needed because the compiler emits
      | awful code for the default constructor for
      | moving the enum
      |
      */
    pub fn new(other: &Vectorized<qint8>) -> Self {
    
        todo!();
        /*
        : vectorizedqi(other.vals),

        
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut c_void,
        count: i32)  {
        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            if (count != size()) {
                memcpy(ptr, &vals, count * sizeof(value_type));
            } else {
                _mm256_storeu_si256((__m256i*)ptr, vals);
            }
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<qint8> {
        
        todo!();
        /*
            return Vectorized<qint8>(ptr);
        */
    }
    
    pub fn cvtepi8_epi32(&self, epi8_vals: __m128i) -> __m256i {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
            return _mm256_cvtepi8_epi32(epi8_vals);
    #else  // target_feature = "avx2"
            __m128i result_data[2];
            __m128i unpacked1 = _mm_unpacklo_epi8(epi8_vals, epi8_vals);
            __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, unpacked1);
            __m128i shifted1 = _mm_srli_si128(epi8_vals, 4);
            __m128i shifted2 = _mm_srai_epi32(unpacked2, 24);
            result_data[0] = shifted2;
            __m128i unpacked3 = _mm_unpacklo_epi8(shifted1, shifted1);
            __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, unpacked3);
            __m128i shifted3 = _mm_srai_epi32(unpacked4, 24);
            result_data[1] = shifted3;
            return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
    #endif
        */
    }
    
    pub fn dequantize(&self, 
        scale:               Vectorized<f32>,
        zero_point:          Vectorized<f32>,
        scale_neg_zp_premul: Vectorized<f32>) -> FloatVecReturnType {
        
        todo!();
        /*
            __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
        __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
        __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
        __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

        __m256 float_val0 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val0));
        __m256 float_val1 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val1));
        __m256 float_val2 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val2));
        __m256 float_val3 = _mm256_cvtepi32_ps(cvtepi8_epi32(int_val3));

    #if defined(target_feature = "avx2")
        auto val0 =
            vec::fmadd(scale, Vectorized<float>(float_val0), scale_neg_zp_premul);
        auto val1 =
            vec::fmadd(scale, Vectorized<float>(float_val1), scale_neg_zp_premul);
        auto val2 =
            vec::fmadd(scale, Vectorized<float>(float_val2), scale_neg_zp_premul);
        auto val3 =
            vec::fmadd(scale, Vectorized<float>(float_val3), scale_neg_zp_premul);
    #else
        auto val0 = scale * (Vectorized<float>(float_val0) - zero_point);
        auto val1 = scale * (Vectorized<float>(float_val1) - zero_point);
        auto val2 = scale * (Vectorized<float>(float_val2) - zero_point);
        auto val3 = scale * (Vectorized<float>(float_val3) - zero_point);
    #endif
        return {val0, val1, val2, val3};
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<qint8> {
        
        todo!();
        /*
            auto* rhs_data = (float*)rhs.data();
        i8 quantized_values[32];
        QuantizeAvx2<qint8>(
            rhs_data, quantized_values, 32, inverse_scale, zero_point);
        return Vectorized<qint8>::loadu(quantized_values);
        */
    }
    
    pub fn maximum(&self, b: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_max_epi8(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i8, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<i8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = max<i8>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn minimum(&self, b: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epi8(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i8, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<i8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<i8>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<qint8>,
        q_six:      Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epi8(
              _mm256_max_epi8(vals, zero_point.vals), q_six.vals);
    #else
          // Pray the compiler can autovectorize this
          array<i8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<i8, size()> zero_point_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
          array<i8, size()> q_six_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
          array<i8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<i8>(
                max<i8>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<qint8>) -> IntVecReturnType {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
          __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
          __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
          __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

          __m256i int32_val0 = cvtepi8_epi32(int_val0);
          __m256i int32_val1 = cvtepi8_epi32(int_val1);
          __m256i int32_val2 = cvtepi8_epi32(int_val2);
          __m256i int32_val3 = cvtepi8_epi32(int_val3);

          __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
          __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
          __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
          __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

          __m256i int32_b0 = cvtepi8_epi32(int_b0);
          __m256i int32_b1 = cvtepi8_epi32(int_b1);
          __m256i int32_b2 = cvtepi8_epi32(int_b2);
          __m256i int32_b3 = cvtepi8_epi32(int_b3);

          __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
          __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
          __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
          __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);

          return {Vectorized<qint32>(res_0),
                  Vectorized<qint32>(res_1),
                  Vectorized<qint32>(res_2),
                  Vectorized<qint32>(res_3)};
    #else
          // Pray the compiler can autovectorize this
          array<i8, size()> int_vals;
          store(int_vals.data());
          array<i8, size()> b_vals;
          b.store(b_vals.data());
          constexpr int elem_per_int_vec = size() / int_num_vecs();
          i32 rv[int_num_vecs()][elem_per_int_vec];
          for (usize i = 0; i < int_num_vecs(); ++i) {
            for (usize j = 0; j < elem_per_int_vec; ++j) {
              rv[i][j] = static_cast<i32>(int_vals[i * elem_per_int_vec + j]) -
                  static_cast<i32>(b_vals[i * elem_per_int_vec + j]);
            }
          }
          return {Vectorized<qint32>::loadu(rv[0]),
                  Vectorized<qint32>::loadu(rv[1]),
                  Vectorized<qint32>::loadu(rv[2]),
                  Vectorized<qint32>::loadu(rv[3])};
    #endif
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<qint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          __m256 multiplier_v = _mm256_set1_ps(multiplier);
          __m256i zero_point_v = _mm256_set1_epi32(zero_point);
          return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
    #else
          // Pray the compiler can autovectorize this
          constexpr int elem_per_int_vec = size() / int_num_vecs();
          constexpr auto min_val = value_type::min;
          constexpr auto max_val = value_type::max;
          i32 rv[int_num_vecs()][elem_per_int_vec];
          for (usize i = 0; i < int_num_vecs(); ++i) {
            inp[i].store(rv[i]);
          }
          array<i8, size()> result_vals;
          for (usize i = 0; i < int_num_vecs(); ++i) {
            for (usize j = 0; j < elem_per_int_vec; ++j) {
              i32 rounded =
                  nearbyint(static_cast<float>(rv[i][j]) * multiplier) + zero_point;
              result_vals[i * elem_per_int_vec + j] =
                  min<i32>(max<i32>(rounded, min_val), max_val);
            }
          }
          return loadu(result_vals.data());
    #endif
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            for (usize i = 0; i < size(); ++i) {
                cout << (int)((value_type*)&vals)[i] << " ";
            }
            cout << endl;
        */
    }
 
    /// Load from memory constructor
    ///
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            vals = _mm256_loadu_si256((const __m256i*)ptr);
        */
    }
}


#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub struct VectorizedQuint8 {
    base: Vectorizedqi,
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
pub mod vectorized_qint8 {

    use super::*;

    pub type FloatVecReturnType = Array<Vectorized<f32>,4>;
    pub type IntVecReturnType   = Array<Vectorized<qint32>,4>;
    pub type ValueType          = quint8::underlying;
}

#[cfg(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows")))]
impl VectorizedQuint8 {
    
    pub fn size() -> i32 {
        
        todo!();
        /*
            return 32;
        */
    }
    
    pub fn float_num_vecs() -> i32 {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn int_num_vecs() -> i32 {
        
        todo!();
        /*
            return 4;
        */
    }
    
    pub fn new(vals: __m256i) -> Self {
    
        todo!();
        /*


            vals = vals_;
        */
    }
    
    /**
      | Broadcast constructor
      |
      */
    pub fn new(val: &quint8) -> Self {
    
        todo!();
        /*


            value_type uw = val.val_;
            vals = _mm256_set1_epi8(uw);
        */
    }
    
    pub fn new(other: &Vectorized<quint8>) -> Self {
    
        todo!();
        /*
        : vectorizedqi(other.vals),

        
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut c_void,
        count: i32)  {
        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            if (count != size()) {
                memcpy(ptr, &vals, count * sizeof(value_type));
            } else {
                _mm256_storeu_si256((__m256i*)ptr, vals);
            }
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<quint8> {
        
        todo!();
        /*
            return Vectorized<quint8>(ptr);
        */
    }
    
    pub fn cvtepu8_epi32(&self, epu8_vals: __m128i) -> __m256i {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
            return _mm256_cvtepu8_epi32(epu8_vals);
    #else  // target_feature = "avx2"
            __m128i result_data[2];
            __m128i zeros = _mm_setzero_si128();
            __m128i unpacked1 = _mm_unpacklo_epi8(epu8_vals, zeros);
            __m128i unpacked2 = _mm_unpacklo_epi16(unpacked1, zeros);
            result_data[0] = unpacked2;
            __m128i shifted = _mm_srli_si128(epu8_vals, 4);
            __m128i unpacked3 = _mm_unpacklo_epi8(shifted, zeros);
            __m128i unpacked4 = _mm_unpacklo_epi16(unpacked3, zeros);
            result_data[1] = unpacked4;
            return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_data));
    #endif
        */
    }
    
    pub fn dequantize(&self, 
        scale:           Vectorized<f32>,
        zero_point:      Vectorized<f32>,
        scale_zp_premul: Vectorized<f32>) -> FloatVecReturnType {
        
        todo!();
        /*
            __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
        __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
        __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
        __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

        __m256 float_val0 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val0));
        __m256 float_val1 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val1));
        __m256 float_val2 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val2));
        __m256 float_val3 = _mm256_cvtepi32_ps(cvtepu8_epi32(int_val3));

    #if defined(target_feature = "avx2")
        auto val0 =
            vec::fmadd(scale, Vectorized<float>(float_val0), scale_zp_premul);
        auto val1 =
            vec::fmadd(scale, Vectorized<float>(float_val1), scale_zp_premul);
        auto val2 =
            vec::fmadd(scale, Vectorized<float>(float_val2), scale_zp_premul);
        auto val3 =
            vec::fmadd(scale, Vectorized<float>(float_val3), scale_zp_premul);
    #else
        auto val0 = scale * (Vectorized<float>(float_val0) - zero_point);
        auto val1 = scale * (Vectorized<float>(float_val1) - zero_point);
        auto val2 = scale * (Vectorized<float>(float_val2) - zero_point);
        auto val3 = scale * (Vectorized<float>(float_val3) - zero_point);
    #endif
        return {val0, val1, val2, val3};
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<quint8> {
        
        todo!();
        /*
            auto* rhs_data = (float*)rhs.data();
        u8 quantized_values[32];
        QuantizeAvx2<quint8>(
            rhs_data, quantized_values, 32, inverse_scale, zero_point);
        return Vectorized<quint8>::loadu(quantized_values);
        */
    }
    
    pub fn maximum(&self, b: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_max_epu8(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<u8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<u8, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<u8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = max<u8>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn minimum(&self, b: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epu8(vals, b.vals);
    #else
          // Pray the compiler can autovectorize this
          array<u8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<u8, size()> b_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&b_vals), b.vals);
          array<u8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<u8>(int_vals[i], b_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<quint8>,
        q_six:      Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          return _mm256_min_epu8(
              _mm256_max_epu8(vals, zero_point.vals), q_six.vals);
    #else
          // Pray the compiler can autovectorize this
          array<u8, size()> int_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&int_vals), vals);
          array<u8, size()> zero_point_vals;
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(&zero_point_vals), zero_point.vals);
          array<u8, size()> q_six_vals;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&q_six_vals), q_six.vals);
          array<u8, size()> result_vals;
          for (usize i = 0; i < size(); ++i) {
            result_vals[i] = min<u8>(
                max<u8>(int_vals[i], zero_point_vals[i]), q_six_vals[i]);
          }
          return _mm256_loadu_si256(reinterpret_cast<__m256i*>(&result_vals));
    #endif
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<quint8>) -> IntVecReturnType {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          __m128i int_val0 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 0));
          __m128i int_val1 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 1));
          __m128i int_val2 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 2));
          __m128i int_val3 = _mm_set1_epi64x(_mm256_extract_epi64(vals, 3));

          __m256i int32_val0 = cvtepu8_epi32(int_val0);
          __m256i int32_val1 = cvtepu8_epi32(int_val1);
          __m256i int32_val2 = cvtepu8_epi32(int_val2);
          __m256i int32_val3 = cvtepu8_epi32(int_val3);

          __m128i int_b0 = _mm_set1_epi64x(_mm256_extract_epi64(b, 0));
          __m128i int_b1 = _mm_set1_epi64x(_mm256_extract_epi64(b, 1));
          __m128i int_b2 = _mm_set1_epi64x(_mm256_extract_epi64(b, 2));
          __m128i int_b3 = _mm_set1_epi64x(_mm256_extract_epi64(b, 3));

          __m256i int32_b0 = cvtepu8_epi32(int_b0);
          __m256i int32_b1 = cvtepu8_epi32(int_b1);
          __m256i int32_b2 = cvtepu8_epi32(int_b2);
          __m256i int32_b3 = cvtepu8_epi32(int_b3);

          __m256i res_0 = _mm256_sub_epi32(int32_val0, int32_b0);
          __m256i res_1 = _mm256_sub_epi32(int32_val1, int32_b1);
          __m256i res_2 = _mm256_sub_epi32(int32_val2, int32_b2);
          __m256i res_3 = _mm256_sub_epi32(int32_val3, int32_b3);
          return {Vectorized<qint32>(res_0),
                  Vectorized<qint32>(res_1),
                  Vectorized<qint32>(res_2),
                  Vectorized<qint32>(res_3)};
    #else
          // Pray the compiler can autovectorize this
          array<u8, size()> int_vals;
          array<u8, size()> b_vals;
          store(int_vals.data());
          b.store(b_vals.data());
          static constexpr int elem_per_int_vec = size() / int_num_vecs();
          i32 rv[int_num_vecs()][elem_per_int_vec];
          for (usize i = 0; i < int_num_vecs(); ++i) {
            for (usize j = 0; j < elem_per_int_vec; ++j) {
              rv[i][j] = static_cast<i32>(int_vals[i * elem_per_int_vec + j]) -
                  static_cast<i32>(b_vals[i * elem_per_int_vec + j]);
            }
          }
          return {Vectorized<qint32>::loadu(rv[0]),
                  Vectorized<qint32>::loadu(rv[1]),
                  Vectorized<qint32>::loadu(rv[2]),
                  Vectorized<qint32>::loadu(rv[3])};
    #endif
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<quint8> {
        
        todo!();
        /*
            #ifdef target_feature = "avx2"
          __m256 multiplier_v = _mm256_set1_ps(multiplier);
          __m256i zero_point_v = _mm256_set1_epi32(zero_point);
          return RequantizeAvx2<value_type>(inp, multiplier_v, zero_point_v);
    #else
          // Pray the compiler can autovectorize this
          constexpr int elem_per_int_vec = size() / int_num_vecs();
          constexpr auto min_val = value_type::min;
          constexpr auto max_val = value_type::max;
          i32 rv[int_num_vecs()][elem_per_int_vec];
          for (usize i = 0; i < int_num_vecs(); ++i) {
            inp[i].store(rv[i]);
          }
          array<u8, size()> result_vals;
          for (usize i = 0; i < int_num_vecs(); ++i) {
            for (usize j = 0; j < elem_per_int_vec; ++j) {
              i32 rounded =
                  nearbyint(static_cast<float>(rv[i][j]) * multiplier) + zero_point;
              result_vals[i * elem_per_int_vec + j] =
                  min<i32>(max<i32>(rounded, min_val), max_val);
            }
          }
          return loadu(result_vals.data());
    #endif
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            for (usize i = 0; i < size(); ++i) {
                cout << (int)((value_type*)&vals)[i] << " ";
            }
            cout << endl;
        */
    }

    /// Load from memory constructor
    ///
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            vals = _mm256_loadu_si256((const __m256i*)ptr);
        */
    }
}


/**
  | NOTE: These are low-performance implementations
  | that we fall back on if we are not building
  | with AVX2.
  |
  | This may not be an issue, because currently for
  | quantization we assume the user has at least
  | AVX512 installed, so these can simply act as
  | a reference implementation.
  |
  | If in the future we relax this requirement
  | (AVX2+), we should probably revisit these
  | implementations
  */
#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
pub struct VectorizedQuantizedConverter<T,float_vec_return_type_,int_vec_return_type_,const size_: i32> {
    vals: Array<ValueType,usize>,
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
pub mod vectorized_quantized_converter {

    use super::*;

    pub type FloatVecReturnType = FloatVecReturnType;
    pub type IntVecReturnType   = IntVecReturnType;
    pub type ValueType          = T::underlying;
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl<T,float_vec_return_type_,int_vec_return_type_,const size_: i32> 
VectorizedQuantizedConverter<T,float_vec_return_type_,int_vec_return_type_,size_> 
{
    pub fn size() -> i32 {
        
        todo!();
        /*
            return size_;
        */
    }
    
    pub fn float_num_vecs() -> i32 {
        
        todo!();
        /*
            return size() / 8;
        */
    }
    
    pub fn int_num_vecs() -> i32 {
        
        todo!();
        /*
            return size() / 8;
        */
    }
    
    pub fn new(val: T) -> Self {
    
        todo!();
        /*


            for (usize i = 0; i < size(); ++i) {
          vals[i] = val.val_;
        }
        */
    }
    
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            memcpy(vals.data(), ptr, sizeof(value_type) * size());
        */
    }
    
    pub fn store(&self, 
        ptr:   *mut c_void,
        count: i32)  {

        let count: i32 = count.unwrap_or(0);

        todo!();
        /*
            memcpy(ptr, vals.data(), count * sizeof(value_type));
        */
    }
    
    pub fn dequantize(&self, 
        scale:           Vectorized<f32>,
        zero_point:      Vectorized<f32>,
        scale_zp_premul: Vectorized<f32>) -> FloatVecReturnType {
        
        todo!();
        /*
            float_vec_return_type rv;
        for (int i = 0; i < float_num_vecs(); ++i) {
          float tmp_vals[8];
          for (int j = 0; j < 8; ++j) {
            tmp_vals[j] = dequantize_val<T>(
                scale[j], zero_point[j], T(vals[8 * i + j]));
          }
          rv[i] = Vectorized<float>(tmp_vals[0],
              tmp_vals[1],
              tmp_vals[2],
              tmp_vals[3],
              tmp_vals[4],
              tmp_vals[5],
              tmp_vals[6],
              tmp_vals[7]);
        }
        return rv;
        */
    }
    
    pub fn dump(&self)  {
        
        todo!();
        /*
            for (int i = 0; i < size(); ++i) {
              cout << vals[i] << " ";
          }
          cout << endl;
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
pub struct VectorizedQint32 {
    base: VectorizedQuantizedConverter<qint32,Array<Vectorized<f32>,1>,Array<Vectorized<qint32>,1>,8>,
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl Default for VectorizedQint32 {
    
    fn default() -> Self {
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint32,
                array<Vectorized<float>, _1>,
                array<Vectorized<qint32>, _1>,
                _8>()
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl VectorizedQint32 {
    
    pub fn new(val: qint32) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint32,
                array<Vectorized<float>, _1>,
                array<Vectorized<qint32>, _1>,
                _8>(val)
        */
    }
    
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint32,
                array<Vectorized<float>, _1>,
                array<Vectorized<qint32>, _1>,
                _8>(ptr)
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<qint32> {
        
        todo!();
        /*
            return Vectorized<qint32>(ptr);
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<qint32> {
        
        todo!();
        /*
            array<value_type, size()> qvals;
        array<float, float_num_vecs() * 8> float_vals;

        for (int i = 0; i < float_num_vecs(); ++i) {
          rhs[i].store(&float_vals[i * 8], 8);
        }

        quantize_vec<qint32, /*precision=*/32>(
            scale,
            zero_point,
            float_vals.data(),
            (qint32*)qvals.data(),
            8 * float_num_vecs());

        return Vectorized<qint32>::loadu(qvals.data());
        */
    }
    
    pub fn maximum(&self, b: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            Vectorized<qint32> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = max<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn minimum(&self, b: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            Vectorized<qint32> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<qint32>,
        q_six:      Vectorized<qint32>) -> Vectorized<qint32> {
        
        todo!();
        /*
            Vectorized<qint32> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(
              max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<qint32>) -> IntVecReturnType {
        
        todo!();
        /*
            int_vec_return_type retval;
        for (usize i = 0; i < size(); ++i) {
          retval[0].vals[i] = vals[i] - b.vals[i];
        }
        return retval;
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<qint32> {
        
        todo!();
        /*
            Vectorized<qint32> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] =
              nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
              zero_point;
        }
        return retval;
        */
    }
}


#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl Mul<&VectorizedQint32> for &VectorizedQint32 {

    type Output = Vectorized<qint32>;
    
    fn mul(self, other: &VectorizedQint32) -> Self::Output {
        todo!();
        /*
            Vectorized<qint32> retval;
      for (usize i = 0; i < decay_t<decltype(a)>::size(); ++i) {
        retval.vals[i] = a.vals[i] * b.vals[i];
      }
      return retval;
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl Add<&VectorizedQint32> for &VectorizedQint32 {

    type Output = Vectorized<qint32>;
    
    fn add(self, other: &&Vectorized<qint32>) -> Self::Output {
        todo!();
        /*
            Vectorized<qint32> retval;
      for (usize i = 0; i < decay_t<decltype(a)>::size(); ++i) {
        retval.vals[i] = a.vals[i] + b.vals[i];
      }
      return retval;
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
pub struct VectorizedQint8 {
    base: VectorizedQuantizedConverter<qint8,Array<Vectorized<f32>,4>,Array<Vectorized<qint32>,4>,32>,
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl Default for VectorizedQint8 {
    
    fn default() -> Self {
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint8,
                array<Vectorized<float>, _4>,
                array<Vectorized<qint32>, _4>,
                _32>()
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl VectorizedQint8 {
    
    pub fn new(val: qint8) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint8,
                array<Vectorized<float>, 4>,
                array<Vectorized<qint32>, 4>,
                32>(val)
        */
    }
    
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                qint8,
                array<Vectorized<float>, 4>,
                array<Vectorized<qint32>, 4>,
                32>(ptr)
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<qint8> {
        
        todo!();
        /*
            return Vectorized<qint8>(ptr);
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<qint8> {
        
        todo!();
        /*
            array<value_type, size()> qvals;
        array<float, float_num_vecs() * 8> float_vals;

        for (int i = 0; i < float_num_vecs(); ++i) {
          rhs[i].store(&float_vals[i * 8], 8);
        }

        quantize_vec<qint8>(
            scale,
            zero_point,
            float_vals.data(),
            (qint8*)qvals.data(),
            8 * float_num_vecs());

        return Vectorized<qint8>::loadu(qvals.data());
        */
    }
    
    pub fn maximum(&self, b: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            Vectorized<qint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = max<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn minimum(&self, b: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            Vectorized<qint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<qint8>,
        q_six:      Vectorized<qint8>) -> Vectorized<qint8> {
        
        todo!();
        /*
            Vectorized<qint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(
              max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<qint8>) -> IntVecReturnType {
        
        todo!();
        /*
            int_vec_return_type retval;
        constexpr int elem_per_int_vec = size() / int_num_vecs();
        for (usize i = 0; i < int_num_vecs(); ++i) {
          for (usize j = 0; j < elem_per_int_vec; ++j) {
            retval[i].vals[j] =
                static_cast<i32>(vals[i * elem_per_int_vec + j]) -
                static_cast<i32>(b.vals[i * elem_per_int_vec + j]);
          }
        }
        return retval;
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<qint8> {
        
        todo!();
        /*
            constexpr int elem_per_int_vec = size() / int_num_vecs();
        constexpr auto min_val = value_type::min;
        constexpr auto max_val = value_type::max;
        Vectorized<qint8> retval;
        for (usize i = 0; i < int_num_vecs(); ++i) {
          for (usize j = 0; j < elem_per_int_vec; ++j) {
            i32 rounded =
                nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
                zero_point;
            retval.vals[i * elem_per_int_vec + j] =
                min<i32>(max<i32>(rounded, min_val), max_val);
          }
        }
        return retval;
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
pub struct VectorizedQuint8 {
    base: VectorizedQuantizedConverter<quint8,Array<Vectorized<f32>,4>,Array<Vectorized<qint32>,4>,32>,
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl Default for Vectorized {
    
    fn default() -> Self {
        todo!();
        /*


            : VectorizedQuantizedConverter<
                quint8,
                array<Vectorized<float>, 4>,
                array<Vectorized<qint32>, 4>,
                32>()
        */
    }
}

#[cfg(not(all(any(target_feature = "avx",target_feature = "avx2"),not(target_os = "windows"))))]
impl VectorizedQuint8 {
    
    pub fn new(val: quint8) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                quint8,
                array<Vectorized<float>, 4>,
                array<Vectorized<qint32>, 4>,
                32>(val)
        */
    }
    
    pub fn new(ptr: *const c_void) -> Self {
    
        todo!();
        /*


            : VectorizedQuantizedConverter<
                quint8,
                array<Vectorized<float>, 4>,
                array<Vectorized<qint32>, 4>,
                32>(ptr)
        */
    }
    
    pub fn loadu(ptr: *const c_void) -> Vectorized<quint8> {
        
        todo!();
        /*
            return Vectorized<quint8>(ptr);
        */
    }
    
    pub fn quantize(
        rhs:           &FloatVecReturnType,
        scale:         f32,
        zero_point:    i32,
        inverse_scale: f32) -> Vectorized<quint8> {
        
        todo!();
        /*
            array<value_type, size()> qvals;
        array<float, float_num_vecs() * 8> float_vals;

        for (int i = 0; i < float_num_vecs(); ++i) {
          rhs[i].store(&float_vals[i * 8], 8);
        }

        quantize_vec<quint8>(
            scale,
            zero_point,
            float_vals.data(),
            (quint8*)qvals.data(),
            8 * float_num_vecs());

        return Vectorized<quint8>::loadu(qvals.data());
        */
    }
    
    pub fn maximum(&self, b: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            Vectorized<quint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = max<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn minimum(&self, b: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            Vectorized<quint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(vals[i], b.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn relu(&self, zero_point: Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            return maximum(zero_point);
        */
    }
    
    pub fn relu6(&mut self, 
        zero_point: Vectorized<quint8>,
        q_six:      Vectorized<quint8>) -> Vectorized<quint8> {
        
        todo!();
        /*
            Vectorized<quint8> retval;
        for (usize i = 0; i < size(); ++i) {
          retval.vals[i] = min<value_type>(
              max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
        }
        return retval;
        */
    }
    
    pub fn widening_subtract(&self, b: Vectorized<quint8>) -> IntVecReturnType {
        
        todo!();
        /*
            int_vec_return_type retval;
        constexpr int elem_per_int_vec = size() / int_num_vecs();
        for (usize i = 0; i < int_num_vecs(); ++i) {
          for (usize j = 0; j < elem_per_int_vec; ++j) {
            retval[i].vals[j] =
                static_cast<i32>(vals[i * elem_per_int_vec + j]) -
                static_cast<i32>(b.vals[i * elem_per_int_vec + j]);
          }
        }
        return retval;
        */
    }
    
    pub fn requantize_from_int(
        inp:        &IntVecReturnType,
        multiplier: f32,
        zero_point: i32) -> Vectorized<quint8> {
        
        todo!();
        /*
            constexpr int elem_per_int_vec = size() / int_num_vecs();
        constexpr auto min_val = value_type::min;
        constexpr auto max_val = value_type::max;
        Vectorized<quint8> retval;
        for (usize i = 0; i < int_num_vecs(); ++i) {
          for (usize j = 0; j < elem_per_int_vec; ++j) {
            i32 rounded =
                nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
                zero_point;
            retval.vals[i * elem_per_int_vec + j] =
                min<i32>(max<i32>(rounded, min_val), max_val);
          }
        }
        return retval;
        */
    }
}

pub trait Q8 {}

impl Q8 for qint8  {}
impl Q8 for quint8 {}
impl Q8 for qint32 {}

#[inline] pub fn maximum<T: Q8>(
    a: &Vectorized<T>,
    b: &Vectorized<T>) -> Vectorized<T> {
    
    todo!();
        /*
            return a.maximum(b);
        */
}


crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/affine_quantizer_base.h]

pub fn dequantize_vec<T>(
    scale:      f64,
    zero_point: i64,
    src:        *const T,
    dst:        *mut f32,
    count:      usize) -> f32 {

    let count: usize = count.unwrap_or(8);

    todo!();
        /*
        
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/affine_quantizer_base.cpp]

pub fn check_zero_point<T>(
    fn_name:    &String,
    zero_point: i64)  {

    todo!();
        /*
            TORCH_CHECK(
          zero_point <= T::max,
          fn_name,
          " zero_point ",
          zero_point,
          " is out of range.");
      TORCH_CHECK(
          zero_point >= T::min,
          fn_name,
          " zero_point ",
          zero_point,
          " is out of range.");
        */
}

pub struct QuantizeValCommand {
    scale:      f64,
    zero_point: i64,
    value:      f32,
}

impl QuantizeValCommand {

    /**
      | Quantize a float value into a uint value
      | given scale and zero_point
      |
      | @note
      | 
      | quantize_val is only explicitly used
      | in test outside of this file
      |
      */
    #[cfg(feature = "fbgemm")]
    #[inline] pub fn quantize<T>(&self) -> T {

        todo!();
            /*
                // Internally, fbgemm::Quantize uses nearbyint.
          // nearbyint results in nearest integer value according to the current
          // rounding mode and the default rounding mode is rounds to even in half-way
          // cases in most popular processor architectures like x86 and ARM. This is
          // typically faster than an alternatives like round that rounds half-way
          // cases away from zero, and can be consistent with SIMD implementations for
          // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
          // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
          i32 qvalue;
          qvalue = fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
              value,
              static_cast<i32>(zero_point),
              static_cast<float>(scale),
              /*result_precision=*/CHAR_BIT * sizeof(typename T::underlying));
          return static_cast<T>(qvalue);
            */

    }

    #[cfg(not(feature = "fbgemm"))]
    #[inline] pub fn quantize<T>(&self) -> T {

        todo!();
            /*
                // nearbyint results in nearest integer value according to the current
          // rounding mode and the default rounding mode is rounds to even in half-way
          // cases in most popular processor architectures like x86 and ARM. This is
          // typically faster than an alternatives like round that rounds half-way
          // cases away from zero, and can be consistent with SIMD implementations for
          // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
          // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
          i64 qvalue;
          constexpr i64 qmin = numeric_limits<typename T::underlying>::min();
          constexpr i64 qmax = numeric_limits<typename T::underlying>::max();
          float inv_scale = 1.0f / static_cast<float>(scale);
          qvalue = static_cast<i64>(zero_point + Round(value * inv_scale));
          qvalue = max<i64>(qvalue, qmin);
          qvalue = min<i64>(qvalue, qmax);
          return static_cast<T>(qvalue);
            */
    }
}

pub fn quantize_val_fbgemm<T>(
    scale:      f64,
    zero_point: i64,
    value:      f32) -> T {

    QuantizeValCommand { scale, zero_point, value }.exec()
}

#[cfg(feature = "fbgemm")]
pub fn quantize_vec<T, const precision: i32 = 8>(
    scale:      f64,
    zero_point: i64,
    src:        *const f32,
    dst:        *mut T,
    count:      Size)  {

    let count: Size = count.unwrap_or(8);

    todo!();
        /*
            fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
          src,
          (typename T::underlying*)dst,
          count,
          fbgemm::TensorQuantizationParams{
              (float)scale, (i32)zero_point, precision});
        */
}

#[cfg(feature = "fbgemm")]
#[inline] pub fn dequantize_val<T>(
    scale:      f64,
    zero_point: i64,
    value:      T) -> f32 {

    todo!();
        /*
      fbgemm::TensorQuantizationParams qparams;
      qparams.scale = static_cast<float>(scale);
      qparams.zero_point = static_cast<i32>(zero_point);
      return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
        */
}


#[cfg(not(feature = "fbgemm"))]
#[cfg(all(__ANDROID__,not(__NDK_MAJOR__)))]
#[inline] pub fn round<T>(x: f32) -> f32 {

    todo!();
        /*
            return ::nearbyintf(x);
        */
}

#[cfg(not(feature = "fbgemm"))]
#[cfg(all(__ANDROID__,not(__NDK_MAJOR__)))]
#[inline] pub fn round(x: f64) -> f64 {
    
    todo!();
        /*
            return ::nearbyint(x);
        */
}

#[cfg(not(feature = "fbgemm"))]
#[cfg(not(all(__ANDROID__,not(__NDK_MAJOR__))))]
#[inline] pub fn round<T>(x: T) -> T {

    todo!();
        /*
            return nearbyint(x);
        */
}

/**
  | TODO combine this with quantize_val
  | once the numerics for ARM are aligned
  | with it
  |
  */
#[cfg(not(feature = "fbgemm"))]
pub fn quantize_val_arm(
    scale:      f32,
    zero_point: i32,
    value:      f32) -> u8 {
    
    todo!();
        /*
            const i32 qmin = u8::min;
      const i32 qmax = u8::max;
      float inv_scale = 1.0f / scale;
      auto r = zero_point + static_cast<i32>(Round(value * inv_scale));
      r = max(r, qmin);
      r = min(r, qmax);
      return static_cast<u8>(r);
        */
}

#[cfg(not(feature = "fbgemm"))]
pub fn quantize_vec<T, const precision: i32 = 8>(
    scale:      f64,
    zero_point: i64,
    src:        *const f32,
    dst:        *mut T,
    count:      usize)  {

    let count: Size = count.unwrap_or(8);

    todo!();
        /*
            checkZeroPoint<typename T::underlying>("quantize_vec", zero_point);
      for (i64 i = 0; i < count; ++i) {
        dst[i] = quantize_val<T>(scale, zero_point, src[i]);
      }
        */
}

#[cfg(not(feature = "fbgemm"))]
pub fn dequantize_val<T>(
    scale:      f64,
    zero_point: i64,
    value:      T) -> f32 {

    todo!();
        /*
            // We need to convert the qint8 value to float to ensure the subtraction
      // subexpression returns a float
      return (static_cast<float>(value.val_) - zero_point) * scale;
        */
}

/**
  | Quantize value based on the following
  | equation
  | 
  | Xq = Round(Xf * inv_scale + zero_point)
  | where zero_point is in float.
  | 
  | -----------
  | @note
  | 
  | For the case of embedding quantization
  | we will set zero_point to (-Xmin/scale),
  | where Xmin is the min value in input tensor
  | row.
  |
  */
pub fn quantize_val_float_qparams(
    scale:      f32,
    zero_point: f32,
    value:      f32,
    qmin:       i32,
    qmax:       i32) -> i32 {

    todo!();
        /*
      int qvalue;

      float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
      qvalue = lrintf(value * inv_scale + zero_point);
      qvalue = max(qmin, min(qvalue, qmax));
      return qvalue;
        */
}

pub fn requantize_val<SRC_T, DST_T>(
    src_scale:      f64,
    src_zero_point: i64,
    dst_scale:      f64,
    dst_zero_point: i64,
    src:            SRC_T) -> DST_T {

    todo!();
        /*
            const auto dq = dequantize_val<SRC_T>(src_scale, src_zero_point, src);
      return quantize_val<DST_T>(dst_scale, dst_zero_point, dq);
        */
}

/**
  | Given a multiplier and a zero_point, requantize
  | i32 computed values back to quantized
  | values.
  |
  | See comment above
  | make_per_tensor_affine_quantizer function for
  | the usage of i64
  */
pub fn requantize_from_int<DST_T>(
    multiplier: f64,
    zero_point: i64,
    src:        i64) -> DST_T {

    todo!();
        /*
            i64 quantize_down =
          zero_point + lrintf(src * static_cast<float>(multiplier));
      i32 min = numeric_limits<typename DST_T::underlying>::min();
      i32 max = numeric_limits<typename DST_T::underlying>::max();
      return static_cast<DST_T>(
          min<i64>(max<i64>(quantize_down, min), max));
        */
}

/**
  | Code here is partially derived from
  | gemmlowp library (https://github.com/google/gemmlowp)
  |
  */

crate::ix!();

/**
  | Initialized QNNPACK (only once).
  | 
  | Throws if initialization failed.
  |
  */
#[inline] pub fn initQNNPACK()  {
    
    todo!();
    /*
        static std::once_flag once;
      static enum qnnp_status qnnpackStatus = qnnp_status_uninitialized;
      std::call_once(once, []() { qnnpackStatus = qnnp_initialize(); });
      CAFFE_ENFORCE(
          qnnpackStatus == qnnp_status_success, "failed to initialize QNNPACK");
    */
}

#[inline] pub fn multiply_by_quantized_multiplier_smaller_than_one(
    x:                    i32,
    quantized_multiplier: i32,
    right_shift:          i32) -> i32 
{
    todo!();
    /*
        using gemmlowp::RoundingDivideByPOT;
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      return RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(x, quantized_multiplier), right_shift);
    */
}

#[cfg(target = "android")]
#[inline] pub fn round<T>(x: f32) -> f32 {
    todo!();
    /*
        return ::nearbyintf(x);
    */
}

#[cfg(target = "android")]
#[inline] pub fn round(x: f64) -> f64 {
    
    todo!();
    /*
        return ::nearbyint(x);
    */
}

#[cfg(not(target = "android"))]
#[inline] pub fn round<T>(x: T) -> T {
    todo!();
    /*
        return std::nearbyint(x);
    */
}

#[inline] pub fn quantize_uint8(
    scale: f32, 
    zero_point: i32, 
    value: f32) -> u8 
{
    todo!();
    /*
        const int32_t qmin = uint8_t::min;
      const int32_t qmax = uint8_t::max;

      auto r = zero_point + static_cast<int32_t>(Round(value / scale));
      r = std::max(r, qmin);
      r = std::min(r, qmax);
      return static_cast<uint8_t>(r);
    */
}

#[inline] pub fn quantize_multiplier_smaller_than_one(
    double_multiplier:    f64,
    quantized_multiplier: *mut i32,
    right_shift:          *mut i32)  
{
    
    todo!();
    /*
        CHECK(double_multiplier >= 0.);
      CHECK(double_multiplier < 1.);
      if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *right_shift = 0;
        return;
      }
      CHECK(double_multiplier > 0.);
      const double q = std::frexp(double_multiplier, right_shift);
      *right_shift *= -1;

      auto q_fixed = static_cast<int64_t>(Round(q * (1ll << 31)));
      CHECK(q_fixed <= (1ll << 31));
      if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        --*right_shift;
      }
      CHECK_GE(*right_shift, 0);
      CHECK_LE(q_fixed, int32_t::max);
      *quantized_multiplier = static_cast<int32_t>(q_fixed);
    */
}

#[inline] pub fn quantize_multiplier_greater_than_one(
    double_multiplier:    f64,
    quantized_multiplier: *mut i32,
    left_shift:           *mut i32)  
{
    
    todo!();
    /*
        CHECK(double_multiplier > 1.);
      const double q = std::frexp(double_multiplier, left_shift);
      auto q_fixed = static_cast<int64_t>(Round(q * (1ll << 31)));
      CHECK(q_fixed <= (1ll << 31));
      if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*left_shift;
      }
      CHECK_GE(*left_shift, 0);
      CHECK_LE(q_fixed, int32_t::max);
      *quantized_multiplier = static_cast<int32_t>(q_fixed);
    */
}

#[inline] pub fn multiply_by_quantized_multiplier_greater_than_one(
    x:                    i32,
    quantized_multiplier: i32,
    left_shift:           i32) -> i32 
{
    todo!();
    /*
        using gemmlowp::SaturatingRoundingDoublingHighMul;
      return SaturatingRoundingDoublingHighMul(
          x * (1 << left_shift), quantized_multiplier);
    */
}

#[inline] pub fn calculate_input_radius(
    input_integer_bits: i32,
    input_left_shift:   i32) -> i32 
{
    todo!();
    /*
        const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
          (1ll << (31 - input_integer_bits)) / (1ll << input_left_shift);
      // Tighten bound using floor.  Suppose that we could use the exact value.
      // After scaling the difference, the result would be at the maximum.  Thus we
      // must ensure that our value has lower magnitude.
      return static_cast<int>(std::floor(max_input_rescaled));
    */
}

#[repr(u8)]
#[derive(PartialEq,Eq)]
pub enum Activation { 
    NONE = 0, 
    RELU = 1 
}

#[inline] pub fn activation_limits(
    scale:      f32,
    zero_point: i32,
    ac:         Activation) -> (u8,u8) 
{
    
    todo!();
    /*
        switch (Ac) {
        case Activation::NONE:
          return {uint8_t::min,
                  uint8_t::max};
        case Activation::RELU:
          return {QuantizeUint8(scale, zero_point, 0.0),
                  uint8_t::max};
        default:
    #ifdef _MSC_VER
          __assume(0);
    #else
          __builtin_unreachable();
    #endif
      }
    */
}

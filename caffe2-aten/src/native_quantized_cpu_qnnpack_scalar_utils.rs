// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/scalar-utils.h]

lazy_static!{
    /*
    #if defined(__clang__)
    #if __clang_major__ == 3 && __clang_minor__ >= 7 || __clang_major__ > 3
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
      __attribute__((__no_sanitize__("shift-base")))
    #else
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
    #endif
    #elif defined(__GNUC__)
    #if __GNUC__ >= 8
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB \
      __attribute__((__no_sanitize__("shift-base")))
    #elif __GNUC__ == 4 && __GNUC_MINOR__ >= 9 || __GNUC__ > 4
    /* 4.9 <= gcc < 8 support ubsan, but doesn't support no_sanitize attribute */
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
    #ifndef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
    #define PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND 1
    #endif
    #else
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
    #endif
    #else
    #define PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB
    #endif
    */
}

//#[PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB]
#[inline] pub fn asr_s32(
        x: i32,
        n: u32) -> i32 {
    
    todo!();
        /*
            #ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
    #if defined(__x86_64__) || defined(__aarch64__)
      return (i32)((u64)(i64)x >> n);
    #else
      return x >= 0 ? x >> n : ~(~x >> n);
    #endif
    #else
      return x >> n;
    #endif
        */
}

//#[PYTORCH_QNNP_IGNORE_SHIFT_BASE_UB]
#[inline] pub fn asr_s64(x: i64, n: u32) -> i64 {
    
    todo!();
        /*
            #ifdef PYTORCH_QNNP_USE_SHIFT_BASE_UB_WORKAROUND
      return x >= 0 ? x >> n : ~(~x >> n);
    #else
      return x >> n;
    #endif
        */
}

#[inline] pub fn pytorch_scalar_requantize_precise(
        value:      i32,
        scale:      f32,
        zero_point: u8,
        qmin:       u8,
        qmax:       u8) -> u8 {
    
    todo!();
        /*
            assert(scale < 1.0f);
      assert(scale >= 0x1.0p-32f);

      const u32 scale_bits = fp32_to_bits(scale);
      const u32 multiplier =
          (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
      const u32 shift = 127 + 23 - (scale_bits >> 23);
      assert(shift >= 24);
      assert(shift < 56);

      /*
       * Compute absolute value of input as unsigned 32-bit int.
       * All further computations will work with unsigned values to avoid undefined
       * behaviour on signed operations.
       */
      const u32 abs_value = (value >= 0) ? (u32)value : -(u32)value;

      /* Compute full 64-bit product of 32-bit factors */
      const u64 product = (u64)abs_value * (u64)multiplier;

      /*
       * Shift the full 64-bit product right with rounding.
       * Rounding is performed towards closest integer, with midpoints rounded up
       * (same as away from zero).
       */
      const u64 rounding = UINT64_C(1) << (shift - 1);
      const u32 abs_scaled_value = (u32)((product + rounding) >> shift);

      /*
       * Copy the sign of input to scaled absolute input value.
       */
      const i32 scaled_value =
          (i32)(value >= 0 ? abs_scaled_value : -abs_scaled_value);

      /* Clamp scaled value with zero point between smin and smax */
      i32 clamped_value = scaled_value;
      const i32 smin = (i32)(u32)qmin - (i32)(u32)zero_point;
      if (clamped_value < smin) {
        clamped_value = smin;
      }
      const i32 smax = (i32)(u32)qmax - (i32)(u32)zero_point;
      if (clamped_value > smax) {
        clamped_value = smax;
      }

      /* Add zero point to clamped value */
      const i32 biased_value = clamped_value + (i32)(u32)zero_point;

      return biased_value;
        */
}

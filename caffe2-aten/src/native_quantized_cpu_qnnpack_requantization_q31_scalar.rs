// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/q31-scalar.c]

pub fn pytorch_qnnp_requantize_q31_scalar(
        n:          Size,
        input:      *const i32,
        scale:      f32,
        zero_point: u8,
        qmin:       u8,
        qmax:       u8,
        output:     *mut u8)  {
    
    todo!();
        /*
            assert(n % 4 == 0);
      assert(scale < 1.0f);
      assert(scale >= 0x1.0p-32f);

      /* Compute requantization parameters */
      const u32 scale_bits = fp32_to_bits(scale);

      /* Multiplier is in [0x40000000, 0x7FFFFF80] range */
      const i32 multiplier = (i32)(
          ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
      assert(multiplier >= INT32_C(0x40000000));
      assert(multiplier <= INT32_C(0x7FFFFF80));

      /* Shift is in [0, 31] range */
      const i32 shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
      assert(shift >= 0);
      assert(shift < 32);

      const i64 q31rounding = INT64_C(0x40000000);
      const i32 remainder_mask =
          (i32)((UINT32_C(1) << shift) - UINT32_C(1));
      const i32 threshold = (i32)((u32)remainder_mask >> 1);
      const i32 smin = (i32)(u32)qmin - (i32)(u32)zero_point;
      const i32 smax = (i32)(u32)qmax - (i32)(u32)zero_point;
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        /*
         * Compute full 64-bit product of signed 32-bit factors.
         *
         * Note: multiplier can be treated as either signed or unsigned.
         */
        const i64 x_product = (i64)x * (i64)multiplier;
        const i64 y_product = (i64)y * (i64)multiplier;
        const i64 z_product = (i64)z * (i64)multiplier;
        const i64 w_product = (i64)w * (i64)multiplier;

        /*
         * Get the Q31 multiplication result by extracting bits 31-62 of the
         * product, with rounding up. Add rounding value (0x40000000) and then shift
         * right by 31 bits and extract the low 32-bit word. Note: casts to unsigned
         * types are needed to avoid undefined behavior. Given the multiplier range,
         * the result of Q31 multiplication is in [-2147483520, 2147483519] range.
         */
        const i32 x_q31product =
            (i32)(u32)((u64)(x_product + q31rounding) >> 31);
        const i32 y_q31product =
            (i32)(u32)((u64)(y_product + q31rounding) >> 31);
        const i32 z_q31product =
            (i32)(u32)((u64)(z_product + q31rounding) >> 31);
        const i32 w_q31product =
            (i32)(u32)((u64)(w_product + q31rounding) >> 31);

        /*
         * Arithmetically shift the adjusted product right with rounding.
         * Rounding is performed towards closest integer, with midpoints rounded
         * away from zero.
         *
         * Shift with correct rounding could be efficiently implemented by
         * pre-adding rounding constant, but with input in
         * [-2147483520, 2147483519] range and rounding constant up to 2**30 we
         * can't rule out overflow. This limitation leaves us with 3 options:
         * 1. Extend input to 64-bit signed integer, perform addition and shift on
         * 64-bit integers, then truncate result to 32 bits.
         * 2. Detect overflow and handle this situation separately. Note that
         * overflow is possible only when input is positive, and even when addition
         * of a rounding constant overflows 32-bit signed integer, it still doesn't
         *    overflow 32-bit unsigned integer. Thus, in case of signed overflow, we
         * can compute the result using unsigned arithmetics, specifically using
         * logical shift right instead of arithmetic shift right.
         * 3. Performs arithmetic shift as is, which will produce division result
         * rounded down. Then compute remainder of this division by a power of 2,
         * and adjust the result. Result needs adjustment (increment by 1) when
         *     - input is positive, shift is non-zero, and remainder >= 2**(shift -
         * 1), e.g. 10 >> 2 needs adjustment
         *     - input is negative, shift is non-zero, and remainder > 2**(shift -
         * 1), e.g. -10 >> 2 doesn't need adjustment These conditions can be
         * generalized as remainder + (input <= 0) > 2**(shift - 1) or equivalently
         *        remainder - (input < 0) > ((2**shift - 1) >> 1)
         *    When shift is 0, remainder is 0 as well, the last condition is always
         * false, and no adjustment is done.
         *
         * Among these options, option 3 is the most performant across the board,
         * although option 1 is promising for 64-bit instruction sets.
         */
        const i32 x_remainder =
            (x_q31product & remainder_mask) - (i32)(x_q31product < 0);
        const i32 y_remainder =
            (y_q31product & remainder_mask) - (i32)(y_q31product < 0);
        const i32 z_remainder =
            (z_q31product & remainder_mask) - (i32)(z_q31product < 0);
        const i32 w_remainder =
            (w_q31product & remainder_mask) - (i32)(w_q31product < 0);

        const i32 x_scaled =
            asr_s32(x_q31product, shift) + (i32)(x_remainder > threshold);
        const i32 y_scaled =
            asr_s32(y_q31product, shift) + (i32)(y_remainder > threshold);
        const i32 z_scaled =
            asr_s32(z_q31product, shift) + (i32)(z_remainder > threshold);
        const i32 w_scaled =
            asr_s32(w_q31product, shift) + (i32)(w_remainder > threshold);

        /*
         * Clamp scaled value with zero point between (qmin - zero point) and (qmax
         * - zero point).
         */
        const i32 x_clamped =
            x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
        const i32 y_clamped =
            y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
        const i32 z_clamped =
            z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
        const i32 w_clamped =
            w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

        /*
         * Add zero point to clamped value.
         * The result is guaranteed to be in [qmin, qmax] range.
         *
         * This addition can not be safely done before clamping, because scaled
         * values are in [-2147483520, 2147483519] range, so addition of zero point
         * (which can be up to 255) can overflow signed 32-bit integer.
         */
        const i32 x_biased = x_clamped + zero_point;
        const i32 y_biased = y_clamped + zero_point;
        const i32 z_biased = z_clamped + zero_point;
        const i32 w_biased = w_clamped + zero_point;

        output[0] = (u8)x_biased;
        output[1] = (u8)y_biased;
        output[2] = (u8)z_biased;
        output[3] = (u8)w_biased;
        output += 4;
      }
        */
}

// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/precise-scalar.c]

pub fn pytorch_qnnp_requantize_precise_scalar_unsigned32(
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

      const u32 scale_bits = fp32_to_bits(scale);
      const u32 multiplier = (scale_bits << 8) | UINT32_C(0x80000000);
      const u32 shift = 127 + 31 - (scale_bits >> 23);
      assert(shift >= 32);
      assert(shift < 64);

      const u64 rounding = UINT64_C(1) << (shift - 1);
      const u32 rounding_hi = (u32)(rounding >> 32);
      const u32 rounding_lo = (u32)rounding;
      const u32 shift_minus_32 = shift - 32;
      const i32 smin = (i32)(u32)qmin - (i32)(u32)zero_point;
      const i32 smax = (i32)(u32)qmax - (i32)(u32)zero_point;
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        /*
         * Compute absolute value of input as unsigned 32-bit int.
         * All further computations will work with unsigned values to avoid
         * undefined behaviour on signed operations.
         */
        const u32 x_abs = (x >= 0) ? (u32)x : -(u32)x;
        const u32 y_abs = (y >= 0) ? (u32)y : -(u32)y;
        const u32 z_abs = (z >= 0) ? (u32)z : -(u32)z;
        const u32 w_abs = (w >= 0) ? (u32)w : -(u32)w;

        /* Compute full 64-bit product of 32-bit factors */
        const u64 x_product = (u64)x_abs * (u64)multiplier;
        const u64 y_product = (u64)y_abs * (u64)multiplier;
        const u64 z_product = (u64)z_abs * (u64)multiplier;
        const u64 w_product = (u64)w_abs * (u64)multiplier;

        /*
         * Shift the full 64-bit product right with rounding.
         * Rounding is performed towards closest integer, with midpoints rounded up
         * (same as away from zero).
         *
         * Generally, this operation requires both 64-bit addition and 64-bit shift,
         * but we use two tricks to replace 64-bit operations with 32-bit
         * operations.
         *
         * To avoid full 64-bit addition we make use of three facts:
         * - 64-bit rounding value added before the shift is a power of 2, and thus
         * has only one bit set.
         * - When 0x1.0p-32f <= scale < 0x1.0p-31f, then the non-zero bit in
         * rounding is in the low 32 bits, and rounding is exactly 0x80000000
         * (2**31), because rounding is 2**(scale-1) and scale >= 32. In this case,
         *   addition of rounding can affect high 32 bits of the product only
         * through overflow, which happens if low 32-bit part of the product equals
         * or exceeds 0x80000000. We can reformulate the latter condition as low
         * 32-bit part of the product has the bit 31 set, and then overflow happens
         * if both the low 32-bit part of the product and the low 32-bit part of the
         * rounding value have bit 31 set. Since 32-bit numbers with the bit 31 set
         * are negative when interpreted as signed integers, we can check the
         * overflow condition as (i32) (LOW(product) & LOW(rounding)) < 0
         * - When 0x1.0p-31f <= scale < 1.0f, then the non-zero bit is in the high
         * 32 bits of rounding. We just need to do 32-bit addition of high 32 bits
         * of rounding and high 32 bits of product. This addition never overflows
         * because product <= 0x80000000 * 0xFFFFFF00 < 2**63 and rounding =
         * 2**(scale-1) <= 2**62.
         *
         * To avoid full 64-bit shift, we leverage the fact that shift >= 32, and do
         * it in two steps:
         * - Shift by 32, which can be implemented by extracting the high 32-bit word
         * on 32-bit systems.
         * - Shift by (shift - 32), which can be implemented as a 32-bit shift of
         * high word of addition result.
         */
        const u32 x_carry_lo =
            (u32)((i32)((u32)x_product & rounding_lo) < 0);
        const u32 y_carry_lo =
            (u32)((i32)((u32)y_product & rounding_lo) < 0);
        const u32 z_carry_lo =
            (u32)((i32)((u32)z_product & rounding_lo) < 0);
        const u32 w_carry_lo =
            (u32)((i32)((u32)w_product & rounding_lo) < 0);

        const u32 x_product_hi = (u32)(x_product >> 32);
        const u32 y_product_hi = (u32)(y_product >> 32);
        const u32 z_product_hi = (u32)(z_product >> 32);
        const u32 w_product_hi = (u32)(w_product >> 32);

        const u32 x_abs_scaled =
            (u32)(x_product_hi + rounding_hi + x_carry_lo) >> shift_minus_32;
        const u32 y_abs_scaled =
            (u32)(y_product_hi + rounding_hi + y_carry_lo) >> shift_minus_32;
        const u32 z_abs_scaled =
            (u32)(z_product_hi + rounding_hi + z_carry_lo) >> shift_minus_32;
        const u32 w_abs_scaled =
            (u32)(w_product_hi + rounding_hi + w_carry_lo) >> shift_minus_32;

        /* Copy the sign of input to scaled absolute input value */
        const i32 x_scaled = (i32)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
        const i32 y_scaled = (i32)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
        const i32 z_scaled = (i32)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
        const i32 w_scaled = (i32)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

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

pub fn pytorch_qnnp_requantize_precise_scalar_unsigned64(
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

      const u32 scale_bits = fp32_to_bits(scale);
      const u32 multiplier =
          (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
      const u32 shift = 127 + 23 - (scale_bits >> 23);
      assert(shift >= 24);
      assert(shift < 56);

      const u64 rounding = UINT64_C(1) << (shift - 1);
      const i32 smin = (i32)(u32)qmin - (i32)(u32)zero_point;
      const i32 smax = (i32)(u32)qmax - (i32)(u32)zero_point;
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        /*
         * Compute absolute value of input as unsigned 32-bit int.
         * All further computations will work with unsigned values to avoid
         * undefined behaviour on signed operations.
         */
        const u32 x_abs = (x >= 0) ? (u32)x : -(u32)x;
        const u32 y_abs = (y >= 0) ? (u32)y : -(u32)y;
        const u32 z_abs = (z >= 0) ? (u32)z : -(u32)z;
        const u32 w_abs = (w >= 0) ? (u32)w : -(u32)w;

        /* Compute full 64-bit product of 32-bit factors */
        const u64 x_product = (u64)x_abs * (u64)multiplier;
        const u64 y_product = (u64)y_abs * (u64)multiplier;
        const u64 z_product = (u64)z_abs * (u64)multiplier;
        const u64 w_product = (u64)w_abs * (u64)multiplier;

        /*
         * Shift the full 64-bit product right with rounding.
         * Rounding is performed towards closest integer, with midpoints rounded up
         * (same as away from zero).
         *
         * Note that although rounding is precomputed, it is dependent on shift
         * value, and on processors with 64-bit "right shift with rounding"
         * instruction each line below can be represented by just one such
         * instruction (e.g. VRSHL.U64 on ARM NEON, URSHL in ARM64 Advanced SIMD).
         */
        const u32 x_abs_scaled = (u32)((x_product + rounding) >> shift);
        const u32 y_abs_scaled = (u32)((y_product + rounding) >> shift);
        const u32 z_abs_scaled = (u32)((z_product + rounding) >> shift);
        const u32 w_abs_scaled = (u32)((w_product + rounding) >> shift);

        /*
         * Copy the sign of input to scaled absolute input value.
         *
         * On x86 processors with SSSE3 instruction set, this operation nicely maps
         * to PSIGND instruction.
         */
        const i32 x_scaled = (i32)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
        const i32 y_scaled = (i32)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
        const i32 z_scaled = (i32)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
        const i32 w_scaled = (i32)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

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


pub fn pytorch_qnnp_requantize_precise_scalar_signed64(
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

      const u32 scale_bits = fp32_to_bits(scale);
      const i32 multiplier =
          ((i32)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
      const u32 shift = 127 + 23 - (scale_bits >> 23);
      assert(shift >= 24);
      assert(shift < 56);

      const i64 rounding = INT64_C(1) << (shift - 1);
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
         * Adjust product before subsequent shift with rounding up to simulate shift
         * with rounding away from zero.
         */
        const i64 x_adjusted_product = x_product - (i64)(x < 0);
        const i64 y_adjusted_product = y_product - (i64)(y < 0);
        const i64 z_adjusted_product = z_product - (i64)(z < 0);
        const i64 w_adjusted_product = w_product - (i64)(w < 0);

        /*
         * Arithmetically shift the full 64-bit product right with rounding.
         * Rounding is performed towards closest integer, with midpoints rounded up.
         *
         * Note that although rounding is precomputed, it is dependent on shift
         * value, and on processors with 64-bit "right shift with rounding"
         * instruction each line below can be represented by just one such
         * instruction (e.g. VRSHL.S64 on ARM NEON, SRSHL in ARM64 Advanced SIMD).
         */
        const i32 x_scaled =
            (i32)asr_s64(x_adjusted_product + rounding, shift);
        const i32 y_scaled =
            (i32)asr_s64(y_adjusted_product + rounding, shift);
        const i32 z_scaled =
            (i32)asr_s64(z_adjusted_product + rounding, shift);
        const i32 w_scaled =
            (i32)asr_s64(w_adjusted_product + rounding, shift);

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

// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-scalar.h]

/**
  | The code below is adapted from Google's
  | gemmlowp library.
  | 
  | It is only used in QNNPACK unit tests
  | and comparative benchmarks, but not
  | the library itself.
  |
  | Copyright 2015 Google Inc. All Rights Reserved.
  |
  | Licensed under the Apache License, Version 2.0
  | (the "License"); you may not use this file
  | except in compliance with the License.
  |
  | You may obtain a copy of the License at
  |
  |     http://www.apache.org/licenses/LICENSE-2.0
  |
  | Unless required by applicable law or agreed to
  | in writing, software distributed under the
  | License is distributed on an "AS IS" BASIS,
  | WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
  | either express or implied.
  |
  | See the License for the specific language
  | governing permissions and limitations under the
  | License.
  */
#[inline] pub fn gemmlowp_scalar_vqrdmulh_s32(
    a: i32,
    b: i32) -> i32 {
    
    todo!();
        /*
            const bool overflow = a == b && a == INT32_MIN;
      const i64 ab_64 = (i64)a * (i64)b;
      const i32 nudge =
          (a ^ b) >= 0 ? INT32_C(0x40000000) : -INT32_C(0x3FFFFFFF);
      const i32 ab_x2_high32 = (i32)((ab_64 + nudge) / INT64_C(0x80000000));
      return overflow ? INT32_MAX : ab_x2_high32;
        */
}

#[inline] pub fn gemmlowp_scalar_rdivbypo2_s32(
    x:        i32,
    exponent: i32) -> i32 {

    todo!();
        /*
            const i32 mask = ((1 << exponent) - 1);
      const i32 remainder = x & mask;
      const i32 threshold = (mask >> 1) + (i32)(x < 0);
      return asr_s32(x, exponent) + (i32)(remainder > threshold);
        */
}


//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-scalar.c]

pub fn pytorch_qnnp_requantize_gemmlowp_scalar(
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

      /* Compute requantization parameters */
      const u32 multiplier =
          ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;
      const i32 exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;
      const i32 shift =
          -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
            exponent);

      const i32 smin = (i32)(u32)qmin;
      const i32 smax = (i32)(u32)qmax;
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        const i32 x_product = gemmlowp_scalar_vqrdmulh_s32(x, multiplier);
        const i32 y_product = gemmlowp_scalar_vqrdmulh_s32(y, multiplier);
        const i32 z_product = gemmlowp_scalar_vqrdmulh_s32(z, multiplier);
        const i32 w_product = gemmlowp_scalar_vqrdmulh_s32(w, multiplier);

        const i32 x_scaled = gemmlowp_scalar_rdivbypo2_s32(x_product, shift);
        const i32 y_scaled = gemmlowp_scalar_rdivbypo2_s32(y_product, shift);
        const i32 z_scaled = gemmlowp_scalar_rdivbypo2_s32(z_product, shift);
        const i32 w_scaled = gemmlowp_scalar_rdivbypo2_s32(w_product, shift);

        /* Add zero point to scaled value */
        const i32 x_biased = x_scaled + zero_point;
        const i32 y_biased = y_scaled + zero_point;
        const i32 z_biased = z_scaled + zero_point;
        const i32 w_biased = w_scaled + zero_point;

        /* Clamp scaled value with zero point between smin and smax */
        const i32 x_clamped =
            x_biased < smin ? smin : x_biased > smax ? smax : x_biased;
        const i32 y_clamped =
            y_biased < smin ? smin : y_biased > smax ? smax : y_biased;
        const i32 z_clamped =
            z_biased < smin ? smin : z_biased > smax ? smax : z_biased;
        const i32 w_clamped =
            w_biased < smin ? smin : w_biased > smax ? smax : w_biased;

        output[0] = (u8)x_clamped;
        output[1] = (u8)y_clamped;
        output[2] = (u8)z_clamped;
        output[3] = (u8)w_clamped;
        output += 4;
      }
        */
}

// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/fp32-scalar.c]

pub fn pytorch_qnnp_requantize_fp32_scalar_lrintf(
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

      const long lmin =
          (long)((i32)(u32)qmin - (i32)(u32)zero_point);
      const long lmax =
          (long)((i32)(u32)qmax - (i32)(u32)zero_point);
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        const float x_scaled = (float)x * scale;
        const float y_scaled = (float)y * scale;
        const float z_scaled = (float)z * scale;
        const float w_scaled = (float)w * scale;

        const long x_rounded = lrintf(x_scaled);
        const long y_rounded = lrintf(y_scaled);
        const long z_rounded = lrintf(z_scaled);
        const long w_rounded = lrintf(w_scaled);

        const i32 x_clamped = (i32)(
            x_rounded < lmin ? lmin : x_rounded > lmax ? lmax : x_rounded);
        const i32 y_clamped = (i32)(
            y_rounded < lmin ? lmin : y_rounded > lmax ? lmax : y_rounded);
        const i32 z_clamped = (i32)(
            z_rounded < lmin ? lmin : z_rounded > lmax ? lmax : z_rounded);
        const i32 w_clamped = (i32)(
            w_rounded < lmin ? lmin : w_rounded > lmax ? lmax : w_rounded);

        const i32 x_biased = x_clamped + (i32)(u32)zero_point;
        const i32 y_biased = y_clamped + (i32)(u32)zero_point;
        const i32 z_biased = z_clamped + (i32)(u32)zero_point;
        const i32 w_biased = w_clamped + (i32)(u32)zero_point;

        output[0] = (u8)x_biased;
        output[1] = (u8)y_biased;
        output[2] = (u8)z_biased;
        output[3] = (u8)w_biased;
        output += 4;
      }
        */
}

pub fn pytorch_qnnp_requantize_fp32_scalar_magic(
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

      const float fmin =
          (float)((i32)(u32)qmin - (i32)(u32)zero_point);
      const float fmax =
          (float)((i32)(u32)qmax - (i32)(u32)zero_point);
      const float fmagic = 12582912.0f;
      const i32 imagic = INT32_C(0x4B400000) - (i32)(u32)zero_point;
      for (; n != 0; n -= 4) {
        const i32 x = input[0];
        const i32 y = input[1];
        const i32 z = input[2];
        const i32 w = input[3];
        input += 4;

        const float x_scaled = (float)x * scale;
        const float y_scaled = (float)y * scale;
        const float z_scaled = (float)z * scale;
        const float w_scaled = (float)w * scale;

        const float x_clamped =
            x_scaled < fmin ? fmin : x_scaled > fmax ? fmax : x_scaled;
        const float y_clamped =
            y_scaled < fmin ? fmin : y_scaled > fmax ? fmax : y_scaled;
        const float z_clamped =
            z_scaled < fmin ? fmin : z_scaled > fmax ? fmax : z_scaled;
        const float w_clamped =
            w_scaled < fmin ? fmin : w_scaled > fmax ? fmax : w_scaled;

        const i32 x_biased = (i32)fp32_to_bits(x_clamped + fmagic) - imagic;
        const i32 y_biased = (i32)fp32_to_bits(y_clamped + fmagic) - imagic;
        const i32 z_biased = (i32)fp32_to_bits(z_clamped + fmagic) - imagic;
        const i32 w_biased = (i32)fp32_to_bits(w_clamped + fmagic) - imagic;

        output[0] = (u8)x_biased;
        output[1] = (u8)y_biased;
        output[2] = (u8)z_biased;
        output[3] = (u8)w_biased;
        output += 4;
      }
        */
}

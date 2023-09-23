// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/requantization.h]

#[inline] pub fn pytorch_qnnp_compute_scalar_requantization_params(
        scale:      f32,
        zero_point: u8,
        min:        u8,
        max:        u8) -> PyTorchQnnpQ31RequantizationParams {
    
    todo!();
        /*
            /* Compute requantization parameters */
      assert(scale < 1.0f);
      assert(scale >= 0x1.0p-32f);
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

      union pytorch_qnnp_q31_requantization_params params;
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      params.scalar.multiplier = multiplier;
      params.scalar.remainder_mask = (i32)remainder_mask;
      params.scalar.remainder_threshold = (i32)remainder_threshold;
      params.scalar.shift = (u32)shift;
      params.scalar.min_less_zero_point =
          (i32)(u32)min - (i32)(u32)zero_point;
      params.scalar.max_less_zero_point =
          (i32)(u32)max - (i32)(u32)zero_point;
      params.scalar.zero_point = (i32)(u32)zero_point;
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_scalar_fp32_requantization_params(
    scales:     *mut f32,
    zero_point: u8,
    min:        u8,
    max:        u8) -> PyTorchQnnpFp32RequantizationParams {

    todo!();
        /*
            union pytorch_qnnp_fp32_requantization_params params;
      params.scalar.scales = scales;
      params.scalar.output_zero_point = zero_point;
      params.scalar.output_max = max;
      params.scalar.output_min = min;
      params.scalar.min_less_zero_point = ((float)((i32)(u32)min -
          (i32)(u32)zero_point));
      params.scalar.max_less_zero_point = ((float)((i32)(u32)max -
          (i32)(u32)zero_point));
      params.scalar.magic = 12582912.0f;
      params.scalar.magic_less_zero_point = (INT32_C(0x4B400000) -
          (i32)(u32)zero_point);
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_requantization_params(
    scale:      f32,
    zero_point: u8,
    min:        u8,
    max:        u8) -> PyTorchQnnpQ31RequantizationParams {
    
    todo!();
        /*
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

      union pytorch_qnnp_q31_requantization_params params;
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      params.sse2.multiplier[0] = multiplier;
      params.sse2.multiplier[1] = multiplier;
      params.sse2.multiplier[2] = multiplier;
      params.sse2.multiplier[3] = multiplier;
      params.sse2.rounding[0] = UINT64_C(0x40000000);
      params.sse2.rounding[1] = UINT64_C(0x40000000);
      params.sse2.remainder_mask[0] = (i32)remainder_mask;
      params.sse2.remainder_mask[1] = (i32)remainder_mask;
      params.sse2.remainder_mask[2] = (i32)remainder_mask;
      params.sse2.remainder_mask[3] = (i32)remainder_mask;
      params.sse2.remainder_threshold[0] = (i32)remainder_threshold;
      params.sse2.remainder_threshold[1] = (i32)remainder_threshold;
      params.sse2.remainder_threshold[2] = (i32)remainder_threshold;
      params.sse2.remainder_threshold[3] = (i32)remainder_threshold;
      params.sse2.shift[0] = (u64)(u32)shift;
      params.sse2.shift[1] = (u64)(u32)shift;
      for (u32 i = 0; i < 8; i++) {
        params.sse2.zero_point[i] = (i16)(u16)zero_point;
      }
      for (u32 i = 0; i < 16; i++) {
        params.sse2.max[i] = max;
        params.sse2.min[i] = min;
      }
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      params.neon.multiplier = multiplier;
      params.neon.right_shift = -shift;
      params.neon.zero_point = (i16)(u16)zero_point;
      params.neon.max = max;
      params.neon.min = min;
    #else
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      params.scalar.multiplier = multiplier;
      params.scalar.remainder_mask = (i32)remainder_mask;
      params.scalar.remainder_threshold = (i32)remainder_threshold;
      params.scalar.shift = (u32)shift;
      params.scalar.min_less_zero_point =
          (i32)(u32)min - (i32)(u32)zero_point;
      params.scalar.max_less_zero_point =
          (i32)(u32)max - (i32)(u32)zero_point;
      params.scalar.zero_point = (i32)(u32)zero_point;
    #endif
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_conv_quantization_params(
    input_zero_point:      u8,
    kernel_zero_points:    *const u8,
    requantization_scales: *const f32,
    output_zero_point:     u8,
    output_min:            u8,
    output_max:            u8) -> PyTorchQnnpConvQuantizationParams {
    
    todo!();
        /*
            union pytorch_qnnp_conv_quantization_params params;
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
      params.sse2.kernel_zero_points = kernel_zero_points;
      for (u32 i = 0; i < 8; i++) {
        params.sse2.input_zero_point[i] = (i16)(u16)input_zero_point;
      }
      params.sse2.requantization_scales = requantization_scales;
      for (u32 i = 0; i < 8; i++) {
        params.sse2.output_zero_point[i] = (i16)(u16)output_zero_point;
      }
      for (u32 i = 0; i < 16; i++) {
        params.sse2.output_max[i] = output_max;
        params.sse2.output_min[i] = output_min;
      }
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      params.neon.input_zero_point = (i16)(u16)input_zero_point;
      params.neon.kernel_zero_points = kernel_zero_points;
      params.neon.requantization_scales = requantization_scales;
      params.neon.output_zero_point = (i16)(u16)output_zero_point;
      params.neon.output_max = output_max;
      params.neon.output_min = output_min;
      params.neon.vfmin = ((float)((i32)(u32)output_min -
          (i32)(u32)output_zero_point));
      params.neon.vfmax = ((float)((i32)(u32)output_max -
          (i32)(u32)output_zero_point));
      params.neon.vfmagic = 12582912.0f;
      params.neon.vimagic = (INT32_C(0x4B400000) -
          (i32)(u32)output_zero_point);
    #else
      params.scalar.input_zero_point = (i32)(u32)input_zero_point;
      params.scalar.kernel_zero_points = kernel_zero_points;
      params.scalar.requantization_scales = requantization_scales;
      params.scalar.output_min_less_zero_point =
          (i32)(u32)output_min - (i32)(u32)output_zero_point;
      params.scalar.output_max_less_zero_point =
          (i32)(u32)output_max - (i32)(u32)output_zero_point;
      params.scalar.output_zero_point = (i32)(u32)output_zero_point;
    #endif
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_avgpool_quantization_params(
    bias:              i32,
    scale:             f32,
    output_zero_point: u8,
    output_min:        u8,
    output_max:        u8) -> PyTorchQnnpAvgPoolQuantizationParams {
    
    todo!();
        /*
            /* Compute requantization parameters */
      assert(scale >= 0x1.0p-32f);
      assert(scale < 256.0f);

      union pytorch_qnnp_avgpool_quantization_params params;
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
      params.sse2.bias[0] = bias;
      params.sse2.bias[1] = bias;
      params.sse2.bias[2] = bias;
      params.sse2.bias[3] = bias;
      params.sse2.scale[0] = scale;
      params.sse2.scale[1] = scale;
      params.sse2.scale[2] = scale;
      params.sse2.scale[3] = scale;
      for (u32 i = 0; i < 8; i++) {
        params.sse2.output_zero_point[i] = (i16)(u16)output_zero_point;
      }
      for (u32 i = 0; i < 16; i++) {
        params.sse2.output_max[i] = output_max;
        params.sse2.output_min[i] = output_min;
      }
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      params.neon.bias = bias;
      params.neon.scale = scale;
      params.neon.output_zero_point = (i16)(u16)output_zero_point;
      params.neon.output_max = output_max;
      params.neon.output_min = output_min;
      params.neon.vfmin = ((float)((i32)(u32)output_min -
          (i32)(u32)output_zero_point));
      params.neon.vfmax = ((float)((i32)(u32)output_max -
          (i32)(u32)output_zero_point));
      params.neon.vfmagic = 12582912.0f;
      params.neon.vimagic = (INT32_C(0x4B400000) -
          (i32)(u32)output_zero_point);
    #else
      params.scalar.bias = bias;
      params.scalar.scale = scale;
      params.scalar.output_zero_point = (i32)(u32)output_zero_point;
      params.scalar.output_max = (i32)(u32)output_max;
      params.scalar.output_min = (i32)(u32)output_min;
    #endif
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_scalar_avgpool_quantization_params(
    bias:              i32,
    scale:             f32,
    output_zero_point: u8,
    output_min:        u8,
    output_max:        u8) -> PyTorchQnnpAvgPoolQuantizationParams {
    
    todo!();
        /*
            /* Compute requantization parameters */
      assert(scale >= 0x1.0p-32f);
      assert(scale < 256.0f);

      union pytorch_qnnp_avgpool_quantization_params params;
      params.scalar.bias = bias;
      params.scalar.scale = scale;
      params.scalar.output_zero_point = (i32)(u32)output_zero_point;
      params.scalar.output_max = (i32)(u32)output_max;
      params.scalar.output_min = (i32)(u32)output_min;
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_u8_clamping_params(
    output_min: u8,
    output_max: u8) -> PyTorchQnnpU8ClampingParams {
    
    todo!();
        /*
            assert(output_min <= output_max);

      union pytorch_qnnp_u8_clamping_params params;
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
      for (u32 i = 0; i < 16; i++) {
        params.sse2.output_max[i] = output_max;
        params.sse2.output_min[i] = output_min;
      }
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      params.neon.output_max = output_max;
      params.neon.output_min = output_min;
    #else
      params.scalar.output_min = (i32)(u32)output_min;
      params.scalar.output_max = (i32)(u32)output_max;
    #endif
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_add_quantization_params(
    a_zero_point:      u8,
    b_zero_point:      u8,
    output_zero_point: u8,
    a_output_scale:    f32,
    b_output_scale:    f32,
    output_min:        u8,
    output_max:        u8) -> PyTorchQnnpAddQuantizationParams {
    
    todo!();
        /*
            assert(a_output_scale >= 0x1.0p-14f);
      assert(b_output_scale >= 0x1.0p-14f);
      assert(a_output_scale < 0x1.0p+8f);
      assert(b_output_scale < 0x1.0p+8f);

      /* Compute requantization parameters */
      const float max_output_scale =
          a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
      assert(max_output_scale >= 0x1.0p-14f);
      assert(max_output_scale < 0x1.0p+8f);
      const u32 max_scale_bits = fp32_to_bits(max_output_scale);
      const i32 max_scale_exponent = (i32)(max_scale_bits >> 23) - 127;
      /* Shift is in [13, 31] range */
      const u32 shift = (u32)(21 - max_scale_exponent);
      assert(shift < 32);
      assert(shift >= 13);

      const float scale_multiplier =
          fp32_from_bits((u32)(21 - max_scale_exponent + 127) << 23);

      /* Multipliers are in [0, 2**22) range, largest multiplier is in [2**21,
       * 2**22) range */
      const u32 a_multiplier =
          (u32)(i32)lrintf(a_output_scale * scale_multiplier);
      const u32 b_multiplier =
          (u32)(i32)lrintf(b_output_scale * scale_multiplier);
      assert(
          (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
          UINT32_C(0x00200000));
      assert(a_multiplier < UINT32_C(0x00400000));
      assert(b_multiplier < UINT32_C(0x00400000));

      union pytorch_qnnp_add_quantization_params params;
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      const i32 zero_point_product = (i32) -
          (a_multiplier * (u32)a_zero_point +
           b_multiplier * (u32)b_zero_point);
      for (u32 i = 0; i < 4; i++) {
        params.sse2.zero_point_product[i] = zero_point_product;
      }
      for (u32 i = 0; i < 8; i++) {
        params.sse2.y_zero_point[i] = (i16)(u16)output_zero_point;
      }
      for (u32 i = 0; i < 8; i++) {
        params.sse2.a_multiplier_lo[i] = (u16)(u32)a_multiplier;
        params.sse2.a_multiplier_hi[i] = (u16)((u32)a_multiplier >> 16);
        params.sse2.b_multiplier_lo[i] = (u16)(u32)b_multiplier;
        params.sse2.b_multiplier_hi[i] = (u16)((u32)b_multiplier >> 16);
      }
      params.sse2.a_multiplier = a_multiplier;
      params.sse2.b_multiplier = b_multiplier;
      for (u32 i = 0; i < 4; i++) {
        params.sse2.remainder_mask[i] = remainder_mask;
        params.sse2.remainder_threshold[i] = remainder_threshold;
      }
      params.sse2.shift = shift;
      for (u32 i = 0; i < 16; i++) {
        params.sse2.y_max[i] = output_max;
        params.sse2.y_min[i] = output_min;
      }
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
      params.neon.a_zero_point = a_zero_point;
      params.neon.b_zero_point = b_zero_point;
      params.neon.y_zero_point = (i16)(u16)output_zero_point;
      params.neon.a_multiplier = (i32)a_multiplier;
      params.neon.b_multiplier = (i32)b_multiplier;
      params.neon.right_shift = (i32)-shift;
      params.neon.y_max = output_max;
      params.neon.y_min = output_min;
    #else
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      params.scalar.zero_point_product = (i32) -
          (a_multiplier * (u32)a_zero_point +
           b_multiplier * (u32)b_zero_point);
      params.scalar.a_multiplier = a_multiplier;
      params.scalar.b_multiplier = b_multiplier;
      params.scalar.remainder_mask = (i32)remainder_mask;
      params.scalar.remainder_threshold = (i32)remainder_threshold;
      params.scalar.shift = shift;
      params.scalar.y_zero_point = (i32)(u32)output_zero_point;
      params.scalar.y_max = (i32)(u32)output_max;
      params.scalar.y_min = (i32)(u32)output_min;
    #endif
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_compute_scalar_add_quantization_params(
    a_zero_point:      u8,
    b_zero_point:      u8,
    output_zero_point: u8,
    a_output_scale:    f32,
    b_output_scale:    f32,
    output_min:        u8,
    output_max:        u8) -> PyTorchQnnpAddQuantizationParams {

    todo!();
    /*
       assert(a_output_scale >= 0x1.0p-10f);
      assert(b_output_scale >= 0x1.0p-10f);
      assert(a_output_scale < 0x1.0p+8f);
      assert(b_output_scale < 0x1.0p+8f);

      /* Compute requantization parameters */
      const float max_output_scale =
          a_output_scale > b_output_scale ? a_output_scale : b_output_scale;
      assert(max_output_scale >= 0x1.0p-10f);
      assert(max_output_scale < 0x1.0p+8f);
      const u32 max_scale_bits = fp32_to_bits(max_output_scale);
      const i32 max_scale_exponent = (i32)(max_scale_bits >> 23) - 127;
      /* Shift is in [13, 31] range */
      const u32 shift = (u32)(21 - max_scale_exponent);
      assert(shift < 32);
      assert(shift >= 13);

      /* Multipliers are in [0, 2**22) range, largest multiplier is in [2**21,
       * 2**22) range */
      const u32 a_multiplier = (u32)(i32)lrintf(
          fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
      const u32 b_multiplier = (u32)(i32)lrintf(
          fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
      assert(
          (a_multiplier > b_multiplier ? a_multiplier : b_multiplier) >=
          UINT32_C(0x00200000));
      assert(a_multiplier < UINT32_C(0x00400000));
      assert(b_multiplier < UINT32_C(0x00400000));

      union pytorch_qnnp_add_quantization_params params;
      const u32 remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
      const u32 remainder_threshold = remainder_mask >> 1;
      params.scalar.zero_point_product = (i32) -
          (a_multiplier * (u32)a_zero_point +
           b_multiplier * (u32)b_zero_point);
      params.scalar.a_multiplier = a_multiplier;
      params.scalar.b_multiplier = b_multiplier;
      params.scalar.remainder_mask = (i32)remainder_mask;
      params.scalar.remainder_threshold = (i32)remainder_threshold;
      params.scalar.shift = shift;
      params.scalar.y_zero_point = (i32)(u32)output_zero_point;
      params.scalar.y_max = (i32)(u32)output_max;
      params.scalar.y_min = (i32)(u32)output_min;
      return params;
        */
}

#[inline] pub fn pytorch_qnnp_q31_requantize(
    n:      i32,
    params: PyTorchQnnpQ31RequantizationParams) -> u8 {

    todo!();
        /*
            const i64 product = (i64)n * (i64)params.scalar.multiplier;
      const i32 q31product =
          (i32)(u32)((u64)(product + INT64_C(0x40000000)) >> 31);
      const i32 remainder =
          (q31product & params.scalar.remainder_mask) - (i32)(n < 0);
      n = asr_s32(q31product, params.scalar.shift) +
          (i32)(remainder > params.scalar.remainder_threshold);
      if (n < params.scalar.min_less_zero_point) {
        n = params.scalar.min_less_zero_point;
      }
      if (n > params.scalar.max_less_zero_point) {
        n = params.scalar.max_less_zero_point;
      }

      return (u8)(n + params.scalar.zero_point);
        */
}

#[inline] pub fn pytorch_qnnp_fp32_requantize(
    n:                    i32,
    params:               PyTorchQnnpFp32RequantizationParams,
    output_channel_index: i32) -> u8 {
    
    todo!();
        /*
            const long lmin =
          (long)((i32)(u32)params.scalar.output_min -
              (i32)(u32)params.scalar.output_zero_point);
      const long lmax =
          (long)((i32)(u32)params.scalar.output_max -
              (i32)(u32)params.scalar.output_zero_point);

      const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
      const long n_rounded = lrintf(n_scaled);
      const i32 n_clamped = (i32)(
          n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);
      const i32 n_biased =
          n_clamped + (i32)(u32)params.scalar.output_zero_point;

      return (u8)n_biased;
        */
}

#[inline] pub fn pytorch_qnnp_fp32_requantize_magic(
    n:                    i32,
    params:               PyTorchQnnpFp32RequantizationParams,
    output_channel_index: i32) -> u8 {
    
    todo!();
        /*
            const float fmin = params.scalar.min_less_zero_point;
      const float fmax = params.scalar.max_less_zero_point;
      const float fmagic = params.scalar.magic;
      const i32 imagic = params.scalar.magic_less_zero_point;

      const float n_scaled = (float)n * params.scalar.scales[output_channel_index];
      const float n_clamped =
          n_scaled < fmin ? fmin : n_scaled > fmax ? fmax : n_scaled;
      const i32 n_biased = (i32)fp32_to_bits(n_clamped + fmagic) - imagic;

      return (u8)n_biased;
        */
}

#[inline] pub fn pytorch_qnnp_avgpool_quantize(
    n:      i32,
    params: PyTorchQnnpAvgPoolQuantizationParams) -> u8 {
    
    todo!();
        /*
            const float scaled_n = ((float)n)*params.scalar.scale;
      i32 n_rounded = (i32)lrintf(scaled_n) + params.scalar.output_zero_point;

      const i32 lmin =
          (i32)(u32)params.scalar.output_min;
      const i32 lmax =
          (i32)(u32)params.scalar.output_max;

      n_rounded = (
          n_rounded < lmin ? lmin : n_rounded > lmax ? lmax : n_rounded);

      return (u8)n_rounded;
        */
}

#[inline] pub fn pytorch_qnnp_add_quantize(
    a:      u8,
    b:      u8,
    params: PyTorchQnnpAddQuantizationParams) -> u8 {
    
    todo!();
        /*
            /* Multiply by factors and accumulate products */
      i32 acc = params.scalar.zero_point_product +
          (i32)((u32)a * params.scalar.a_multiplier) +
          (i32)((u32)b * params.scalar.b_multiplier);

      /* Shift right and round */
      const i32 rem = (acc & params.scalar.remainder_mask) - (i32)(acc < 0);
      acc = asr_s32(acc, params.scalar.shift) +
          (i32)(rem > params.scalar.remainder_threshold);

      /* Clamp and add output zero point */
      i32 y = acc + params.scalar.y_zero_point;
      if (y >= params.scalar.y_max) {
        y = params.scalar.y_max;
      }
      if (y <= params.scalar.y_min) {
        y = params.scalar.y_min;
      }
      return (u8)y;
        */
}

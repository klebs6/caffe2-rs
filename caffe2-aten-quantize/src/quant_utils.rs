crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/quant_utils.h]

pub fn raw_uint_16to_fp16(value: u16) -> f32 {
    
    todo!();
        /*
            // Convert raw 16 bits half precision floating point number
        // to single precision floating point number.
        const unsigned short sign_bits = value >> 15;
        const unsigned short exponent_bits = value >> 10 & 0x1f;
        const unsigned short significand_bits = value & 0x3ff;

        const float sign = sign_bits ? -1 : 1;
        const float significand =
            1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10;
        const float exponent = exponent_bits - 0xf;

        return sign * ldexp(significand, exponent);
        */
}

pub fn check_and_saturate<T>(
        max_val: T,
        element: *mut T) -> bool {

    todo!();
        /*
            if (*element > max_val) {
        *element = max_val;
        return true;
      }
      if (*element < -max_val) {
        *element = -max_val;
        return true;
      }
      return false;
        */
}

/**
  | A structure to hold quantization parameters
  | 'scale' and 'zero_point'.
  |
  | The meaning of these values is as the constants
  | in the quantization equation
  |
  |   real_value = scale * (quantized_value - zero_point)
  |
  | In other words, 'zero_point' is the quantized
  | value that corresponds to the real value 0, and
  | 'scale' is the difference of real values
  | corresponding to consecutive quantized values.
  |
  */
pub struct TensorQuantizationParams {
    scale:      f64,
    zero_point: i32,
    precision:  i32,
}

#[inline] pub fn choose_quantization_params(
        min:                      f32,
        max:                      f32,
        qmin:                     i32,
        qmax:                     i32,
        preserve_sparsity:        bool,
        force_scale_power_of_two: bool,
        reduce_range:             bool) -> TensorQuantizationParams {

    let preserve_sparsity:        bool = preserve_sparsity.unwrap_or(false);
    let force_scale_power_of_two: bool = force_scale_power_of_two.unwrap_or(false);
    let reduce_range:             bool = reduce_range.unwrap_or(false);

    todo!();
        /*
            TORCH_CHECK(
          min <= max,
          "In ChooseQuantizationParams, min should be less than or equal to max");

      if (reduce_range) {
        qmin = qmin/2;
        qmax = qmax/2;
      }
      if (min < 0 && max > 0 && preserve_sparsity) {
        int symmetric_qmin = -((qmax - qmin) / 2 + 1);
        int symmetric_qmax = (qmax - qmin) / 2;
        double max_scale =
            max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
        min = max_scale * symmetric_qmin;
        max = max_scale * symmetric_qmax;
      }

      // We extend the [min, max] interval to ensure that it contains 0.
      // Otherwise, we would not meet the requirement that 0 be an exactly
      // representable value.
      min = min(min, 0.f);
      max = max(max, 0.f);

      TORCH_CHECK(
          qmin < qmax,
          "In ChooseQuantizationParams, qmin should be less than qmax");

      // Use double precision for intermediate computation but use single precision
      // in final number to reflect the actual number used during quantization.
      double scale = (static_cast<double>(max) - min) / (qmax - qmin);
      // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
      // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
      // infinity because some of fbgemm code pre-computes scale's reciprocal to do
      // multiplication instead of division in the time critical part of code.
      if (float(scale) == 0.0f || isinf(1.0f / float(scale))) {
        scale = 0.1;
      }
      TORCH_CHECK(scale > 0, "quantization scale should be > 0");

      if (force_scale_power_of_two) {
        if (scale < 1) {
          scale = 1.0 / (1 << static_cast<int>(floor(log(1.0 / scale) / log(2))));
        } else {
          scale = 1 << static_cast<int>(ceil(log(scale) / log(2)));
        }
      }

      // Zero-point computation.
      // First the initial floating-point computation. The zero-point can be
      // determined from solving an affine equation for any known pair
      // (real value, corresponding quantized value).
      // We know two such pairs: (rmin, qmin) and (rmax, qmax).
      // The arithmetic error on the zero point computed from either pair
      // will be roughly machine_epsilon * (sum of absolute values of terms)
      // so we want to use the variant that adds the smaller terms.
      double zero_point_from_min = qmin - min / static_cast<double>(scale);
      double zero_point_from_max = qmax - max / static_cast<double>(scale);
      double zero_point_from_min_error =
          abs(qmin) - abs(min / static_cast<double>(scale));
      double zero_point_from_max_error =
          abs(qmax) - abs(max / static_cast<double>(scale));
      double initial_zero_point =
          zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

      // for symmetric quantization (preserve_sparsity == true), we force zero_point
      // to be a middle value between qmin and qmax.
      // If either min or max is 0, then we just use 0 as zero_point.
      if (min < 0 && max > 0 && preserve_sparsity) {
        const auto midpoint = qmin + (qmax - qmin) / 2;  // Overflow-safe midpoint
        initial_zero_point = midpoint + 1;
      }

      // Now we need to nudge the zero point to be an integer
      // (our zero points are integer, and this is motivated by the requirement
      // to be able to represent the real value "0" exactly as a quantized value,
      // which is required in multiple places, for example in Im2col with zero
      // padding).
      i32 nudged_zero_point = 0;
      if (initial_zero_point < qmin) {
        nudged_zero_point = qmin;
      } else if (initial_zero_point > qmax) {
        nudged_zero_point = qmax;
      } else {
        nudged_zero_point = nearbyint(initial_zero_point);
      }

      TensorQuantizationParams result;
      result.scale = scale;
      result.zero_point = nudged_zero_point;
      return result;
        */
}

pub const K_CONV1D_SQUEEZE_DIM: i64 = 0;

/**
  | This function helps to convert the Conv1D
  | dimensions usable by the Conv2d op.
  |
  */
pub fn make_arg_for_conv1d(
        arg:        &TorchList<i64>,
        base_value: i64) -> TorchList<i64> {
    
    todo!();
        /*
            TORCH_CHECK(arg.size() > 0, "Argument must have elements.");
      TorchList<i64> result({arg.get(0), base_value});
      if (arg.size() == 1) {
        result[1] = arg.get(0);
      } else {
        result[1] = arg.get(1);
      }
      result[kConv1dSqueezeDim] = base_value;
      return result;
        */
}

/**
  | The range for using FP16 quantization of
  | weights requires that the elements should be in
  | the range of [5.96e-8, 65504].
  |
  | If it is out of range, then the number will be
  | saturated to max or min representable values by
  | FP16.
  |
  */
#[inline] pub fn handle_weights_saturation(
        N:      i64,
        weight: *mut f32)  {
    
    todo!();
        /*
            const float kFp16Max = RawUint16ToFp16(0x7BFF);
      bool found_out_of_range = false;
      for (i64 i = 0; i < N; ++i) {
        bool saturate = CheckAndSaturate<float>(kFp16Max, weight + i);
        if (saturate) {
          found_out_of_range = true;
        }
      }
      if (found_out_of_range) {
        TORCH_WARN("FOUND weight out of range ");
      }
        */
}

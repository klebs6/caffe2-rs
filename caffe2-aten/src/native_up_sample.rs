/*!
 | Note [compute_scales_value]
 | Note [area_pixel_compute_scale]
 | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | Interpolate with scale_factor can have different behaviors
 | depending on the value of recompute_scale_factor:
 |
 | - With recompute_scale_factor = True (current default behavior):
 | the scale_factor, when provided by the user, are used to calculate
 | the output size. The input size and the computed output_size
 | are then used to infer new values for the scales which are
 | used in the interpolation.  Because floating-point math is not exact,
 | this may be a different value from the user-supplied scales.
 |
 | - With recompute_scale_factor = False (which will be the default
 | behavior starting 1.5.0):
 | the behavior follows opencv logic, and the scales provided by
 | the user are the ones used in the interpolation calculations.
 |
 | If the scales are not provided or if they are provided but
 | recompute_scale_factor is set to True (default behavior), the scales
 | are computed from the input and the output size;
 |
 |
 | When the scales are inferred from the input and output sizes,
 | we view each pixel as an area, idx + 0.5 as its center index.
 | Here is an example formula in 1D case.
 | if align_corners: center of two corner pixel areas are preserved,
 |     (0.5, 0.5) -> (0.5, 0.5),
 |     (input_size - 0.5, 0.5) -> (output_size - 0.5)
 |     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
 |     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
 | if not align_corners: the whole range is scaled accordingly
 |     scale = input_size / output_size
 |     src_idx + 0.5 = scale * (dst_index + 0.5)
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UpSample.h]

#[inline] pub fn upsample_get_scale_value(
    scales: Option<&[f64]>,
    idx:    i32) -> Option<f64> {
    
    todo!();
        /*
            if (!scales) {
        return nullopt;
      }
      return scales->at(idx);
        */
}

pub type Scale = Option<f64>;

pub type UpsamplingNearest1d = fn(
        output:   &Tensor,
        input:    &Tensor,
        scales_w: Scale
) -> ();

pub type UpsamplingNearest2d = fn(
        output:   &Tensor,
        input:    &Tensor,
        scales_h: Scale,
        scales_w: Scale
) -> ();

pub type UpsamplingNearest3d = fn(
        output:   &Tensor,
        input:    &Tensor,
        scales_d: Scale,
        scales_h: Scale,
        scales_w: Scale
) -> ();

pub type UpsamplingLinear1d = fn(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_w:      Scale
) -> ();

pub type UpsamplingBilinear2d = fn(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_h:      Scale,
        scales_w:      Scale
) -> ();

pub type UpsamplingTrilinear3d = fn(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_d:      Scale,
        scales_h:      Scale,
        scales_w:      Scale
) -> ();

pub type UpsamplingBicubic2d = fn(
        output:        &Tensor,
        input:         &Tensor,
        align_corners: bool,
        scales_h:      Scale,
        scales_w:      Scale
) -> ();

declare_dispatch!{upsampling_nearest1d, upsample_nearest1d_kernel}
declare_dispatch!{upsampling_nearest2d, upsample_nearest2d_kernel}
declare_dispatch!{upsampling_nearest3d, upsample_nearest3d_kernel}
declare_dispatch!{upsampling_nearest1d, upsample_nearest1d_backward_kernel}
declare_dispatch!{upsampling_nearest2d, upsample_nearest2d_backward_kernel}
declare_dispatch!{upsampling_nearest3d, upsample_nearest3d_backward_kernel}
declare_dispatch!{upsampling_linear1d, upsample_linear1d_kernel}
declare_dispatch!{upsampling_bilinear2d, upsample_bilinear2d_kernel}
declare_dispatch!{upsampling_trilinear3d, upsample_trilinear3d_kernel}
declare_dispatch!{upsampling_linear1d, upsample_linear1d_backward_kernel}
declare_dispatch!{upsampling_bilinear2d, upsample_bilinear2d_backward_kernel}
declare_dispatch!{upsampling_trilinear3d, upsample_trilinear3d_backward_kernel}
declare_dispatch!{upsampling_bicubic2d, upsample_bicubic2d_kernel}

pub fn upsample_1d_common_check(
        input_size:  &[i32],
        output_size: &[i32]) -> [i64; 3] {
    
    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 1,
          "It is expected output_size equals to 1, but got size ",
          output_size.size());

      TORCH_CHECK(
          input_size.size() == 3,
          "It is expected input_size equals to 3, but got size ",
          input_size.size());

      i64 output_width = output_size[0];

      i64 nbatch = input_size[0];
      i64 channels = input_size[1];
      i64 input_width = input_size[2];

      TORCH_CHECK(
          input_width > 0 && output_width > 0,
          "Input and output sizes should be greater than 0, but got input (W: ",
          input_width,
          ") and output (W: ",
          output_width,
          ")");

      return {nbatch, channels, output_width};
        */
}

pub fn upsample_2d_common_check(
    input_size:  &[i32],
    output_size: &[i32]) -> [i64; 4] {
    
    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 2,
          "It is expected output_size equals to 2, but got size ",
          output_size.size());

      TORCH_CHECK(
          input_size.size() == 4,
          "It is expected input_size equals to 4, but got size ",
          input_size.size());

      i64 output_height = output_size[0];
      i64 output_width = output_size[1];

      i64 nbatch = input_size[0];
      i64 channels = input_size[1];
      i64 input_height = input_size[2];
      i64 input_width = input_size[3];

      TORCH_CHECK(
          input_height > 0 && input_width > 0 && output_height > 0 &&
              output_width > 0,
          "Input and output sizes should be greater than 0,"
          " but got input (H: ",
          input_height,
          ", W: ",
          input_width,
          ") output (H: ",
          output_height,
          ", W: ",
          output_width,
          ")");

      return {nbatch, channels, output_height, output_width};
        */
}

pub fn upsample_3d_common_check(
        input_size:  &[i32],
        output_size: &[i32]) -> [i64; 5] {
    
    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 3,
          "It is expected output_size equals to 3, but got size ",
          output_size.size());

      TORCH_CHECK(
          input_size.size() == 5,
          "It is expected input_size equals to 5, but got size ",
          input_size.size());

      i64 output_depth = output_size[0];
      i64 output_height = output_size[1];
      i64 output_width = output_size[2];

      i64 nbatch = input_size[0];
      i64 channels = input_size[1];
      i64 input_depth = input_size[2];
      i64 input_height = input_size[3];
      i64 input_width = input_size[4];

      TORCH_CHECK(
          input_depth > 0 && input_height > 0 && input_width > 0 &&
              output_depth > 0 && output_height > 0 && output_width > 0,
          "Input and output sizes should be greater than 0, but got input (D: ",
          input_depth,
          ", H: ",
          input_height,
          ", W: ",
          input_width,
          ") output (D: ",
          output_depth,
          ", H: ",
          output_height,
          ", W: ",
          output_width,
          ")");

      return {nbatch, channels, output_depth, output_height, output_width};
        */
}

#[inline] pub fn upsample_2d_shape_check(
        input:         &Tensor,
        grad_output:   &Tensor,
        nbatch:        i64,
        nchannels:     i64,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
          input_height > 0 && input_width > 0 && output_height > 0 &&
              output_width > 0,
          "Input and output sizes should be greater than 0,"
          " but got input (H: ",
          input_height,
          ", W: ",
          input_width,
          ") output (H: ",
          output_height,
          ", W: ",
          output_width,
          ")");

      if (input.defined()) {
        // Allow for empty batch size but not other dimensions
        TORCH_CHECK(
                    (input.numel() != 0 ||
                     (input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0)
                     ) &&
                    input.dim() == 4,
                    "Non-empty 4D data tensor expected but got a tensor with sizes ",
                    input.sizes());
      } else if (grad_output.defined()) {
        check_dim_size(grad_output, 4, 0, nbatch);
        check_dim_size(grad_output, 4, 1, nchannels);
        check_dim_size(grad_output, 4, 2, output_height);
        check_dim_size(grad_output, 4, 3, output_width);
      }
        */
}

#[inline] pub fn compute_scales_value<Scalar>(
        scale:       Option<f64>,
        input_size:  i64,
        output_size: i64) -> Scalar {

    todo!();
        /*
            // see Note [compute_scales_value]
          // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
          return (scale.has_value() && scale.value() > 0.)
              ? static_cast<Scalar>(1.0 / scale.value())
              : (static_cast<Scalar>(input_size) / output_size);
        */
}

#[inline] pub fn area_pixel_compute_scale<Scalar>(
        input_size:    i64,
        output_size:   i64,
        align_corners: bool,
        scale:         Option<f64>) -> Scalar {

    todo!();
        /*
            // see Note [area_pixel_compute_scale]
      if (output_size > 1) {
        return align_corners
            ? static_cast<Scalar>(input_size - 1) / (output_size - 1)
            : compute_scales_value<Scalar>(scale, input_size, output_size);
      } else {
        return Scalar(0);
      }
        */
}

#[inline] pub fn area_pixel_compute_source_index<Scalar>(
        scale:         Scalar,
        dst_index:     i64,
        align_corners: bool,
        cubic:         bool) -> Scalar {

    todo!();
        /*
            if (align_corners) {
        return scale * dst_index;
      } else {
        Scalar src_idx = scale * (dst_index + 0.5) - 0.5;
        // [Note] Follow Opencv resize logic:
        // We allow negative src_idx here and later will use
        //   dx = src_idx - floorf(src_idx)
        // to compute the "distance"(which affects weights).
        // For linear modes, weight distribution doesn't matter
        // for negative indices as they use 2 pixels to interpolate.
        // For example, [-1, 0], they both use pixel 0 value so it
        // doesn't affect if we bound the src_idx to 0 or not.
        // TODO: Our current linear mode impls use unbound indices
        // where we should and then remove this cubic flag.
        // This matters in cubic mode, as we might need [-1, 0, 1, 2]
        // to interpolate and the weights can be affected.
        return (!cubic && src_idx < 0) ? Scalar(0) : src_idx;
      }
        */
}

#[inline] pub fn nearest_neighbor_compute_source_index(
        scale:      f32,
        dst_index:  i64,
        input_size: i64) -> i64 {
    
    todo!();
        /*
            const i64 src_index =
          min(static_cast<i64>(floorf(dst_index * scale)), input_size - 1);
      return src_index;
        */
}



pub fn upsample_get_value_bounded<Scalar>(
        data:   *mut Scalar,
        width:  i64,
        height: i64,
        x:      i64,
        y:      i64) -> Scalar {

    todo!();
        /*
            i64 access_x = max(min(x, width - 1), static_cast<i64>(0));
      i64 access_y = max(min(y, height - 1), static_cast<i64>(0));
      return data[access_y * width + access_x];
        */
}

pub fn upsample_increment_value_bounded<Scalar>(
        data:   *mut Scalar,
        width:  i64,
        height: i64,
        x:      i64,
        y:      i64,
        value:  Scalar)  {

    todo!();
        /*
            i64 access_x = max(min(x, width - 1), static_cast<i64>(0));
      i64 access_y = max(min(y, height - 1), static_cast<i64>(0));
      data[access_y * width + access_x] += value;
        */
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
#[inline] pub fn cubic_convolution1<Scalar>(
        x: Scalar,
        A: Scalar) -> Scalar {

    todo!();
        /*
            return ((A + 2) * x - (A + 3)) * x * x + 1;
        */
}


#[inline] pub fn cubic_convolution2<Scalar>(
        x: Scalar,
        A: Scalar) -> Scalar {

    todo!();
        /*
            return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
        */
}



#[inline] pub fn get_cubic_upsample_coefficients<Scalar>(
        coeffs: [Scalar; 4],
        t:      Scalar)  {

    todo!();
        /*
            Scalar A = -0.75;

      Scalar x1 = t;
      coeffs[0] = cubic_convolution2<Scalar>(x1 + 1.0, A);
      coeffs[1] = cubic_convolution1<Scalar>(x1, A);

      // opposite coefficients
      Scalar x2 = 1.0 - t;
      coeffs[2] = cubic_convolution1<Scalar>(x2, A);
      coeffs[3] = cubic_convolution2<Scalar>(x2 + 1.0, A);
        */
}



#[inline] pub fn cubic_interp1d<Scalar>(
        x0: Scalar,
        x1: Scalar,
        x2: Scalar,
        x3: Scalar,
        t:  Scalar) -> Scalar {

    todo!();
        /*
            Scalar coeffs[4];
      get_cubic_upsample_coefficients<Scalar>(coeffs, t);

      return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
        */
}



#[inline] pub fn compute_source_index_and_lambda<Scalar>(
        input_index0:  &mut i64,
        input_index1:  &mut i64,
        lambda0:       &mut Scalar,
        lambda1:       &mut Scalar,
        ratio:         Scalar,
        output_index:  i64,
        input_size:    i64,
        output_size:   i64,
        align_corners: bool)  {

    todo!();
        /*
            if (output_size == input_size) {
        // scale_factor = 1, simply copy
        input_index0 = output_index;
        input_index1 = output_index;
        lambda0 = static_cast<Scalar>(1);
        lambda1 = static_cast<Scalar>(0);
      } else {
        const Scalar real_input_index = area_pixel_compute_source_index<Scalar>(
            ratio, output_index, align_corners, /*cubic=*/false);
        input_index0 = static_cast<i64>(real_input_index);
        i64 offset = (input_index0 < input_size - 1) ? 1 : 0;
        input_index1 = input_index0 + offset;
        lambda1 = real_input_index - input_index0;
        lambda0 = static_cast<Scalar>(1.) - lambda1;
      }
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UpSample.cpp]

pub fn upsample_compute_output_size(

    // Full input tensor size.
    input_size:    &[i32],

    output_size:   Option<&[i32]>,
    scale_factors: Option<&[f64]>) -> SmallVector<i64,3> {

    todo!();
        /*
      int spatial_dimensions = input_size.size() - 2;
      if (output_size) {
        TORCH_CHECK(!scale_factors, "Must specify exactly one of output_size and scale_factors");
        TORCH_CHECK(output_size->size() == spatial_dimensions);
        return {output_size->data(), output_size->data() + output_size->size()};
      }
      if (scale_factors) {
        TORCH_CHECK(!output_size, "Must specify exactly one of output_size and scale_factors");
        TORCH_CHECK(scale_factors->size() == spatial_dimensions);
        SmallVector<i64, 3> ret;
        for (int i = 0; i < spatial_dimensions; ++i) {
          ret.push_back(static_cast<double>(input_size[i+2]) * scale_factors.value()[i]);
        }
        return ret;
      }
      TORCH_CHECK(false, "Must specify exactly one of output_size and scale_factors");
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UpSampleBicubic2d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(upsample_bicubic2d) (
      const Tensor& input, IntArrayRef output_size, bool align_corners, optional<double> scales_h, optional<double> scales_w
    ) {
      auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

      // Allow for empty batch size but not other dimensions
      TORCH_CHECK(
          input.numel() != 0 || multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
          "Non-empty 4D data tensor expected but got a tensor with sizes ",
          input.sizes());

      set_output(full_output_size, input.options());
    }
    */
}


lazy_static!{
    /*
    TORCH_META_FUNC(upsample_bicubic2d_backward) (
      const Tensor& grad_output,
      IntArrayRef output_size,
      IntArrayRef input_size,
      bool align_corners,
      optional<double> scales_h,
      optional<double> scales_w
    ) {
      auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

      TORCH_CHECK(
          grad_output.dim() == 4,
          "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

      for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(
            grad_output.size(i) == full_output_size[i],
            "Expected grad_output to have the same shape as output;",
            " output.size(", i, ") = ", full_output_size[i],
            " but got grad_output.size(", i, ") = ", grad_output.size(i));
      }

      set_output(input_size, grad_output.options());
    }
    */
}

pub fn upsample_bicubic2d_backward_out_frame<scalar_t>(
        odata:         *mut Scalar,
        idata:         *mut Scalar,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        nbatch:        i64,
        channels:      i64,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {

    todo!();
        /*
            channels = channels * nbatch;

      // Special case: input/output same size, just copy
      if (input_height == output_height && input_width == output_width) {
        for (i64 output_y = 0; output_y < output_height; output_y++) {
          for (i64 output_x = 0; output_x < output_width; output_x++) {
            scalar_t* in = &idata[output_y * input_width + output_x];
            scalar_t* out = &odata[output_y * output_width + output_x];
            for (i64 c = 0; c < channels; ++c) {
              in[0] = out[0];
              in += input_width * input_height;
              out += output_width * output_height;
            }
          }
        }
        return;
      }

      const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
          input_height, output_height, align_corners, scales_h);
      const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
          input_width, output_width, align_corners, scales_w);

      for (i64 output_y = 0; output_y < output_height; output_y++) {
        for (i64 output_x = 0; output_x < output_width; output_x++) {
          scalar_t* in = idata;
          scalar_t* out = odata;

          const scalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
          i64 input_x = floorf(real_x);
          scalar_t t_x = real_x - input_x;

          const scalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
          i64 input_y = floorf(real_y);
          scalar_t t_y = real_y - input_y;

          scalar_t x_coeffs[4];
          scalar_t y_coeffs[4];

          get_cubic_upsample_coefficients<scalar_t>(x_coeffs, t_x);
          get_cubic_upsample_coefficients<scalar_t>(y_coeffs, t_y);

          for (i64 c = 0; c < channels; c++) {
            scalar_t out_value = out[output_y * output_width + output_x];

            for (i64 i = 0; i < 4; i++) {
              for (i64 j = 0; j < 4; j++) {
                upsample_increment_value_bounded<scalar_t>(
                    in,
                    input_width,
                    input_height,
                    input_x - 1 + i,
                    input_y - 1 + j,
                    out_value * y_coeffs[j] * x_coeffs[i]);
              }
            }

            in += input_width * input_height;
            out += output_width * output_height;
          }
        }
      }
        */
}


pub fn upsample_bicubic2d_backward_kernel(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        output_size:   &[i32],
        input_size:    &[i32],
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            i64 output_height = output_size[0];
      i64 output_width = output_size[1];

      i64 nbatch = input_size[0];
      i64 channels = input_size[1];
      i64 input_height = input_size[2];
      i64 input_width = input_size[3];

      auto grad_output = grad_output_.contiguous();

      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          grad_output.scalar_type(), "upsample_bicubic2d_backward", [&] {
            scalar_t* idata = grad_input.data_ptr<scalar_t>();
            scalar_t* odata = grad_output.data_ptr<scalar_t>();

            upsample_bicubic2d_backward_out_frame<scalar_t>(
                odata,
                idata,
                input_height,
                input_width,
                output_height,
                output_width,
                nbatch,
                channels,
                align_corners,
                scales_h,
                scales_w);
          });
        */
}


lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_bicubic2d_out_cpu) (
        const Tensor& input,
        IntArrayRef output_size,
        bool align_corners,
        optional<double> scales_h,
        optional<double> scales_w,
        const Tensor& output
    ) {
      upsample_bicubic2d_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_cpu) (
        const Tensor& grad_output,
        IntArrayRef output_size,
        IntArrayRef input_size,
        bool align_corners,
        optional<double> scales_h,
        optional<double> scales_w,
        const Tensor& grad_input
    ) {
      grad_input.zero_();
      upsample_bicubic2d_backward_kernel(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
    */
}

pub fn upsample_bicubic2d(
        input:         &Tensor,
        output_size:   Option<&[i32]>,
        align_corners: bool,
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
      auto scale_h = get_scale_value(scale_factors, 0);
      auto scale_w = get_scale_value(scale_factors, 1);
      return upsample_bicubic2d(input, osize, align_corners, scale_h, scale_w);
        */
}

pub fn upsample_bicubic2d_backward(
        grad_output:   &Tensor,
        output_size:   Option<&[i32]>,
        input_size:    &[i32],
        align_corners: bool,
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input_size, output_size, scale_factors);
      auto scale_h = get_scale_value(scale_factors, 0);
      auto scale_w = get_scale_value(scale_factors, 1);
      return upsample_bicubic2d_backward(grad_output, osize, input_size, align_corners, scale_h, scale_w);
        */
}

define_dispatch!{upsample_bicubic2d_kernel}

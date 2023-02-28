crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UpSampleNearest2d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(upsample_nearest2d) (
      const Tensor& input, IntArrayRef output_size, optional<double> scales_h, optional<double> scales_w
    ) {
      auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

      // Allow for empty batch size but not other dimensions
      TORCH_CHECK(
          input.numel() != 0 || multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
          "Non-empty 4D data tensor expected but got a tensor with sizes ",
          input.sizes());

      set_output(full_output_size, input.options().memory_format(input.suggest_memory_format()));
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(upsample_nearest2d_backward) (
      const Tensor& grad_output,
      IntArrayRef output_size,
      IntArrayRef input_size,
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

lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_nearest2d_out_cpu) (
        const Tensor& input,
        IntArrayRef output_size,
        optional<double> scales_h,
        optional<double> scales_w,
        const Tensor& output
    ) {
      upsample_nearest2d_kernel(kCPU, output, input, scales_h, scales_w);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cpu) (
        const Tensor& grad_output,
        IntArrayRef output_size,
        IntArrayRef input_size,
        optional<double> scales_h,
        optional<double> scales_w,
        const Tensor& grad_input) {
      grad_input.zero_();
      upsample_nearest2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);
    }
    */
}

pub fn upsample_nearest2d(
        input:         &Tensor,
        output_size:   Option<&[i32]>,
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
      auto scale_h = get_scale_value(scale_factors, 0);
      auto scale_w = get_scale_value(scale_factors, 1);
      return upsample_nearest2d(input, osize, scale_h, scale_w);
        */
}

pub fn upsample_nearest2d_backward(
        grad_output:   &Tensor,
        output_size:   Option<&[i32]>,
        input_size:    &[i32],
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input_size, output_size, scale_factors);
      auto scale_h = get_scale_value(scale_factors, 0);
      auto scale_w = get_scale_value(scale_factors, 1);
      return upsample_nearest2d_backward(grad_output, osize, input_size, scale_h, scale_w);
        */
}

define_dispatch!{upsample_nearest2d_kernel}
define_dispatch!{upsample_nearest2d_backward_kernel}

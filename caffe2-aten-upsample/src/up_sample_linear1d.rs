/**
  | Adapted from interp.cpp from Caffe util by
  | Pauline Luc
  |
  | Originally developed by George Papandreou
  |
  */
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/UpSampleLinear1d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(upsample_linear1d) (
        const Tensor& input,
        IntArrayRef output_size,
        bool align_corners,
        optional<double> scales
    ) {
      auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

      // Allow for empty batch size but not other dimensions
      TORCH_CHECK(
          (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
          "Non-empty 3D data tensor expected but got a tensor with sizes ",
          input.sizes());

      set_output(full_output_size, input.options());
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(upsample_linear1d_backward) (
        const Tensor& grad_output,
        IntArrayRef output_size,
        IntArrayRef input_size,
        bool align_corners,
        optional<double> scales
    ) {
      auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

      TORCH_CHECK(
          input_size.size() == 3,
          "It is expected input_size equals to 3, but got size ",
          input_size.size());

      check_dim_size(grad_output, 3, 0, full_output_size[0]);
      check_dim_size(grad_output, 3, 1, full_output_size[1]);
      check_dim_size(grad_output, 3, 2, full_output_size[2]);

      set_output(input_size, grad_output.options());
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_linear1d_out_cpu) (
        const Tensor& input,
        IntArrayRef output_size,
        bool align_corners,
        optional<double> scales,
        const Tensor& output
    ) {
      upsample_linear1d_kernel(kCPU, output, input, align_corners, scales);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(upsample_linear1d_backward_out_cpu) (
        const Tensor& grad_output,
        IntArrayRef output_size,
        IntArrayRef input_size,
        bool align_corners,
        optional<double> scales,
        const Tensor& grad_input
    ) {
      grad_input.zero_();
      upsample_linear1d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales);
    }
    */
}

pub fn upsample_linear1d(
        input:         &Tensor,
        output_size:   Option<&[i32]>,
        align_corners: bool,
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
      auto scale_w = get_scale_value(scale_factors, 0);
      return upsample_linear1d(input, osize, align_corners, scale_w);
        */
}

pub fn upsample_linear1d_backward(
        grad_output:   &Tensor,
        output_size:   Option<&[i32]>,
        input_size:    &[i32],
        align_corners: bool,
        scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input_size, output_size, scale_factors);
      auto scale_w = get_scale_value(scale_factors, 0);
      return upsample_linear1d_backward(grad_output, osize, input_size, align_corners, scale_w);
        */
}

define_dispatch!{upsample_linear1d_kernel}
define_dispatch!{upsample_linear1d_backward_kernel}

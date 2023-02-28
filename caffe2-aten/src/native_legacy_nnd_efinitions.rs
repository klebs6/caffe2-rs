crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LegacyNNDefinitions.cpp]

pub fn thnn_conv_depthwise2d_out(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
        */
}

pub fn thnn_conv_depthwise2d(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
        */
}

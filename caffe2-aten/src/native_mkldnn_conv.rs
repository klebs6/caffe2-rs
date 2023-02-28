crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Conv.cpp]

#[cfg(not(feature = "mkldnn"))]
pub mod mkldnn_disabled {

    use super::*;

    pub fn mkldnn_convolution(
            input:    &Tensor,
            weight:   &Tensor,
            bias_opt: &Option<Tensor>,
            padding:  &[i32],
            stride:   &[i32],
            dilation: &[i32],
            groups:   i64) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_convolution_backward_input(
            input_size:   &[i32],
            grad_output:  &Tensor,
            weight:       &Tensor,
            padding:      &[i32],
            stride:       &[i32],
            dilation:     &[i32],
            groups:       i64,
            bias_defined: bool) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_convolution_backward_weights(
            weight_size:  &[i32],
            grad_output:  &Tensor,
            input:        &Tensor,
            padding:      &[i32],
            stride:       &[i32],
            dilation:     &[i32],
            groups:       i64,
            bias_defined: bool) -> (Tensor,Tensor) {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_convolution_backward(
            input:         &Tensor,
            grad_output_t: &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
            */
    }
}

#[cfg(feature = "mkldnn")]
pub mod mkldnn_enabled {

    use super::*;

    pub fn mkldnn_convolution(
            x:        &IDEEP::Tensor,
            w:        &IDEEP::Tensor,
            b:        &Option<IDEEP::Tensor>,
            padding:  &[i32],
            stride:   &[i32],
            dilation: &[i32],
            groups:   i64) -> IDEEP::Tensor {
        
        todo!();
            /*
                auto kernel_size = w.get_dims();

          std::vector<i64> input_size = x.get_dims();
          std::vector<i64> output_sizes =
              conv_output_size(input_size, kernel_size, padding, stride, dilation);

          ideep::tensor y;
          if (b.has_value()) {
            ideep::convolution_forward::compute(
                x,
                w,
                b.value(),
                {output_sizes.cbegin(), output_sizes.cend()},
                y,
                {stride.begin(), stride.end()},
                {dilation.begin(), dilation.end()},
                {padding.begin(), padding.end()},
                {padding.begin(), padding.end()},
                groups);
          } else {
            ideep::convolution_forward::compute(
                x,
                w,
                {output_sizes.cbegin(), output_sizes.cend()},
                y,
                {stride.begin(), stride.end()},
                {dilation.begin(), dilation.end()},
                {padding.begin(), padding.end()},
                {padding.begin(), padding.end()},
                groups);
          }
          return y;
            */
    }

    pub fn mkldnn_convolution(
            input:    &Tensor,
            weight:   &Tensor,
            bias_opt: &Option<Tensor>,
            padding:  &[i32],
            stride:   &[i32],
            dilation: &[i32],
            groups:   i64) -> Tensor {
        
        todo!();
            /*
                // See [Note: hacky wrapper removal for optional tensor]
          c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
          const Tensor& bias = *bias_maybe_owned;

          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_convolution: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }
          const ideep::tensor mkldnn_input = itensor_from_tensor(input);
          const ideep::tensor mkldnn_weight = itensor_from_tensor(weight);
          c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
          if (bias.defined()) {
            mkldnn_bias = itensor_from_tensor(bias);
          }

          ideep::tensor mkldnn_output = _mkldnn_convolution(
              mkldnn_input,
              mkldnn_weight,
              mkldnn_bias,
              padding,
              stride,
              dilation,
              groups);

          if (input.is_mkldnn()) {
            return new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                           input.options().device_opt());
          } else {
            return mkldnn_to_dense(
                new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                        input.options().device_opt()));
          }
            */
    }

    pub fn mkldnn_convolution_backward_input(
            input_size:   &[i32],
            grad_output:  &Tensor,
            weight:       &Tensor,
            padding:      &[i32],
            stride:       &[i32],
            dilation:     &[i32],
            groups:       i64,
            bias_defined: bool) -> Tensor {
        
        todo!();
            /*
                // for training case, grad_output can be cpu tensor or MKLDNN tensor,
          // but weight and bias always cpu tensor.
          auto mkldnn_grad_output = itensor_from_tensor(grad_output);
          auto mkldnn_weight = itensor_view_from_dense(weight);

          ideep::tensor mkldnn_grad_input;
          ideep::convolution_backward_data::compute(
              mkldnn_grad_output,
              mkldnn_weight,
              input_size.vec(),
              mkldnn_grad_input,
              stride.vec(),
              dilation.vec(),
              padding.vec(),
              padding.vec(),
              groups);

          if (grad_output.is_mkldnn()) {
            return new_with_itensor_mkldnn(std::move(mkldnn_grad_input),
                                           optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                           grad_output.options().device_opt());

          } else {
            return mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_input),
                                                           optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                                           grad_output.options().device_opt()));
          }
            */
    }

    pub fn mkldnn_convolution_backward_weights(
            weight_size:  &[i32],
            grad_output:  &Tensor,
            input:        &Tensor,
            padding:      &[i32],
            stride:       &[i32],
            dilation:     &[i32],
            groups:       i64,
            bias_defined: bool) -> (Tensor,Tensor) {
        
        todo!();
            /*
                // for training case, grad_output and input can be cpu tensor or MKLDNN tensor,
          // but weight and bias are always cpu tensor.
          const ideep::tensor mkldnn_grad_output = itensor_from_tensor(grad_output);
          const ideep::tensor mkldnn_input = itensor_from_tensor(input);

          ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
          if (bias_defined) {
            ideep::convolution_backward_weights::compute(
                mkldnn_input,
                mkldnn_grad_output,
                weight_size.vec(),
                mkldnn_grad_weight,
                mkldnn_grad_bias,
                stride.vec(),
                dilation.vec(),
                padding.vec(),
                padding.vec(),
                groups);
          } else {
            ideep::convolution_backward_weights::compute(
                mkldnn_input,
                mkldnn_grad_output,
                weight_size.vec(),
                mkldnn_grad_weight,
                stride.vec(),
                dilation.vec(),
                padding.vec(),
                padding.vec(),
                groups);
          }

          return std::make_tuple(
              mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_weight),
                                                      optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                                      grad_output.options().device_opt())),
              mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_bias),
                                                      optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                                      grad_output.options().device_opt())));
            */
    }

    pub fn mkldnn_convolution_backward(
            input:         &Tensor,
            grad_output_t: &Tensor,
            weight:        &Tensor,
            padding:       &[i32],
            stride:        &[i32],
            dilation:      &[i32],
            groups:        i64,
            output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
        
        todo!();
            /*
                Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous();

          Tensor grad_input, grad_weight, grad_bias;
          if (output_mask[0]) {
            grad_input = at::mkldnn_convolution_backward_input(
              input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
          }
          if (output_mask[1] || output_mask[2]) {
            std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
              weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
          }

          return std::make_tuple(grad_input, grad_weight, grad_bias);
            */
    }
}

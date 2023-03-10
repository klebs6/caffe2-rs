crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/NaiveDilatedConvolution.cpp]

/// hyper-volume to column, CPU
pub fn hvol2col<Dtype, const dim: i64>(
        data_hvol:     *const Dtype,
        channels:      i32,
        input_size:    &[i32],
        output_size:   &[i32],
        kernel_size:   &[i32],
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32],
        data_col:      *mut Dtype)  {

    todo!();
        /*
            if (dim == 3) {
        vol2col<Dtype>(
            data_hvol,
            channels,
            input_size[0],
            input_size[1],
            input_size[2],
            output_size[0],
            output_size[1],
            output_size[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            pad_size[0],
            pad_size[1],
            pad_size[2],
            stride_size[0],
            stride_size[1],
            stride_size[2],
            dilation_size[0],
            dilation_size[1],
            dilation_size[2],
            data_col);
      }
      if (dim == 2) {
        im2col<Dtype>(
            data_hvol,
            channels,
            input_size[0],
            input_size[1],
            output_size[0],
            output_size[1],
            kernel_size[0],
            kernel_size[1],
            pad_size[0],
            pad_size[1],
            stride_size[0],
            stride_size[1],
            dilation_size[0],
            dilation_size[1],
            data_col);
      }
        */
}

/// column to hyper-volume, CPU
pub fn col2hvol<Dtype, const dim: i64>(
        data_col:      *const Dtype,
        channels:      i32,
        input_size:    &[i32],
        output_size:   &[i32],
        kernel_size:   &[i32],
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32],
        data_hvol:     *mut Dtype)  {

    todo!();
        /*
            if (dim == 3) {
        col2vol<Dtype>(
            data_col,
            channels,
            input_size[0],
            input_size[1],
            input_size[2],
            output_size[0],
            output_size[1],
            output_size[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            pad_size[0],
            pad_size[1],
            pad_size[2],
            stride_size[0],
            stride_size[1],
            stride_size[2],
            dilation_size[0],
            dilation_size[1],
            dilation_size[2],
            data_hvol);
      }
      if (dim == 2) {
        col2im<Dtype>(
            data_col,
            channels,
            input_size[0],
            input_size[1],
            output_size[0],
            output_size[1],
            kernel_size[0],
            kernel_size[1],
            pad_size[0],
            pad_size[1],
            stride_size[0],
            stride_size[1],
            dilation_size[0],
            dilation_size[1],
            data_hvol);
      }
        */
}

/**
  | check tensor data locations
  |
  */
pub fn slow_conv_dilated_location_check(
        input:       &Tensor,
        weight:      &Tensor,
        bias:        &Tensor,
        grad_output: &Tensor)  {
    
    todo!();
        /*
            // checking data locations of user-provided tensor arguments
      checkBackend("slow_conv_dilated_location_check", {input, weight}, Backend::CPU);
      if (bias.defined()) {
        checkBackend("slow_conv_dilated_location_check", {bias}, Backend::CPU);
      }
      if (grad_output.defined()) {
        checkBackend("slow_conv_dilated_location_check", {grad_output}, Backend::CPU);
      }
      // we are not checking the data locations of other tensor
      // arguments such as output, grad_input, etc because of these are
      // allocated based on input options and hence these tensors always
      // have the same data location as of input tensor.
        */
}

/**
  | slow_conv_dilated_all_cpu_template
  | 
  | Main worker. Computes tensors output,
  | grad_input, grad_weight, and/or grad_bias
  | if defined, respectively.
  |
  */
pub fn slow_conv_dilated_all_cpu_template<const dim: i64>(
        output:        &mut Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        bias:          &Tensor,
        grad_output:   &Tensor,
        grad_input:    &mut Tensor,
        grad_weight:   &mut Tensor,
        grad_bias:     &mut Tensor,
        kernel_size:   &[i32],
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32])  {

    todo!();
        /*
            slow_conv_dilated_location_check(input, weight, bias, grad_output);
      auto options = input.options();
      // The rear part of input tensor sizes:
      auto input_size = input.sizes().slice(2);
      // The rear part of output tensor sizes:
      auto output_size = internal::get_output_size<dim>(
          input, kernel_size, stride_size, pad_size, dilation_size);
      i64 batchSize = input.size(0);
      i64 nInputPlane = weight.size(1);
      i64 nOutputPlane = weight.size(0);
      // Temporary buffer:
      Tensor columns = empty({0}, options);
      if (output.defined() || grad_weight.defined() || grad_input.defined()) {
        const i64 m = multiply_integers(kernel_size);
        const i64 n = multiply_integers(output_size);
        columns.resize_({nInputPlane * m, n});
      }
      // Initialize
      if (grad_weight.defined()) {
        grad_weight.zero_();
      }
      if (grad_bias.defined()) {
        grad_bias.zero_();
      }
      if (output.defined() && !bias.defined()) {
        output.zero_();
      }
      // Helpers
      Tensor grad_output_n;
      vector<i64> dims(dim);
      iota(dims.begin(), dims.end(), 1);

        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(), "slow_conv_dilated<>", [&] {
        // For each elt in batch, do:
        for (const auto elt : irange(batchSize)) {
          // Matrix multiply per output:
          Tensor input_n = input.select(0, elt);

          // Output
          if (output.defined()) {
            Tensor output_n = output.select(0, elt);
            if (bias.defined()) {
              /*
                Compute:

                  output_n = bias * ones^T

                where

                  bias is viewed as bias.view(nOutputPlane, 1)

                  ones is viewed as ones.view(outputHeight * outputWidth, 1)

                  output_n is viewed as output_n.view(nOutputPlane, outputHeight
              * outputWidth)

              gemm assumes column-major matrices:

                output_n^T = ones * bias^T
                C = alpha * op(A) * op(B)
                op(A) = 't', op(B) = 'n', alpha=1, beta=0
              */
              // The following for-loop is equivalent to the above
              // gemm setup but avoids allocation of ones tensor:
              for (const auto n : irange(nOutputPlane)) {
                output_n.select(0, n).fill_(bias[n]);
              }
            }
            // Extract columns:
            hvol2col<Scalar, dim>(
                input_n.data_ptr<Scalar>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                columns.data_ptr<Scalar>());
            /*
              Compute:

                output_n = weight * columns + output_n

              where

                weight is viewed as weight.view(nOutputPlane, nInputPlane * kD *
              kH * kW)

                columns size is (nInputPlane * kH * kW) x (outputHeight *
              outputWidth)

                output_n is viewed as output_n.view(nOutputPlane, outputHeight *
              outputWidth)

              gemm assumes column-major matrices:

                output_n^T = columns^T * weight^T + output_n^T
                C = alpha * op(A) * op(B) + beta * C
                op(A) = 'n', op(B) = 'n', alpha=1, beta=1
            */
            cpublas::gemm(
                /*transa=*/cpublas::NoTranspose,
                /*transb=*/cpublas::NoTranspose,
                /*     m=*/columns.size(1),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(0),
                /* alpha=*/1,
                /*     A=*/columns.data_ptr<Scalar>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.data_ptr<Scalar>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/1,
                /*     C=*/output_n.data_ptr<Scalar>(),
                /*   ldc=*/columns.size(1));

          } else {
            // All gradients
            grad_output_n = grad_output.select(0, elt);
          }

          // Gradient of input:
          if (grad_input.defined()) {
            /*
              Compute:

                columns = weight^T * grad_output_n

              where

                weight is viewed as weight.view(nOutputPlane, nInputPlane * kH *
              kW)

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

                columns size is (nInputPlane * kH * kW) x (outputHeight *
              outputWidth)

              gemm assumes column-major matrices:

                columns^T = grad_output_n^T * weight
                C = alpha * op(A) * op(B) + beta * C
                op(A) = 'n', op(B) = 't', alpha=1, beta=0
             */
            cpublas::gemm(
                /*transa=*/cpublas::NoTranspose,
                /*transb=*/cpublas::Transpose,
                /*     m=*/columns.size(1),
                /*     n=*/columns.size(0),
                /*     k=*/nOutputPlane,
                /* alpha=*/1,
                /*     A=*/grad_output_n.data_ptr<Scalar>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.data_ptr<Scalar>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/0,
                /*     C=*/columns.data_ptr<Scalar>(),
                /*   ldc=*/columns.size(1));
            // Unpack columns back into input:
            Tensor grad_input_n = grad_input.select(0, elt);

            col2hvol<Scalar, dim>(
                columns.data_ptr<Scalar>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                grad_input_n.data_ptr<Scalar>());
          }

          // Gradient of weight:
          if (grad_weight.defined()) {
            // Extract columns:
            hvol2col<Scalar, dim>(
                input_n.data_ptr<Scalar>(),
                nInputPlane,
                input_size,
                output_size,
                kernel_size,
                stride_size,
                pad_size,
                dilation_size,
                columns.data_ptr<Scalar>());
            Scalar scale = 1; // TODO: expose as argument?
            /*
              Compute:

                grad_weight = scale * grad_output_n * columns^T + grad_weight

              where

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

                columns size is (nInputPlane * kD * kH * kW) x (outputHeight *
              outputWidth)

                grad_weight is viewed as grad_weight.view(nOutputPlane,
              nInputPlane * kH * kW)

              gemm assumes column-major matrices:

                grad_weight^T = scale * columns * grad_output_n^T +
              grad_weight^T C = alpha * op(A) * op(B) + beta * C op(A) = 't',
              op(B) = 'n', alpha=scale, beta=1
            */
            cpublas::gemm(
                /*transa=*/cpublas::Transpose,
                /*transb=*/cpublas::NoTranspose,
                /*     m=*/columns.size(0),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(1),
                /* alpha=*/scale,
                /*     A=*/columns.data_ptr<Scalar>(),
                /*   lda=*/columns.size(1),
                /*     B=*/grad_output_n.data_ptr<Scalar>(),
                /*   ldb=*/columns.size(1),
                /*  beta=*/1,
                /*     C=*/grad_weight.data_ptr<Scalar>(),
                /*   ldc=*/columns.size(0));
          }

          // Gradient of bias:
          if (grad_bias.defined()) {
            /*
              Compute:
                grad_bias = scale * grad_output_n * ones + grad_bias

              where

                grad_bias is viewed as grad_bias.view(nOutputPlane, 1)

                ones is viewed as ones.view(outputHeight * outputWidth, 1)

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

              gemm assumes column-major matrices:

                grad_bias^T = scale * grad_output_n * ones + grad_bias^T
                y = alpha * op(A) * x + beta * y
                op(A) = 't', alpha=scale, beta=1
             */
            // The following expression is equivalent to the above
            // gemm setup but avoids allocation of ones tensor:
            grad_bias += grad_output_n.sum(dims);
            /*
              TODO: when scale != 1 is introduced then use:
                grad_bias += scale * grad_output_n.sum(dims);
             */
          }
        }
      });
        */
}

pub fn slow_conv_dilated2d_cpu(
        input:         &Tensor,
        weight:        &Tensor,
        kernel_size:   &[i32],
        bias_opt:      &Option<Tensor>,
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor undefined;
      internal::slow_conv_dilated_shape_check<2>(
          input,
          weight,
          bias,
          undefined,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      auto is_batch = input.dim() == 4;
      auto options = input.options();
      // calculate output tensor size
      auto output_size = internal::get_output_size<2>(
          input, weight, kernel_size, stride_size, pad_size, dilation_size);
      // template function assumes batched tensors.  unsqueeze(0) will
      // insert batch dimension without affecting the original tensor.
      const Tensor input_ =
          (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
      const Tensor weight_ = weight.contiguous();
      const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
      Tensor output = empty(output_size, options);
      Tensor output_ = (is_batch ? output : output.unsqueeze(0));

      slow_conv_dilated_all_cpu_template<2>(
          output_,
          input_,
          weight_,
          bias_,
          undefined,
          undefined,
          undefined,
          undefined,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      return output;
        */
}

pub fn slow_conv_dilated2d_backward_cpu(
        grad_output:   &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        kernel_size:   &[i32],
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32],
        output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor undefined;
      internal::slow_conv_dilated_shape_check<2>(
          input,
          weight,
          undefined,
          grad_output,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      auto is_batch = input.dim() == 4;
      auto options = grad_output.options();
      // template function assumes batched tensors.  unsqueeze(0) will
      // insert batch dimension without affecting the original tensor.
      const Tensor grad_output_ =
          (is_batch ? grad_output.contiguous()
                    : grad_output.contiguous().unsqueeze(0));
      const Tensor input_ =
          (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
      const Tensor weight_ = weight.contiguous();
      // compute only gradients for which the corresponding output_mask is true:
      Tensor grad_input =
          (output_mask[0] ? empty(input.sizes(), options) : undefined);
      Tensor grad_weight =
          (output_mask[1] ? empty(weight.sizes(), options) : undefined);
      Tensor grad_bias =
          (output_mask[2] ? empty(weight.size(0), options) : undefined);
      Tensor grad_input_ =
          (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                          : undefined);
      slow_conv_dilated_all_cpu_template<2>(
          undefined,
          input_,
          weight_,
          undefined,
          grad_output_,
          grad_input,
          grad_weight,
          grad_bias,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      return tie(grad_input, grad_weight, grad_bias);
        */
}

pub fn slow_conv_dilated3d_cpu(
        input:         &Tensor,
        weight:        &Tensor,
        kernel_size:   &[i32],
        bias_opt:      &Option<Tensor>,
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor undefined;
      internal::slow_conv_dilated_shape_check<3>(
          input,
          weight,
          bias,
          undefined,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      auto is_batch = input.dim() == 5;
      auto options = input.options();
      // calculate output tensor size
      auto output_size = internal::get_output_size<3>(
          input, weight, kernel_size, stride_size, pad_size, dilation_size);
      // template function assumes batched tensors.  unsqueeze(0) will
      // insert batch dimension without affecting the original tensor.
      const Tensor input_ =
          (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
      const Tensor weight_ = weight.contiguous();
      const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
      Tensor output = empty(output_size, options);
      Tensor output_ = (is_batch ? output : output.unsqueeze(0));

      slow_conv_dilated_all_cpu_template<3>(
          output,
          input_,
          weight_,
          bias_,
          undefined,
          undefined,
          undefined,
          undefined,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      return output;
        */
}

pub fn slow_conv_dilated3d_backward_cpu(
        grad_output:   &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        kernel_size:   &[i32],
        stride_size:   &[i32],
        pad_size:      &[i32],
        dilation_size: &[i32],
        output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor undefined;
      internal::slow_conv_dilated_shape_check<3>(
          input,
          weight,
          undefined,
          grad_output,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      auto is_batch = input.dim() == 5;
      auto options = grad_output.options();
      // template function assumes batched tensors.  unsqueeze(0) will
      // insert batch dimension without affecting the original tensor.
      const Tensor grad_output_ =
          (is_batch ? grad_output.contiguous()
                    : grad_output.contiguous().unsqueeze(0));
      const Tensor input_ =
          (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
      const Tensor weight_ = weight.contiguous();
      // compute only gradients for which the corresponding output_mask is true:
      Tensor grad_input =
          (output_mask[0] ? empty(input.sizes(), options) : undefined);
      Tensor grad_weight =
          (output_mask[1] ? empty(weight.sizes(), options) : undefined);
      Tensor grad_bias =
          (output_mask[2] ? empty(weight.size(0), options) : undefined);
      Tensor grad_input_ =
          (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                          : undefined);
      slow_conv_dilated_all_cpu_template<3>(
          undefined,
          input_,
          weight_,
          undefined,
          grad_output_,
          grad_input,
          grad_weight,
          grad_bias,
          kernel_size,
          stride_size,
          pad_size,
          dilation_size);
      return tie(grad_input, grad_weight, grad_bias);
        */
}

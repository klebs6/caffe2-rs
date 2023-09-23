crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp]

pub fn gemv<Scalar>(
        trans: u8,
        m:     i64,
        n:     i64,
        alpha: Scalar,
        a:     *mut Scalar,
        lda:   i64,
        x:     *mut Scalar,
        incx:  i64,
        beta:  Scalar,
        y:     *mut Scalar,
        incy:  i64)  {
    
    todo!();
        /*
        
        */
}

#[inline] pub fn slow_conv_transpose2d_shape_check(
        input:                 &Tensor,
        grad_output:           &Tensor,
        weight:                &Tensor,
        bias:                  &Tensor,
        kernel_height:         i32,
        kernel_width:          i32,
        stride_height:         i32,
        stride_width:          i32,
        pad_height:            i32,
        pad_width:             i32,
        output_padding_height: i32,
        output_padding_width:  i32,
        dilation_height:       i32,
        dilation_width:        i32,
        weight_nullable:       bool)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_width > 0 && kernel_height > 0,
          "kernel size should be greater than zero, but got kernel_height: ",
          kernel_height,
          " kernel_width: ",
          kernel_width);
      TORCH_CHECK(
          stride_width > 0 && stride_height > 0,
          "stride should be greater than zero, but got stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width);
      TORCH_CHECK(
          dilation_width > 0 && dilation_height > 0,
          "dilation should be greater than zero, but got dilation_height: ",
          dilation_height,
          ", dilation_width: ",
          dilation_width);
      TORCH_CHECK(
          (output_padding_width < stride_width ||
           output_padding_width < dilation_width) &&
              (output_padding_height < stride_height ||
               output_padding_height < dilation_height),
          "output padding must be smaller than either stride or dilation, but got output_padding_height: ",
          output_padding_height,
          " output_padding_width: ",
          output_padding_width,
          " stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width,
          " dilation_height: ",
          dilation_height,
          " dilation_width: ",
          dilation_width);

      if (weight.defined()) {
        TORCH_CHECK(
            weight.numel() != 0 && (weight.dim() == 2 || weight.dim() == 4),
            "non-empty 2D or 4D weight tensor expected, but got: ",
            weight.sizes());
        if (bias.defined()) {
          check_dim_size(bias, 1, 0, weight.size(1));
        }
      } else if (!weight_nullable) {
        AT_ERROR("weight tensor is expected to be non-nullable");
      }

      int ndim = input.dim();
      int dimf = 0;
      int dimh = 1;
      int dimw = 2;

      if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
      }

      TORCH_CHECK(
          input.numel() != 0 && (ndim == 3 || ndim == 4),
          "non-empty 3D or 4D input tensor expected but got a tensor with size ",
          input.sizes());

      i64 input_height = input.size(dimh);
      i64 input_width = input.size(dimw);
      i64 output_height = (input_height - 1) * stride_height - 2 * pad_height +
          (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
      i64 output_width = (input_width - 1) * stride_width - 2 * pad_width +
          (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

      if (output_width < 1 || output_height < 1) {
        AT_ERROR(
            "Given input size per channel: (",
            input_height,
            " x ",
            input_width,
            "). "
            "Calculated output size per channel: (",
            output_height,
            " x ",
            output_width,
            "). Output size is too small");
      }

      if (weight.defined()) {
        i64 n_input_plane = weight.size(0);
        check_dim_size(input, ndim, dimf, n_input_plane);
      }

      if (grad_output.defined()) {
        if (weight.defined()) {
          i64 n_output_plane = weight.size(1);
          check_dim_size(grad_output, ndim, dimf, n_output_plane);
        } else if (bias.defined()) {
          i64 n_output_plane = bias.size(0);
          check_dim_size(grad_output, ndim, dimf, n_output_plane);
        }
        check_dim_size(grad_output, ndim, dimh, output_height);
        check_dim_size(grad_output, ndim, dimw, output_width);
      }
        */
}

pub fn slow_conv_transpose2d_out_cpu_template(
        output:         &mut Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        bias:           &Tensor,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        columns:        &mut Tensor,
        ones:           &mut Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 2,
          "It is expected kernel_size equals to 2, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 2,
          "It is expected dilation equals to 2, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 2,
          "It is expected padding equals to 2, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 2,
          "It is expected stride equals to 2, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 2,
          "It is expected stride equals to 2, but got size ",
          output_padding.size());

      Tensor columns = columns_;
      Tensor ones = ones_;

      i64 kernel_height = kernel_size[0];
      i64 kernel_width = kernel_size[1];
      i64 dilation_height = dilation[0];
      i64 dilation_width = dilation[1];
      i64 pad_height = padding[0];
      i64 pad_width = padding[1];
      i64 stride_height = stride[0];
      i64 stride_width = stride[1];
      i64 output_padding_height = output_padding[0];
      i64 output_padding_width = output_padding[1];

      slow_conv_transpose2d_shape_check(
          input_,
          Tensor(),
          weight_,
          bias_,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          output_padding_height,
          output_padding_width,
          dilation_height,
          dilation_width,
          false);

      int n_input_plane = weight_.size(0);
      int n_output_plane = weight_.size(1);

      Tensor input = input_.contiguous();
      Tensor weight = weight_.contiguous();

      TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");

      Tensor bias = Tensor();

      if (bias_.defined()) {
        bias = bias_.contiguous();
        TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
      }

      bool is_batch = false;
      if (input.dim() == 3) {
        // Force batch
        is_batch = true;
        input.resize_({1, input.size(0), input.size(1), input.size(2)});
      }

      i64 input_height = input.size(2);
      i64 input_width = input.size(3);
      i64 output_height = (input_height - 1) * stride_height - 2 * pad_height +
          (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
      i64 output_width = (input_width - 1) * stride_width - 2 * pad_width +
          (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

      // Batch size + input planes
      i64 batch_size = input.size(0);

      // Resize output
      output.resize_({batch_size, n_output_plane, output_height, output_width});

      // Resize temporary columns
      columns.resize_({n_output_plane * kernel_width * kernel_height,
                       input_height * input_width});
      columns.zero_();

      // Define a buffer of ones, for bias accumulation
      // Note: this buffer can be shared with other modules, it only ever gets
      // increased, and always contains ones.
      if (ones.dim() != 2 ||
          ones.size(0) * ones.size(1) < output_height * output_width) {
        // Resize plane and fill with ones...
        ones.resize_({output_height, output_width});
        ones.fill_(1);
      }

      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long,
          input.scalar_type(), "slow_conv_transpose2d_out_cpu", [&] {
            // For each elt in batch, do:
            for (const auto elt : irange(batch_size)) {
              // Helpers
              Tensor input_n;
              Tensor output_n;

              // Matrix mulitply per output:
              input_n = input.select(0, elt);
              output_n = output.select(0, elt);

              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              i64 m = weight.size(1) * weight.size(2) * weight.size(3);
              i64 n = columns.size(1);
              i64 k = weight.size(0);

              // Do GEMM (note: this is a bit confusing because gemm assumes
              // column-major matrices)
              cpublas::gemm(
                  cpublas::NoTranspose,
                  cpublas::Transpose,
                  n,
                  m,
                  k,
                  1,
                  input_n.data_ptr<Scalar>(),
                  n,
                  weight.data_ptr<Scalar>(),
                  m,
                  0,
                  columns.data_ptr<Scalar>(),
                  n);

              // Unpack columns back into input:
              col2im<Scalar>(
                  columns.data_ptr<Scalar>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  input_height,
                  input_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  output_n.data_ptr<Scalar>());

              // Do Bias after:
              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              i64 m_ = n_output_plane;
              i64 n_ = output_height * output_width;
              i64 k_ = 1;

              // Do GEMM (note: this is a bit confusing because gemm assumes
              // column-major matrices)
              if (bias_.defined()) {
                cpublas::gemm(
                    cpublas::Transpose,
                    cpublas::NoTranspose,
                    n_,
                    m_,
                    k_,
                    1,
                    ones.data_ptr<Scalar>(),
                    k_,
                    bias.data_ptr<Scalar>(),
                    k_,
                    1,
                    output_n.data_ptr<Scalar>(),
                    n_);
              }
            }

            // Resize output
            if (is_batch) {
              output.resize_({n_output_plane, output_height, output_width});
              input.resize_({n_input_plane, input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose2d_backward_out_cpu_template(
        input:          &Tensor,
        grad_output:    &Tensor,
        grad_input:     &mut Tensor,
        weight:         &Tensor,
        grad_columns:   &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 2,
          "It is expected kernel_size equals to 2, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 2,
          "It is expected dilation equals to 2, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 2,
          "It is expected padding equals to 2, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 2,
          "It is expected stride equals to 2, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 2,
          "It is expected stride equals to 2, but got size ",
          output_padding.size());

      i64 kernel_height = kernel_size[0];
      i64 kernel_width = kernel_size[1];
      i64 dilation_height = dilation[0];
      i64 dilation_width = dilation[1];
      i64 pad_height = padding[0];
      i64 pad_width = padding[1];
      i64 stride_height = stride[0];
      i64 stride_width = stride[1];
      i64 output_padding_height = output_padding[0];
      i64 output_padding_width = output_padding[1];

      i64 n_input_plane = weight_.size(0);
      i64 n_output_plane = weight_.size(1);

      Tensor grad_columns = grad_columns_;

      slow_conv_transpose2d_shape_check(
          input_,
          grad_output_,
          weight_,
          Tensor(),
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          output_padding_height,
          output_padding_width,
          dilation_height,
          dilation_width,
          false);

      Tensor input = input_.contiguous();
      Tensor grad_output = grad_output_.contiguous();
      Tensor weight = weight_.contiguous();

      TORCH_CHECK(
          grad_columns.is_contiguous(), "grad_columns needs to be contiguous");

      bool is_batch = false;
      if (input.dim() == 3) {
        // Force batch
        is_batch = true;
        input.resize_({1, input.size(0), input.size(1), input.size(2)});
        grad_output.resize_(
            {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
      }

      i64 input_width = input.size(3);
      i64 input_height = input.size(2);
      i64 output_height = (input_height - 1) * stride_height - 2 * pad_height +
          (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
      i64 output_width = (input_width - 1) * stride_width - 2 * pad_width +
          (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

      // Batch size + input planes
      i64 batch_size = input.size(0);

      // Resize output
      grad_input.resize_({batch_size, n_input_plane, input_height, input_width});
      grad_input.zero_();

      // Resize temporary columns
      grad_columns.resize_({n_output_plane * kernel_width * kernel_height,
                            input_height * input_width});

      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(), "slow_conv_transpose2d_backward_out_cpu", [&] {
            // Helpers
            Tensor grad_input_n = Tensor();
            Tensor grad_output_n = Tensor();

            // For each elt in batch, do:
            for (const auto elt : irange(batch_size)) {
              // Matrix mulitply per sample:
              grad_input_n = grad_input.select(0, elt);
              grad_output_n = grad_output.select(0, elt);

              if (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
                  stride_width != 1 || pad_height != 0 || pad_width != 0 ||
                  dilation_height != 1 || dilation_width != 1) {
                // Extract columns:
                im2col<Scalar>(
                      grad_output_n.data_ptr<Scalar>(),
                      n_output_plane,
                      output_height,
                      output_width,
                      input_height,
                      input_width,
                      kernel_height,
                      kernel_width,
                      pad_height,
                      pad_width,
                      stride_height,
                      stride_width,
                      dilation_height,
                      dilation_width,
                      grad_columns.data_ptr<Scalar>());
              }

              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              i64 m = weight.size(0);
              i64 n = grad_columns.size(1);
              i64 k = weight.size(1) * weight.size(2) * weight.size(3);

              // Do GEMM (note: this is a bit confusing because gemm assumes
              // column-major matrices)
              auto gemm_in_ptr =
                  (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
                   stride_width != 1 || pad_height != 0 || pad_width != 0 ||
                   dilation_height != 1 || dilation_width != 1)
                  ? grad_columns.data_ptr<Scalar>()
                  : grad_output_n.data_ptr<Scalar>();
              cpublas::gemm(
                  cpublas::NoTranspose,
                  cpublas::NoTranspose,
                  n,
                  m,
                  k,
                  1,
                  gemm_in_ptr,
                  n,
                  weight.data_ptr<Scalar>(),
                  k,
                  0,
                  grad_input_n.data_ptr<Scalar>(),
                  n);
            }

            // Resize output
            if (is_batch) {
              grad_output.resize_({n_output_plane, output_height, output_width});
              input.resize_({n_input_plane, input_height, input_width});
              grad_input.resize_({n_input_plane, input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose2d_acc_grad_parameters_cpu(
        input:          &Tensor,
        grad_output:    &Tensor,
        grad_weight:    &mut Tensor,
        grad_bias:      &mut Tensor,
        columns:        &Tensor,
        ones:           &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        scale:          i32)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 2,
          "It is expected kernel_size equals to 2, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 2,
          "It is expected dilation equals to 2, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 2,
          "It is expected padding equals to 2, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 2,
          "It is expected stride equals to 2, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 2,
          "It is expected stride equals to 2, but got size ",
          output_padding.size());

      i64 kernel_height = kernel_size[0];
      i64 kernel_width = kernel_size[1];
      i64 dilation_height = dilation[0];
      i64 dilation_width = dilation[1];
      i64 pad_height = padding[0];
      i64 pad_width = padding[1];
      i64 stride_height = stride[0];
      i64 stride_width = stride[1];
      i64 output_padding_height = output_padding[0];
      i64 output_padding_width = output_padding[1];

      Tensor columns = columns_;
      Tensor ones = ones_;

      slow_conv_transpose2d_shape_check(
          input_,
          grad_output_,
          grad_weight,
          grad_bias,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          output_padding_height,
          output_padding_width,
          dilation_height,
          dilation_width,
          true);

      i64 n_output_plane;
      if (grad_weight.defined()) {
        n_output_plane = grad_weight.size(1);
      } else if (grad_bias.defined()) {
        n_output_plane = grad_bias.size(0);
      } else {
        return;
      }

      Tensor input = input_.contiguous();
      Tensor grad_output = grad_output_.contiguous();

      if (grad_weight.defined()) {
        TORCH_CHECK(
            grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
      }
      TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");
      if (grad_bias.defined()) {
        TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
        TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
      }

      bool is_batch = false;
      if (input.dim() == 3) {
        // Force batch
        is_batch = true;
        input.resize_({1, input.size(0), input.size(1), input.size(2)});
        grad_output.resize_(
            {1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
      }

      i64 input_width = input.size(3);
      i64 input_height = input.size(2);
      i64 output_height = (input_height - 1) * stride_height - 2 * pad_height +
          (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
      i64 output_width = (input_width - 1) * stride_width - 2 * pad_width +
          (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

      // Batch size + input planes
      i64 batch_size = input.size(0);

      // Define a buffer of ones, for bias accumulation
      if (ones.dim() != 2 ||
          ones.size(0) * ones.size(1) < output_height * output_width) {
        // Resize plane and fill with ones...
        ones.resize_({output_height, output_width});
        ones.fill_(1);
      }

      // Resize temporary columns
      columns.resize_({n_output_plane * kernel_width * kernel_height,
                       input_height * input_width});

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "slow_conv_transpose2d_acc_grad_parameters_cpu", [&] {
            // Helpers
            Tensor input_n = Tensor();
            Tensor grad_output_n = Tensor();

            Scalar scale = static_cast<Scalar>(scale_);

            // For each elt in batch, do:
            for (const auto elt : irange(batch_size)) {
              // Matrix mulitply per output:
              grad_output_n = grad_output.select(0, elt);

              // Do Weight:
              if (grad_weight.defined()) {
                // Matrix mulitply per output:
                input_n = input.select(0, elt);

                if (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
                    stride_width != 1 || pad_height != 0 || pad_width != 0 ||
                    dilation_height != 1 || dilation_width != 1) {
                  // Extract columns:
                  im2col<Scalar>(
                      grad_output_n.data_ptr<Scalar>(),
                      n_output_plane,
                      output_height,
                      output_width,
                      input_height,
                      input_width,
                      kernel_height,
                      kernel_width,
                      pad_height,
                      pad_width,
                      stride_height,
                      stride_width,
                      dilation_height,
                      dilation_width,
                      columns.data_ptr<Scalar>());
                }

                // M,N,K are dims of matrix A and B
                // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
                i64 n = columns.size(0); // n_output_plane * kh * kw
                i64 m = input_n.size(0); // n_input_plane
                i64 k = columns.size(1); // input_height * input_width

                // Do GEMM (note: this is a bit confusing because gemm assumes
                // column-major matrices)
                auto gemm_in_ptr =
                    (kernel_height != 1 || kernel_width != 1 ||
                     stride_height != 1 || stride_width != 1 || pad_height != 0 ||
                     pad_width != 0 || dilation_height != 1 || dilation_width != 1)
                    ? columns.data_ptr<Scalar>()
                    : grad_output_n.data_ptr<Scalar>();
                cpublas::gemm(
                    cpublas::Transpose,
                    cpublas::NoTranspose,
                    n,
                    m,
                    k,
                    scale,
                    gemm_in_ptr,
                    k,
                    input_n.data_ptr<Scalar>(),
                    k,
                    1,
                    grad_weight.data_ptr<Scalar>(),
                    n);
              }

              // Do Bias:
              if (grad_bias.defined()) {
                // M,N,K are dims of matrix A and B
                // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
                i64 m_ = n_output_plane;
                i64 k_ = output_height * output_width;

                // Do GEMV (note: this is a bit confusing because gemv assumes
                // column-major matrices)
                native::gemv<Scalar>(
                    't',
                    k_,
                    m_,
                    scale,
                    grad_output_n.data_ptr<Scalar>(),
                    k_,
                    ones.data_ptr<Scalar>(),
                    1,
                    1,
                    grad_bias.data_ptr<Scalar>(),
                    1);
              }
            }

            // Resize
            if (is_batch) {
              grad_output.resize_({n_output_plane, output_height, output_width});
              input.resize_({input.size(1), input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose2d_out_cpu<'a>(
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        output:         &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor columns = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor ones = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      slow_conv_transpose2d_out_cpu_template(
          output,
          input,
          weight,
          kernel_size,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          columns,
          ones);

      return output;
        */
}

pub fn slow_conv_transpose2d_cpu(
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor output = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor columns = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor ones = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      slow_conv_transpose2d_out_cpu_template(
          output,
          input,
          weight,
          kernel_size,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          columns,
          ones);

      return output;
        */
}

pub fn slow_conv_transpose2d_backward_out_cpu<'a>(
        grad_output:    &Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        columns:        &Tensor,
        ones:           &Tensor,
        grad_input:     &mut Tensor,
        grad_weight:    &mut Tensor,
        grad_bias:      &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            if (grad_input.defined()) {
        slow_conv_transpose2d_backward_out_cpu_template(
            input,
            grad_output,
            grad_input,
            weight,
            columns,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation);
      }

      if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
      }

      if (grad_bias.defined()) {
        grad_bias.resize_({weight.size(1)});
        grad_bias.zero_();
      }

      if (grad_weight.defined() || grad_bias.defined()) {
        slow_conv_transpose2d_acc_grad_parameters_cpu(
            input,
            grad_output,
            grad_weight,
            grad_bias,
            columns,
            ones,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            1);
      }

      return tuple<Tensor&, Tensor&, Tensor&>(
          grad_input, grad_weight, grad_bias);
        */
}

pub fn slow_conv_transpose2d_backward_cpu(
        grad_output:    &Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        columns:        &Tensor,
        ones:           &Tensor,
        output_mask:    [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor grad_input;
      Tensor grad_weight;
      Tensor grad_bias;

      if (output_mask[0]) {
        grad_input = empty({0}, grad_output.options());
      } else {
        grad_input = Tensor();
      }

      if (output_mask[1]) {
        grad_weight = empty({0}, grad_output.options());
      } else {
        grad_weight = Tensor();
      }

      if (output_mask[2]) {
        grad_bias = empty({0}, grad_output.options());
      } else {
        grad_bias = Tensor();
      }

      if (grad_input.defined()) {
        slow_conv_transpose2d_backward_out_cpu_template(
            input,
            grad_output,
            grad_input,
            weight,
            columns,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation);
      }

      if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
      }

      if (grad_bias.defined()) {
        grad_bias.resize_({weight.size(1)});
        grad_bias.zero_();
      }

      if (grad_weight.defined() || grad_bias.defined()) {
        slow_conv_transpose2d_acc_grad_parameters_cpu(
            input,
            grad_output,
            grad_weight,
            grad_bias,
            columns,
            ones,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            1);
      }

      return tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
        */
}

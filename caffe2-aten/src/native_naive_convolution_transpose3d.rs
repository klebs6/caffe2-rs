crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/NaiveConvolutionTranspose3d.cpp]

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

#[inline] pub fn slow_conv_transpose3d_shape_check(
        input:                 &Tensor,
        grad_output:           &Tensor,
        weight:                &Tensor,
        bias:                  &Tensor,
        kernel_depth:          i32,
        kernel_width:          i32,
        kernel_height:         i32,
        stride_depth:          i32,
        stride_width:          i32,
        stride_height:         i32,
        padding_depth:         i32,
        padding_width:         i32,
        padding_height:        i32,
        dilation_depth:        i32,
        dilation_width:        i32,
        dilation_height:       i32,
        output_padding_depth:  i32,
        output_padding_width:  i32,
        output_padding_height: i32,
        weight_nullable:       i32)  {
    
    todo!();
        /*
            TORCH_CHECK(
          input.numel() != 0 && (input.dim() == 4 || input.dim() == 5),
          "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
          input.sizes());
      TORCH_CHECK(
          stride_depth > 0 && stride_width > 0 && stride_height > 0,
          "stride should be greater than zero, but got stride_depth: ",
          stride_depth,
          " stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width);
      TORCH_CHECK(
          dilation_depth > 0 && dilation_width > 0 && dilation_height > 0,
          "dilation should be greater than zero, but got dilation_depth: ",
          dilation_depth,
          ", dilation_height: ",
          dilation_height,
          ", dilation_width: ",
          dilation_width);
      TORCH_CHECK(
          (output_padding_depth < stride_depth ||
           output_padding_depth < dilation_depth) &&
              (output_padding_width < stride_width ||
               output_padding_width < dilation_width) &&
              (output_padding_height < stride_height ||
               output_padding_height < dilation_height),
          "output padding must be smaller than either stride or dilation,",
          " but got output_padding_depth: ",
          output_padding_depth,
          " output_padding_height: ",
          output_padding_height,
          " output_padding_width: ",
          output_padding_width,
          " stride_depth: ",
          stride_depth,
          " stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width,
          " dilation_depth: ",
          dilation_depth,
          " dilation_height: ",
          dilation_height,
          " dilation_width: ",
          dilation_width);

      // number of input & output planes and kernel size is indirectly defined by
      // the weight tensor
      if (weight.defined()) {
        /* TODO: TORCH_CHECK just have 2 args: condition and message */
        TORCH_CHECK(
            weight.numel() != 0 && weight.dim() == 5,
            "non-empty 5D (n_output_plane x n_input_plane x kernel_depth",
            " x kernel_height x kernel_width) tensor ",
            "expected for weight, but got: ",
            weight.sizes());
        if (bias.defined()) {
          check_dim_size(bias, 1, 0, weight.size(1));
        }
      } else if (!weight_nullable) {
        AT_ERROR("weight tensor is expected to be non-nullable");
      }

      int ndim = input.dim();
      int dimf = 0;
      int dimd = 1;
      int dimh = 2;
      int dimw = 3;

      if (ndim == 5) {
        dimf++;
        dimd++;
        dimh++;
        dimw++;
      }

      if (weight.defined()) {
        const i64 n_input_plane = weight.size(0);
        check_dim_size(input, ndim, dimf, n_input_plane);
      }

      const i64 input_width = input.size(dimw);
      const i64 input_height = input.size(dimh);
      const i64 input_depth = input.size(dimd);
      const i64 output_depth = (input_depth - 1) * stride_depth -
          2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
          output_padding_depth;
      const i64 output_height = (input_height - 1) * stride_height -
          2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
          output_padding_height;
      const i64 output_width = (input_width - 1) * stride_width -
          2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
          output_padding_width;

      if (output_depth < 1 || output_width < 1 || output_height < 1) {
        AT_ERROR(
            "Given input size per channel: (",
            input_depth,
            " x ",
            input_height,
            " x ",
            input_width,
            "). "
            "Calculated output size per channel: (",
            output_depth,
            " x ",
            output_height,
            " x ",
            output_width,
            "). Output size is too small");
      }

      if (grad_output.defined()) {
        if (weight.defined()) {
          const i64 n_output_plane = weight.size(1);
          check_dim_size(grad_output, ndim, dimf, n_output_plane);
        } else if (bias.defined()) {
          const i64 n_output_plane = bias.size(0);
          check_dim_size(grad_output, ndim, dimf, n_output_plane);
        }
        check_dim_size(grad_output, ndim, dimd, output_depth);
        check_dim_size(grad_output, ndim, dimh, output_height);
        check_dim_size(grad_output, ndim, dimw, output_width);
      }
        */
}

pub fn slow_conv_transpose3d_out_cpu_template(
        output:         &mut Tensor,

        // 4D or 5D (batch) tensor
        input:          &Tensor,

        // weight tensor (n_input_plane
        // x n_output_plane x kernel_depth
        // x kernel_height x kernel_width)
        //
        weight:         &Tensor,
        kernel_size:    &[i32],
        bias:           &Tensor,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        finput:         &mut Tensor,
        fgrad_input:    &mut Tensor)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 3,
          "It is expected kernel_size equals to 3, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 3,
          "It is expected dilation equals to 3, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 3,
          "It is expected padding equals to 3, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 3,
          "It is expected stride equals to 3, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 3,
          "It is expected stride equals to 3, but got size ",
          output_padding.size());

      i64 kernel_depth = kernel_size[0];
      i64 kernel_height = kernel_size[1];
      i64 kernel_width = kernel_size[2];
      i64 dilation_depth = dilation[0];
      i64 dilation_height = dilation[1];
      i64 dilation_width = dilation[2];
      i64 padding_depth = padding[0];
      i64 padding_height = padding[1];
      i64 padding_width = padding[2];
      i64 stride_depth = stride[0];
      i64 stride_height = stride[1];
      i64 stride_width = stride[2];
      i64 output_padding_depth = output_padding[0];
      i64 output_padding_height = output_padding[1];
      i64 output_padding_width = output_padding[2];

      // internal columns buffer
      Tensor& columns = finput;
      // internal ones buffer
      Tensor& ones = fgrad_input;

      slow_conv_transpose3d_shape_check(
          input_,
          Tensor(),
          weight_,
          bias_,
          kernel_depth,
          kernel_width,
          kernel_height,
          stride_depth,
          stride_width,
          stride_height,
          padding_depth,
          padding_width,
          padding_height,
          dilation_depth,
          dilation_width,
          dilation_height,
          output_padding_depth,
          output_padding_width,
          output_padding_height,
          0);

      Tensor input = input_.contiguous();
      Tensor weight = weight_.contiguous();
      Tensor bias = bias_.defined() ? bias_.contiguous() : bias_;

      const int n_input_plane = (int)weight.size(0);
      const int n_output_plane = (int)weight.size(1);

      bool is_batch = false;
      if (input.dim() == 4) {
        // Force batch
        is_batch = true;
        input.resize_(
            {1, input.size(0), input.size(1), input.size(2), input.size(3)});
      }

      const i64 input_width = input.size(4);
      const i64 input_height = input.size(3);
      const i64 input_depth = input.size(2);

      const i64 output_depth = (input_depth - 1) * stride_depth -
          2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
          output_padding_depth;
      const i64 output_height = (input_height - 1) * stride_height -
          2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
          output_padding_height;
      const i64 output_width = (input_width - 1) * stride_width -
          2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
          output_padding_width;

      // Batch size + input planes
      const i64 batch_size = input.size(0);

      // Resize output
      output.resize_(
          {batch_size, n_output_plane, output_depth, output_height, output_width});

      // Resize temporary columns
      columns.resize_({n_output_plane * kernel_width * kernel_height * kernel_depth,
                       input_depth * input_height * input_width});
      columns.zero_();

      // Define a buffer of ones, for bias accumulation
      // Note: this buffer can be shared with other modules, it only ever gets
      // increased, and always contains ones.
      if (ones.dim() != 3 ||
          ones.size(0) * ones.size(1) * ones.size(2) <
              output_depth * output_height * output_width) {
        // Resize plane and fill with ones...
        ones.resize_({output_depth, output_height, output_width});
        ones.fill_(1);
      }

      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long,
          input.scalar_type(), "slow_conv_transpose3d_out_cpu", [&] {
            // Helpers
            Tensor input_n;
            Tensor output_n;

            i64 elt;
            // For each elt in batch, do:
            for (elt = 0; elt < batch_size; ++elt) {
              // Matrix mulitply per output:
              input_n = input.select(0, elt);
              output_n = output.select(0, elt);

              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              const i64 m =
                  weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
              const i64 n = columns.size(1);
              const i64 k = weight.size(0);

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
              native::col2vol<Scalar>(
                  columns.data_ptr<Scalar>(),
                  n_output_plane,
                  output_depth,
                  output_height,
                  output_width,
                  input_depth,
                  input_height,
                  input_width,
                  kernel_depth,
                  kernel_height,
                  kernel_width,
                  padding_depth,
                  padding_height,
                  padding_width,
                  stride_depth,
                  stride_height,
                  stride_width,
                  dilation_depth,
                  dilation_height,
                  dilation_width,
                  output_n.data_ptr<Scalar>());

              // Do Bias after:
              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              const i64 m_ = n_output_plane;
              const i64 n_ = output_depth * output_height * output_width;
              const i64 k_ = 1;

              // Do GEMM (note: this is a bit confusing because gemm assumes
              // column-major matrices)
              if (bias.defined()) {
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
              output.resize_(
                  {n_output_plane, output_depth, output_height, output_width});
              input.resize_(
                  {n_input_plane, input_depth, input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose3d_backward_out_cpu_template(
        input:          &Tensor,
        grad_output:    &Tensor,
        grad_input:     &mut Tensor,
        weight:         &Tensor,
        finput:         &Tensor,
        fgrad_input:    &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 3,
          "It is expected kernel_size equals to 3, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 3,
          "It is expected dilation equals to 3, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 3,
          "It is expected padding equals to 3, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 3,
          "It is expected stride equals to 3, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 3,
          "It is expected stride equals to 3, but got size ",
          output_padding.size());

      Tensor grad_columns = finput;

      i64 kernel_depth = kernel_size[0];
      i64 kernel_height = kernel_size[1];
      i64 kernel_width = kernel_size[2];
      i64 dilation_depth = dilation[0];
      i64 dilation_height = dilation[1];
      i64 dilation_width = dilation[2];
      i64 padding_depth = padding[0];
      i64 padding_height = padding[1];
      i64 padding_width = padding[2];
      i64 stride_depth = stride[0];
      i64 stride_height = stride[1];
      i64 stride_width = stride[2];
      i64 output_padding_depth = output_padding[0];
      i64 output_padding_height = output_padding[1];
      i64 output_padding_width = output_padding[2];

      // number of input & output planes and kernel size is indirectly defined by
      // the weight tensor
      slow_conv_transpose3d_shape_check(
          input_,
          grad_output_,
          weight_,
          Tensor(),
          kernel_depth,
          kernel_width,
          kernel_height,
          stride_depth,
          stride_width,
          stride_height,
          padding_depth,
          padding_width,
          padding_height,
          dilation_depth,
          dilation_width,
          dilation_height,
          output_padding_depth,
          output_padding_width,
          output_padding_height,
          0);

      Tensor input = input_.contiguous();
      Tensor weight = weight_.contiguous();
      Tensor grad_output = grad_output_.contiguous();

      const i64 n_input_plane = weight.size(0);
      const i64 n_output_plane = weight.size(1);

      bool is_batch = false;
      if (input.dim() == 4) {
        // Force batch
        is_batch = true;
        input.resize_(
            {1, input.size(0), input.size(1), input.size(2), input.size(3)});
        grad_output.resize_({1,
                             grad_output.size(0),
                             grad_output.size(1),
                             grad_output.size(2),
                             grad_output.size(3)});
      }

      const i64 input_width = input.size(4);
      const i64 input_height = input.size(3);
      const i64 input_depth = input.size(2);

      const i64 output_depth = (input_depth - 1) * stride_depth -
          2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
          output_padding_depth;
      const i64 output_height = (input_height - 1) * stride_height -
          2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
          output_padding_height;
      const i64 output_width = (input_width - 1) * stride_width -
          2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
          output_padding_width;

      // Batch size + input planes
      const i64 batch_size = input.size(0);

      // Resize output
      grad_input.resize_(
          {batch_size, n_input_plane, input_depth, input_height, input_width});
      grad_input.zero_();

      // Resize temporary columns
      grad_columns.resize_(
          {n_output_plane * kernel_width * kernel_height * kernel_depth,
           input_depth * input_height * input_width});

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "slow_conv_transpose3d_backward_out_cpu", [&] {
            // Helpers
            Tensor grad_input_n;
            Tensor grad_output_n;

            i64 elt;
            // For each elt in batch, do:
            for (elt = 0; elt < batch_size; ++elt) {
              // Matrix mulitply per sample:
              grad_input_n = grad_input.select(0, elt);
              grad_output_n = grad_output.select(0, elt);

              if (kernel_depth != 1 || kernel_height != 1 || kernel_width != 1 ||
                  stride_depth != 1 || stride_height != 1 || stride_width != 1 ||
                  dilation_depth != 1 || dilation_height != 1 ||
                  dilation_width != 1 || padding_depth != 0 ||
                  padding_height != 0 || padding_width != 0) {
                // Extract columns:
                native::vol2col<Scalar>(
                    grad_output_n.data_ptr<Scalar>(),
                    n_output_plane,
                    output_depth,
                    output_height,
                    output_width,
                    input_depth,
                    input_height,
                    input_width,
                    kernel_depth,
                    kernel_height,
                    kernel_width,
                    padding_depth,
                    padding_height,
                    padding_width,
                    stride_depth,
                    stride_height,
                    stride_width,
                    dilation_depth,
                    dilation_height,
                    dilation_width,
                    grad_columns.data_ptr<Scalar>());
              }

              // M,N,K are dims of matrix A and B
              // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
              const i64 m = weight.size(0);
              const i64 n = grad_columns.size(1);
              const i64 k =
                  weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);

              // Do GEMM (note: this is a bit confusing because gemm assumes
              // column-major matrices)
              auto gemm_in_ptr =
                  (kernel_depth != 1 || kernel_height != 1 || kernel_width != 1 ||
                   stride_depth != 1 || stride_height != 1 || stride_width != 1 ||
                   dilation_depth != 1 || dilation_height != 1 ||
                   dilation_width != 1 || padding_depth != 0 ||
                   padding_height != 0 || padding_width != 0)
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
              grad_output.resize_(
                  {n_output_plane, output_depth, output_height, output_width});
              input.resize_(
                  {n_input_plane, input_depth, input_height, input_width});
              grad_input.resize_(
                  {n_input_plane, input_depth, input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose3d_acc_grad_parameters_cpu(
        input:          &Tensor,
        grad_output:    &Tensor,
        grad_weight:    &mut Tensor,
        grad_bias:      &mut Tensor,
        finput:         &Tensor,
        fgrad_input:    &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        scale:          i32)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 3,
          "It is expected kernel_size equals to 3, but got size ",
          kernel_size.size());

      TORCH_CHECK(
          dilation.size() == 3,
          "It is expected dilation equals to 3, but got size ",
          dilation.size());

      TORCH_CHECK(
          padding.size() == 3,
          "It is expected padding equals to 3, but got size ",
          padding.size());

      TORCH_CHECK(
          stride.size() == 3,
          "It is expected stride equals to 3, but got size ",
          stride.size());

      TORCH_CHECK(
          output_padding.size() == 3,
          "It is expected stride equals to 3, but got size ",
          output_padding.size());

      i64 kernel_depth = kernel_size[0];
      i64 kernel_height = kernel_size[1];
      i64 kernel_width = kernel_size[2];
      i64 dilation_depth = dilation[0];
      i64 dilation_height = dilation[1];
      i64 dilation_width = dilation[2];
      i64 padding_depth = padding[0];
      i64 padding_height = padding[1];
      i64 padding_width = padding[2];
      i64 stride_depth = stride[0];
      i64 stride_height = stride[1];
      i64 stride_width = stride[2];
      i64 output_padding_depth = output_padding[0];
      i64 output_padding_height = output_padding[1];
      i64 output_padding_width = output_padding[2];

      // number of input & output planes and kernel size is indirectly defined by
      // the grad_weight tensor
      slow_conv_transpose3d_shape_check(
          input_,
          grad_output_,
          grad_weight,
          grad_bias,
          kernel_depth,
          kernel_width,
          kernel_height,
          stride_depth,
          stride_width,
          stride_height,
          padding_depth,
          padding_width,
          padding_height,
          dilation_depth,
          dilation_width,
          dilation_height,
          output_padding_depth,
          output_padding_width,
          output_padding_height,
          1);

      i64 n_output_plane;
      if (grad_weight.defined()) {
        n_output_plane = grad_weight.size(1);
      } else if (grad_bias.defined()) {
        n_output_plane = grad_bias.size(0);
      } else {
        return;
      }

      Tensor columns = finput;
      Tensor ones = fgrad_input;

      Tensor input = input_.contiguous();
      Tensor grad_output = grad_output_.contiguous();

      if (grad_weight.defined()) {
        TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
      }
      if (grad_bias.defined()) {
        TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
        TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
      }

      bool is_batch = false;
      if (input.dim() == 4) {
        // Force batch
        is_batch = true;
        input.resize_(
            {1, input.size(0), input.size(1), input.size(2), input.size(3)});
        grad_output.resize_({1,
                             grad_output.size(0),
                             grad_output.size(1),
                             grad_output.size(2),
                             grad_output.size(3)});
      }

      const i64 input_width = input.size(4);
      const i64 input_height = input.size(3);
      const i64 input_depth = input.size(2);

      const i64 output_depth = (input_depth - 1) * stride_depth -
          2 * padding_depth + (dilation_depth * (kernel_depth - 1) + 1) +
          output_padding_depth;
      const i64 output_height = (input_height - 1) * stride_height -
          2 * padding_height + (dilation_height * (kernel_height - 1) + 1) +
          output_padding_height;
      const i64 output_width = (input_width - 1) * stride_width -
          2 * padding_width + (dilation_width * (kernel_width - 1) + 1) +
          output_padding_width;

      // Batch size + input planes
      const i64 batch_size = input.size(0);

      // Define a buffer of ones, for bias accumulation
      if (ones.dim() != 3 ||
          ones.size(0) * ones.size(1) * ones.size(2) <
              output_depth * output_height * output_width) {
        // Resize plane and fill with ones...
        ones.resize_({output_depth, output_height, output_width});
        ones.fill_(1);
      }

      // Resize temporary columns
      columns.resize_({n_output_plane * kernel_width * kernel_height * kernel_depth,
                       input_depth * input_height * input_width});

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(),
          "slow_conv_transpose3d_acc_grad_parameters_cpu",
          [&] {
            // Helpers
            Tensor input_n;
            Tensor grad_output_n;

            Scalar scale = static_cast<Scalar>(scale_);

            i64 elt;
            // For each elt in batch, do:
            for (elt = 0; elt < batch_size; ++elt) {
              // Matrix mulitply per output:
              grad_output_n = grad_output.select(0, elt);

              // Do Weight:
              if (grad_weight.defined()) {
                // Matrix mulitply per output:
                input_n = input.select(0, elt);

                if (kernel_depth != 1 || kernel_height != 1 || kernel_width != 1 ||
                    stride_depth != 1 || stride_height != 1 || stride_width != 1 ||
                    dilation_depth != 1 || dilation_height != 1 ||
                    dilation_width != 1 || padding_depth != 0 ||
                    padding_height != 0 || padding_width != 0) {
                  // Extract columns:
                  native::vol2col<Scalar>(
                      grad_output_n.data_ptr<Scalar>(),
                      n_output_plane,
                      output_depth,
                      output_height,
                      output_width,
                      input_depth,
                      input_height,
                      input_width,
                      kernel_depth,
                      kernel_height,
                      kernel_width,
                      padding_depth,
                      padding_height,
                      padding_width,
                      stride_depth,
                      stride_height,
                      stride_width,
                      dilation_depth,
                      dilation_height,
                      dilation_width,
                      columns.data_ptr<Scalar>());
                }

                // M,N,K are dims of matrix A and B
                // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
                const i64 n = columns.size(0); // n_output_plane * kt * kh * kw
                const i64 m = input_n.size(0); // n_input_plane
                const i64 k = columns.size(1); // input_height * input_width

                // Do GEMM (note: this is a bit confusing because gemm assumes
                // column-major matrices)
                auto gemm_in_ptr =
                    (kernel_depth != 1 || kernel_height != 1 || kernel_width != 1 ||
                     stride_depth != 1 || stride_height != 1 || stride_width != 1 ||
                     dilation_depth != 1 || dilation_height != 1 ||
                     dilation_width != 1 || padding_depth != 0 ||
                     padding_height != 0 || padding_width != 0)
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
                const i64 m_ = n_output_plane;
                const i64 k_ = output_depth * output_height * output_width;

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
              grad_output.resize_(
                  {n_output_plane, output_depth, output_height, output_width});
              input.resize_(
                  {input.size(1), input_depth, input_height, input_width});
            }
          });
        */
}

pub fn slow_conv_transpose3d_out_cpu(
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        bias_opt:       &Option<Tensor>,
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        output:         &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor finput = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor fgrad = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      slow_conv_transpose3d_out_cpu_template(
          output,
          input,
          weight,
          kernel_size,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          finput,
          fgrad);

      return output;
        */
}

pub fn slow_conv_transpose3d_cpu(
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
      Tensor finput = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor fgrad = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      slow_conv_transpose3d_out_cpu_template(
          output,
          input,
          weight,
          kernel_size,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          finput,
          fgrad);

      return output;
        */
}

pub fn slow_conv_transpose3d_backward_out_cpu(
        grad_output:    &Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        finput:         &Tensor,
        fgrad:          &Tensor,
        grad_input:     &mut Tensor,
        grad_weight:    &mut Tensor,
        grad_bias:      &mut Tensor) -> (&mut Tensor,&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            if (grad_input.defined()) {
        slow_conv_transpose3d_backward_out_cpu_template(
            input,
            grad_output,
            grad_input,
            weight,
            finput,
            fgrad,
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
        slow_conv_transpose3d_acc_grad_parameters_cpu(
            input,
            grad_output,
            grad_weight,
            grad_bias,
            finput,
            fgrad,
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

pub fn slow_conv_transpose3d_backward_cpu(
        grad_output:    &Tensor,
        input:          &Tensor,
        weight:         &Tensor,
        kernel_size:    &[i32],
        stride:         &[i32],
        padding:        &[i32],
        output_padding: &[i32],
        dilation:       &[i32],
        finput:         &Tensor,
        fgrad:          &Tensor,
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
        slow_conv_transpose3d_backward_out_cpu_template(
            input,
            grad_output,
            grad_input,
            weight,
            finput,
            fgrad,
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
        slow_conv_transpose3d_acc_grad_parameters_cpu(
            input,
            grad_output,
            grad_weight,
            grad_bias,
            finput,
            fgrad,
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

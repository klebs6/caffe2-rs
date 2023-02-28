crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Im2Col.cpp]

pub fn im2col_out_cpu_template(
        output:      &mut Tensor,
        input:       &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32])  {
    
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

      i64 kernel_height = kernel_size[0];
      i64 kernel_width = kernel_size[1];
      i64 dilation_height = dilation[0];
      i64 dilation_width = dilation[1];
      i64 pad_height = padding[0];
      i64 pad_width = padding[1];
      i64 stride_height = stride[0];
      i64 stride_width = stride[1];

      im2col_shape_check(
          input_,
          Tensor(),
          kernel_height,
          kernel_width,
          dilation_height,
          dilation_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width);

      Tensor input = input_.contiguous();

      bool batched_input = true;

      if (input.dim() == 3) {
        batched_input = false;
        input.resize_({1, input.size(0), input.size(1), input.size(2)});
      }

      i64 batch_size = input.size(0);
      i64 n_input_plane = input.size(1);
      i64 input_height = input.size(2);
      i64 input_width = input.size(3);

      i64 output_height = (input_height + 2 * pad_height -
                               (dilation_height * (kernel_height - 1) + 1)) /
              stride_height +
          1;
      i64 output_width = (input_width + 2 * pad_width -
                              (dilation_width * (kernel_width - 1) + 1)) /
              stride_width +
          1;
      i64 n_output_plane = n_input_plane * kernel_width * kernel_height;
      i64 output_length = output_height * output_width;

      output.resize_({batch_size, n_output_plane, output_length});
      output.zero_();

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
          input.scalar_type(), "im2col_out_cpu", [&] {
            Tensor input_n;
            Tensor output_n;

            for (i64 elt = 0; elt < batch_size; elt++) {
              input_n = input.select(0, elt);
              output_n = output.select(0, elt);

              im2col<Scalar>(
                  input_n.data_ptr<Scalar>(),
                  n_input_plane,
                  input_height,
                  input_width,
                  output_height,
                  output_width,
                  kernel_height,
                  kernel_width,
                  pad_height,
                  pad_width,
                  stride_height,
                  stride_width,
                  dilation_height,
                  dilation_width,
                  output_n.data_ptr<Scalar>());
            }

            if (!batched_input) {
              output.resize_({n_output_plane, output_length});
            }
          });
        */
}

pub fn im2col_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input_size:  &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(
          input_size.size() == 2,
          "It is expected input_size equals to 2, but got size ",
          input_size.size());
      // col2im_out_cpu checks size of kernel_size, dilation, padding and stride
      native::col2im_out_cpu(
          grad_output,
          input_size,
          kernel_size,
          dilation,
          padding,
          stride,
          grad_input);
        */
}

pub fn im2col_out_cpu(
        input:       &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32],
        output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            im2col_out_cpu_template(
          output, input, kernel_size, dilation, padding, stride);
      return output;
        */
}

pub fn im2col_cpu(
        input:       &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor output = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      im2col_out_cpu_template(
          output, input, kernel_size, dilation, padding, stride);
      return output;
        */
}

pub fn im2col_backward_out_cpu(
        grad_output: &Tensor,
        input_size:  &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32],
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            im2col_backward_out_cpu_template(
          grad_input,
          grad_output,
          input_size,
          kernel_size,
          dilation,
          padding,
          stride);
      return grad_input;
        */
}

pub fn im2col_backward_cpu(
        grad_output: &Tensor,
        input_size:  &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      im2col_backward_out_cpu_template(
          grad_input,
          grad_output,
          input_size,
          kernel_size,
          dilation,
          padding,
          stride);
      return grad_input;
        */
}

/*!
  | Note [im2col/col2im output padding]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Our implementations of im2col and col2im take
  | both the input height/width as well as
  | a seemingly redundant output height/width.  In
  | principle, you could compute the output
  | height/width by using the convolution shape
  | formulas.  So, what's up with that?
  |
  | The trouble arises when one runs the backward
  | of a transposed convolution with output_padding
  | >= stride.  (BTW, output_padding is known as
  | adj inside THNN.) Let's consider a simple case
  | where we have kernel=2, dilation=2, stride=1,
  | output_padding=1 for a 4x4 input:
  |
  | Input:  X
  |
  | Output: X.X.
  |         ....
  |         X.X.
  |         ....
  |
  | If we compute backwards of output with
  | a standard convolution on the output with the
  | same parameters, we would end up with a 2x2
  | grad_input (because you can slide the stencil
  | over to the right once and down once).  But
  | that is all out-of-bounds if you're computing
  | backwards for a 1x1 input.
  |
  | "Now Edward," you might say, "the real problem
  | is that you set output_padding >= stride,
  | surely an error should have been raised in this
  | case."  To understand why it is useful to
  | handle this case, we have to understand how we
  | compute the weight gradient of a convolution.
  | Suppose we have a convolution with kernel=2,
  | stride=2 on a 5x5 input.  Let us see all the
  | contributions of weight[0][0] (which we have
  | labeled w) in the output:
  |
  | Input:  a.b..  Weight: w.
  |         .....          ..
  |         c.d..
  |         .....
  |         .....
  |
  | Output: [ aw+...  bw+... ]
  |         [ cw+...  dw+... ]
  |
  | From this diagram, it easy to see that we can
  | compute the weight gradient by performing
  | a *dilated* convolution between the input and
  | the output gradients with kernel=2, dilation=2,
  | stride=1.  But there's a rub: if we do
  | a dilated convolution directly, we'll end up
  | with a 3x3 weight gradient, when we clearly
  | wanted a 2x2.  So how do we avoid going out of
  | bounds?  We could add a notion of
  | 'output_padding' for non-transposed
  | convolution, but another simple and effective
  | fix is to just accept the desired output size
  | directly, and compute only within those bounds.
  |
  |
  | ALSO do vol2col
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Col2Im.cpp]

pub fn col2im_out_cpu_template(
        output:      &mut Tensor,
        input:       &Tensor,
        output_size: &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 2,
          "It is expected output_size equals to 2, but got size ",
          output_size.size());

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

      i64 output_height = output_size[0];
      i64 output_width = output_size[1];
      i64 kernel_height = kernel_size[0];
      i64 kernel_width = kernel_size[1];
      i64 dilation_height = dilation[0];
      i64 dilation_width = dilation[1];
      i64 pad_height = padding[0];
      i64 pad_width = padding[1];
      i64 stride_height = stride[0];
      i64 stride_width = stride[1];

      col2im_shape_check(
          input_,
          Tensor(),
          output_height,
          output_width,
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
      if (input.dim() == 2) {
        // Force batch
        batched_input = false;
        input.resize_({1, input.size(0), input.size(1)});
      }

      i64 batch_size = input.size(0);
      i64 n_input_plane = input.size(1);
      i64 n_output_plane = n_input_plane / (kernel_width * kernel_height);

      output.resize_({batch_size, n_output_plane, output_height, output_width});
      output.zero_();

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
          input.scalar_type(), "col2im_out_cpu", [&] {
            Tensor input_n = Tensor();
            Tensor output_n = Tensor();

            i64 height_col = (output_height + 2 * pad_height -
                                  (dilation_height * (kernel_height - 1) + 1)) /
                    stride_height +
                1;
            i64 width_col = (output_width + 2 * pad_width -
                                 (dilation_width * (kernel_width - 1) + 1)) /
                    stride_width +
                1;

            for (i64 elt = 0; elt < batch_size; elt++) {
              input_n = input.select(0, elt);
              output_n = output.select(0, elt);

              col2im<Scalar>(
                  input_n.data_ptr<Scalar>(),
                  n_output_plane,
                  output_height,
                  output_width,
                  height_col,
                  width_col,
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
              output.resize_({n_output_plane, output_height, output_width});
            }
          });
        */
}

pub fn col2im_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32])  {
    
    todo!();
        /*
            // im2col_out_cpu checks size of kernel_size, dilation, padding and stride
      native::im2col_out_cpu(
          grad_output, kernel_size, dilation, padding, stride, grad_input);
        */
}

pub fn col2im_out_cpu<'a>(
        input:       &Tensor,
        output_size: &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32],
        output:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            col2im_out_cpu_template(
          output, input, output_size, kernel_size, dilation, padding, stride);
      return output;
        */
}

pub fn col2im_cpu(
        input:       &Tensor,
        output_size: &[i32],
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor output = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      col2im_out_cpu_template(
          output, input, output_size, kernel_size, dilation, padding, stride);
      return output;
        */
}

pub fn col2im_backward_out_cpu<'a>(
        grad_output: &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32],
        grad_input:  &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            col2im_backward_out_cpu_template(
          grad_input, grad_output, kernel_size, dilation, padding, stride);
      return grad_input;
        */
}

pub fn col2im_backward_cpu(
        grad_output: &Tensor,
        kernel_size: &[i32],
        dilation:    &[i32],
        padding:     &[i32],
        stride:      &[i32]) -> Tensor {
    
    todo!();
        /*
            Tensor grad_input = empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      col2im_backward_out_cpu_template(
          grad_input, grad_output, kernel_size, dilation, padding, stride);
      return grad_input;
        */
}

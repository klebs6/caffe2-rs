crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/im2col_shape_check.h]

#[inline] pub fn col2im_shape_check(
        input:           &Tensor,
        grad_output:     &Tensor,
        output_height:   i64,
        output_width:    i64,
        kernel_height:   i64,
        kernel_width:    i64,
        dilation_height: i64,
        dilation_width:  i64,
        pad_height:      i64,
        pad_width:       i64,
        stride_height:   i64,
        stride_width:    i64)  {
    
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
          " dilation_width: ",
          dilation_width);

      i64 ndim = input.ndimension();
      // allow dim=0 only the batch dimension.
      TORCH_CHECK(
          (ndim == 2 && input.size(0) != 0 && input.size(1) != 0) ||
          (ndim == 3 && input.size(1) != 0 && input.size(2) != 0),
          "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input, but got: ",
          input.sizes());

      i64 batch_dim = (ndim == 3) ? 0 : -1;
      i64 n_input_plane = input.size(batch_dim + 1);

      if (n_input_plane % (kernel_width * kernel_height) != 0) {
        AT_ERROR(
            "Expected size of input's dimension 1 to be divisible by the "
            "product of kernel_size, but got input.size(1)=",
            n_input_plane,
            " and kernel_size=(",
            kernel_height,
            ", ",
            kernel_width,
            ").");
      }

      i64 input_length = input.size(batch_dim + 2);
      i64 n_blocks_height =
          div_rtn<i64>(
              output_height + 2 * pad_height -
                  dilation_height * (kernel_height - 1) - 1,
              stride_height) +
          1;
      i64 n_blocks_width = div_rtn<i64>(
                                       output_width + 2 * pad_width -
                                           dilation_width * (kernel_width - 1) - 1,
                                       stride_width) +
          1;

      if (input_length != (n_blocks_height * n_blocks_width)) {
        AT_ERROR(
            "Given output_size=(",
            output_height,
            ", ",
            output_width,
            "), kernel_size=(",
            kernel_height,
            ", ",
            kernel_width,
            "), dilation=(",
            dilation_height,
            ", ",
            dilation_width,
            "), padding=(",
            pad_height,
            ", ",
            pad_width,
            "), stride=(",
            stride_height,
            ", ",
            stride_width,
            "), expected size of input's dimension 2 to match the calculated number of ",
            "sliding blocks ",
            n_blocks_height,
            " * ",
            n_blocks_width,
            " = ",
            (n_blocks_height * n_blocks_width),
            ", but got input.size(2)=",
            input_length,
            ".");
      }

      if (output_width < 1 || output_height < 1) {
        AT_ERROR(
            "Expected output spatial size to be positive, but got: output_size=(",
            output_height,
            ", ",
            output_width,
            ").");
      }
        */
}

#[inline] pub fn im2col_shape_check(
        input:           &Tensor,
        grad_output:     &Tensor,
        kernel_height:   i64,
        kernel_width:    i64,
        dilation_height: i64,
        dilation_width:  i64,
        pad_height:      i64,
        pad_width:       i64,
        stride_height:   i64,
        stride_width:    i64)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_width > 0 && kernel_height > 0,
          "kernel size should be greater than zero, but got kernel_height: ",
          kernel_height,
          " kernel_width: ",
          kernel_width);

      TORCH_CHECK(
          dilation_width > 0 && dilation_height > 0,
          "dilation should be greater than zero, but got dilation_height: ",
          dilation_height,
          " dilation_width: ",
          dilation_width);

      TORCH_CHECK(
          pad_width >= 0 && pad_height >= 0,
          "padding should be non-negative, but got pad_height: ",
          pad_height,
          " pad_width: ",
          pad_width);

      TORCH_CHECK(
          stride_width > 0 && stride_height > 0,
          "stride should be greater than zero, but got stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width);

      i64 ndim = input.ndimension();

      // allow dim=0 only the batch dimension.
      bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
      TORCH_CHECK(
          (ndim == 3 && input.size(0) && valid_dims) ||
          (ndim == 4 && valid_dims && input.size(3) != 0),
          "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
          input.sizes());

      i64 dim_batch = 0;

      if (ndim == 3) {
        dim_batch = -1;
      }

      i64 input_height = input.size(dim_batch + 2);
      i64 input_width = input.size(dim_batch + 3);
      i64 output_height = div_rtn<i64>(
                                  input_height + 2 * pad_height -
                                      (dilation_height * (kernel_height - 1) + 1),
                                  stride_height) +
          1;
      i64 output_width = div_rtn<i64>(
                                 input_width + 2 * pad_width -
                                     (dilation_width * (kernel_width - 1) + 1),
                                 stride_width) +
          1;

      if (output_height < 1 || output_width < 1) {
        AT_ERROR(
            "Given input with spatial size (",
            input_height,
            ", ",
            input_height,
            "), kernel_size=(",
            kernel_height,
            ", ",
            kernel_width,
            "), dilation=(",
            dilation_height,
            ", ",
            dilation_width,
            "), padding=(",
            pad_height,
            ", ",
            pad_width,
            "), calculated shape of the array of sliding blocks as (",
            output_height,
            ", ",
            output_width,
            "), which is too small (non-positive).");
      }
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Pool.h]

lazy_static!{
    /*
    using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input,
        int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);

    using max_pool2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
    */
}

declare_dispatch!{max_pool2d_fn, max_pool2d_kernel}
declare_dispatch!{max_pool2d_backward_fn, max_pool2d_backward_kernel}

// averge pooling has same signature for forward and backward
lazy_static!{
    /*
    using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, int kW, int kH,
        int dW, int dH, int padW, int padH, bool count_include_pad, optional<i64> divisor_override);
    */
}

declare_dispatch!{avg_pool2d_fn, avg_pool2d_kernel}
declare_dispatch!{avg_pool2d_fn, avg_pool2d_backward_kernel}

#[inline] pub fn safe_downcast<dest_t, src_t>(v: Src) -> Dest {

    todo!();
        /*
            TORCH_CHECK(dest_t::min <= v && v <= dest_t::max,
                  "integer out of range");

      return static_cast<dest_t>(v);
        */
}

#[inline] pub fn pooling_output_shape_pad_lr<T>(
        input_size:  T,
        kernel_size: T,
        pad_l:       T,
        pad_r:       T,
        stride:      T,
        dilation:    T,
        ceil_mode:   bool) -> T {

    todo!();
        /*
            T outputSize = div_rtn<T>(
            inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
            (ceil_mode ? stride - 1 : 0), stride) + 1;
        if (ceil_mode) {
            // ensure that the last pooling starts inside the image
            // needed to avoid problems in ceil mode
            if ((outputSize - 1) * stride >= inputSize + pad_l) {
              --outputSize;
            }
        }
        return outputSize;
        */
}


#[inline] pub fn pooling_output_shape<T>(
        input_size:  T,
        kernel_size: T,
        pad:         T,
        stride:      T,
        dilation:    T,
        ceil_mode:   bool) -> T {

    todo!();
        /*
            TORCH_CHECK(stride != 0, "stride should not be zero");
        return pooling_output_shape_pad_lr(
            inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
        */
}

#[inline] pub fn pooling_same_mode_padding_lr(
        input_size:  i64,
        kernel_size: i64,
        stride:      i64,
        dilation:    i64) -> (i64,i64) {
    
    todo!();
        /*
            // NOTE: with strides, the output shape is ceil(inputSize/stride)
      auto total_padding = dilation * (kernelSize - 1);

      // Prefer symmetric padding if possible
      if (stride > 2 && (total_padding % 2 == 1)) {
        // The floor in the output size calculation gives us a little wiggle room
        auto wiggle_room = inputSize % stride - 1;
        if (wiggle_room > 0) {
          --total_padding;
        }
      }

      auto left = total_padding / 2;
      return {left, total_padding - left};
        */
}

/// AveragePool2d/DilatedMaxPool2d (forward)
///
#[inline] pub fn pool2d_shape_check(
        input:         &Tensor,
        kh:            i32,
        kw:            i32,
        dh:            i32,
        dw:            i32,
        padh:          i32,
        padw:          i32,
        dilationh:     i32,
        dilationw:     i32,
        n_input_plane: i64,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        memory_format: MemoryFormat)  {
    
    todo!();
        /*
            const i64 ndim = input.ndimension();
      const i64 nOutputPlane = nInputPlane;

      TORCH_CHECK(kW > 0 && kH > 0,
                  "kernel size should be greater than zero, but got ",
                  "kH: ", kH, " kW: ", kW);
      TORCH_CHECK(dW > 0 && dH > 0,
                  "stride should be greater than zero, but got "
                  "dH: ", dH, " dW: ", dW);
      TORCH_CHECK(dilationH > 0 && dilationW > 0,
                  "dilation should be greater than zero, but got ",
                  "dilationH: ", dilationH, " dilationW: ", dilationW);

      bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
      if (memory_format == MemoryFormat::ChannelsLast){
        // Expect tensor in NHWC format and allow 0-dim only for N.
        TORCH_CHECK((ndim == 4 && valid_dims && input.size(3) != 0),
          "Expected 4D (batch mode) tensor expected for input with channels_last layout"
          " with optional 0 dim batch size for input, but got: ", input.sizes());
      } else {
        TORCH_CHECK((ndim == 3 && input.size(0) != 0 && valid_dims) ||
          (ndim == 4 && valid_dims && input.size(3) != 0),
          "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:",
          input.sizes());
      }

      TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
                  "pad should be smaller than or equal to half of kernel size, but got ",
                  "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

      TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
                  "Given input size: (",
                  nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
                  "Calculated output size: (",
                  nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
                  "Output size is too small");
        */
}

/// DilatedMaxPool2d (backward)
///
#[inline] pub fn max_pool2d_backward_shape_check(
        input:         &Tensor,
        grad_output:   &Tensor,
        indices:       &Tensor,
        nbatch:        i64,
        kh:            i32,
        kw:            i32,
        dh:            i32,
        dw:            i32,
        padh:          i32,
        padw:          i32,
        dilationh:     i32,
        dilationw:     i32,
        n_input_plane: i64,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        memory_format: MemoryFormat,
        cuda:          bool)  {
    let cuda: bool = cuda.unwrap_or(false);

    todo!();
        /*
            pool2d_shape_check(
        input,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW,
        nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, memory_format);

      const i64 ndim = input.ndimension();
      const i64 nOutputPlane = nInputPlane;

      check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
      check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
      check_dim_size(gradOutput, ndim, ndim-1, outputWidth);

      check_dim_size(indices, ndim, ndim-3, nOutputPlane);
      check_dim_size(indices, ndim, ndim-2, outputHeight);
      check_dim_size(indices, ndim, ndim-1, outputWidth);
        */
}

/// AveragePool2d (backward)
///
#[inline] pub fn avg_pool2d_backward_shape_check(
        input:         &Tensor,
        grad_output:   &Tensor,
        nbatch:        i64,
        kh:            i32,
        kw:            i32,
        dh:            i32,
        dw:            i32,
        padh:          i32,
        padw:          i32,
        n_input_plane: i64,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        memory_format: MemoryFormat)  {
    
    todo!();
        /*
            pool2d_shape_check(
        input,
        kH, kW, dH, dW, padH, padW, 1, 1,
        nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
        memory_format);

      const i64 ndim = input.ndimension();
      const i64 nOutputPlane = nInputPlane;

      check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
      check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
      check_dim_size(gradOutput, ndim, ndim-1, outputWidth);
        */
}

/// AveragePool3d/DilatedMaxPool3d (forward)
#[inline] pub fn pool3d_shape_check(
        input:            &Tensor,
        nslices:          i64,
        kt:               i32,
        kh:               i32,
        kw:               i32,
        dt:               i32,
        dh:               i32,
        dw:               i32,
        pt:               i32,
        ph:               i32,
        pw:               i32,
        dilationt:        i32,
        dilationh:        i32,
        dilationw:        i32,
        itime:            i64,
        iheight:          i64,
        iwidth:           i64,
        otime:            i64,
        oheight:          i64,
        owidth:           i64,
        check_input_size: bool)  {
    let check_input_size: bool = check_input_size.unwrap_or(false);

    todo!();
        /*
            const i64 ndim = input.ndimension();

      TORCH_CHECK(kT > 0 && kW > 0 && kH > 0,
                  "kernel size should be greater than zero, but got ",
                  "kT: ", kT, " kH: ", kH, " kW: ", kW);
      TORCH_CHECK(dT > 0 && dW > 0 && dH > 0,
                  "stride should be greater than zero, but got ",
                  "dT: ", dT, " dH: ", dH, " dW: ", dW);
      TORCH_CHECK(dilationT > 0 && dilationW > 0 && dilationH > 0,
                  "dilation should be greater than zero, but got ",
                  "dilationT: ", dilationT, " dilationH: ", dilationH, " dilationW: ", dilationW);

      TORCH_CHECK(input.numel() > 0 && (ndim == 4 || ndim == 5),
                  "non-empty 4D or 5D (batch mode) tensor expected for input, but got ndim: ", ndim);

      if (check_input_size) { // AveragePool3d
        TORCH_CHECK(itime >= kT && iheight >= kH && iwidth >= kW,
                    "input image ", "(T: ", itime, " H: ", iheight, " W: ", iwidth, ") smaller than ",
                    "kernel size ", "(kT: ", kT, " kH: ", kH, " kW: ", kW, ")");
      }

      TORCH_CHECK(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH,
                  "pad should be smaller than or equal to half of kernel size, but got "
                  "kT: ", kT, " kW: ", kW, " kH: ", kH, " padT: ", pT, " padW: ", pW, " padH: ", pH);

      TORCH_CHECK(otime >= 1 && owidth >= 1 && oheight >= 1,
                  "Given input size: (",
                  nslices,"x", itime, "x", iheight, "x", iwidth, "). ",
                  "Calculated output size: (",
                  nslices, "x", otime, "x", oheight, "x", owidth, "). ",
                  "Output size is too small");
        */
}

#[inline] pub fn max_pool3d_backward_shape_check(
        input:       &Tensor,
        grad_output: &Tensor,
        indices:     &Tensor,
        nslices:     i64,
        kt:          i32,
        kh:          i32,
        kw:          i32,
        dt:          i32,
        dh:          i32,
        dw:          i32,
        pt:          i32,
        ph:          i32,
        pw:          i32,
        dilationt:   i32,
        dilationh:   i32,
        dilationw:   i32,
        itime:       i64,
        iheight:     i64,
        iwidth:      i64,
        otime:       i64,
        oheight:     i64,
        owidth:      i64)  {
    
    todo!();
        /*
            const i64 ndim = input.ndimension();

      pool3d_shape_check(
        input,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        pT, pH, pW,
        dilationT, dilationH, dilationW,
        itime, iheight, iwidth,
        otime, oheight, owidth);

      check_dim_size(gradOutput, ndim, ndim-4, nslices);
      check_dim_size(gradOutput, ndim, ndim-3, otime);
      check_dim_size(gradOutput, ndim, ndim-2, oheight);
      check_dim_size(gradOutput, ndim, ndim-1, owidth);

      check_dim_size(indices, ndim, ndim-4, nslices);
      check_dim_size(indices, ndim, ndim-3, otime);
      check_dim_size(indices, ndim, ndim-2, oheight);
      check_dim_size(indices, ndim, ndim-1, owidth);
        */
}

#[inline] pub fn avg_pool3d_backward_shape_check(
        input:       &Tensor,
        grad_output: &Tensor,
        nslices:     i64,
        kt:          i32,
        kh:          i32,
        kw:          i32,
        dt:          i32,
        dh:          i32,
        dw:          i32,
        pt:          i32,
        ph:          i32,
        pw:          i32,
        itime:       i64,
        iheight:     i64,
        iwidth:      i64,
        otime:       i64,
        oheight:     i64,
        owidth:      i64)  {
    
    todo!();
        /*
            const i64 ndim = input.ndimension();

      pool3d_shape_check(
        input,
        nslices,
        kT, kH, kW,
        dT, dH, dW,
        pT, pH, pW,
        1, 1, 1,
        itime, iheight, iwidth,
        otime, oheight, owidth,
        true);

      check_dim_size(gradOutput, ndim, ndim-4, nslices);
      check_dim_size(gradOutput, ndim, ndim-3, otime);
      check_dim_size(gradOutput, ndim, ndim-2, oheight);
      check_dim_size(gradOutput, ndim, ndim-1, owidth);
        */
}

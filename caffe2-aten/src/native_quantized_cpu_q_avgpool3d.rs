crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/q_avgpool3d.cpp]

define_dispatch!{qavg_pool3d_nhwc_stub}

#[inline] pub fn get_kernel(kernel_size: &[i32]) -> (i32,i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_size.size() == 1 || kernel_size.size() == 3,
          "avg_pool3d: kernel_size must either be a single int, or a tuple of three ints");
      const int kD = safe_downcast<int, i64>(kernel_size[0]);
      const int kH = kernel_size.size() == 1
          ? kD
          : safe_downcast<int, i64>(kernel_size[1]);
      const int kW = kernel_size.size() == 1
          ? kD
          : safe_downcast<int, i64>(kernel_size[2]);
      return make_tuple(kW, kH, kD);
        */
}

#[inline] pub fn get_stride(
        stride: &[i32],
        kw:     i32,
        kh:     i32,
        kd:     i32) -> (i32,i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          stride.empty() || stride.size() == 1 || stride.size() == 3,
          "avg_pool3d: stride must either be omitted, a single int, or a tuple of three ints");
      const int dD = stride.empty() ? kD : safe_downcast<int, i64>(stride[0]);
      const int dH = stride.empty()
          ? kH
          : stride.size() == 1 ? dD : safe_downcast<int, i64>(stride[1]);
      const int dW = stride.empty()
          ? kW
          : stride.size() == 1 ? dD : safe_downcast<int, i64>(stride[2]);
      return make_tuple(dW, dH, dD);
        */
}

#[inline] pub fn get_padding(padding: &[i32]) -> (i32,i32,i32) {
    
    todo!();
        /*
            TORCH_CHECK(
          padding.size() == 1 || padding.size() == 3,
          "avg_pool3d: padding must either be a single int, or a tuple of three ints");
      const int padD = safe_downcast<int, i64>(padding[0]);
      const int padH =
          padding.size() == 1 ? padD : safe_downcast<int, i64>(padding[1]);
      const int padW =
          padding.size() == 1 ? padD : safe_downcast<int, i64>(padding[2]);
      return make_tuple(padW, padH, padD);
        */
}

pub fn get_output_shape(
        input:     &Tensor,
        kw:        i32,
        kh:        i32,
        kd:        i32,
        dw:        i32,
        dh:        i32,
        dd:        i32,
        padw:      i32,
        padh:      i32,
        padd:      i32,
        ceil_mode: bool) -> Vec<i64> {
    
    todo!();
        /*
            const i64 nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
      const i64 nInputPlane = input_.size(-4);
      const i64 inputDepth = input_.size(-3);
      const i64 inputHeight = input_.size(-2);
      const i64 inputWidth = input_.size(-1);
      const i64 outputDepth =
          pooling_output_shape<i64>(inputDepth, kD, padD, dD, 1, ceil_mode);
      const i64 outputHeight =
          pooling_output_shape<i64>(inputHeight, kH, padH, dH, 1, ceil_mode);
      const i64 outputWidth =
          pooling_output_shape<i64>(inputWidth, kW, padW, dW, 1, ceil_mode);
      if (input_.ndimension() == 4) {
        return {nInputPlane, outputDepth, outputHeight, outputWidth};
      }
      return {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
        */
}

pub fn q_avg_pool3d<Scalar>(
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {

    todo!();
        /*
      int kD, kW, kH, dD, dW, dH, padD, padW, padH;
      tie(kW, kH, kD) = get_kernel(kernel_size);
      tie(dW, dH, dD) = get_stride(stride, kW, kH, kD);
      tie(padW, padH, padD) = get_padding(padding);

      const i64 nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
      const i64 nInputPlane = input.size(-4);
      const i64 inputDepth = input.size(-3);
      const i64 inputHeight = input.size(-2);
      const i64 inputWidth = input.size(-1);

      TORCH_CHECK(
          !divisor_override.has_value() || divisor_override.value() != 0,
          "divisor must be not zero");

      auto output_shape =
          get_output_shape(input, kW, kH, kD, dW, dH, dD, padW, padH, padD, ceil_mode);
      const i64 outputDepth = output_shape[output_shape.size() - 3];
      const i64 outputHeight = output_shape[output_shape.size() - 2];
      const i64 outputWidth = output_shape[output_shape.size() - 1];

      auto input_nhwc = input.contiguous(MemoryFormat::ChannelsLast3d);

      auto output = _empty_affine_quantized(
          output_shape,
          input_nhwc.options().memory_format(input_nhwc.suggest_memory_format()),
          input_nhwc.q_scale(),
          input_nhwc.q_zero_point(),
          nullopt);
      // fast path for channel last: qavg_pool_2d_nhwc_stub
      if (output_shape.size() == 4) {
        qavg_pool3d_nhwc_stub(
            input_nhwc.device().type(),
            input_nhwc,
            output,
            0,
            nInputPlane,
            inputWidth,
            inputHeight,
            inputDepth,
            outputWidth,
            outputHeight,
            outputDepth,
            kW,
            kH,
            kD,
            dW,
            dH,
            dD,
            padW,
            padH,
            padD,
            count_include_pad,
            divisor_override);
      } else {
        parallel_for(0, nbatch, 0, [&](i64 start, i64 end) {
          for (auto b = start; b < end; b++) {
            qavg_pool3d_nhwc_stub(
                input_nhwc.device().type(),
                input_nhwc,
                output,
                b,
                nInputPlane,
                inputWidth,
                inputHeight,
                inputDepth,
                outputWidth,
                outputHeight,
                outputDepth,
                kW,
                kH,
                kD,
                dW,
                dH,
                dD,
                padW,
                padH,
                padD,
                count_include_pad,
                divisor_override);
          }
        });
      }
      return output;
        */
}

pub fn avg_pool3d_quantized_cpu(
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "avg_pool3d_quantized_cpu", [&]() {
        output = q_avg_pool3d<Scalar>(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override);
      });
      return output;
        */
}

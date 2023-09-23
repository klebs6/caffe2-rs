// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/DilatedMaxPool2d.cpp]

lazy_static!{
    /*
    TORCH_META_FUNC(max_pool2d_with_indices)
    (const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
      // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
      const int kH = safe_downcast<int, i64>(kernel_size[0]);
      const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, i64>(kernel_size[1]);

      // NB: stride default is not expressible as an integer constant, so we accept
      // empty stride for this case
      TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
        "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
      const int dH = stride.empty() ? kH : safe_downcast<int, i64>(stride[0]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dH : safe_downcast<int, i64>(stride[1]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "max_pool2d: padding must be either be a single int, or a tuple of two ints");
      const int padH = safe_downcast<int, i64>(padding[0]);
      const int padW = padding.size() == 1 ? padH : safe_downcast<int, i64>(padding[1]);

      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
        "max_pool2d: dilation must be either a single int, or a tuple of two ints");
      const int dilationH = safe_downcast<int, i64>(dilation[0]);
      const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, i64>(dilation[1]);

      const auto memory_format = input.suggest_memory_format();
      if (memory_format == MemoryFormat::ChannelsLast) {
        TORCH_CHECK(input.ndimension() == 4,
          "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
      } else if (memory_format == MemoryFormat::Contiguous) {
        TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
          "non-empty 3D or 4D (batch mode) tensor expected for input");
      } else {
        TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
      }

      /* sizes */
      const i64 nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
      const i64 nInputPlane = input.size(-3);
      const i64 inputHeight = input.size(-2);
      const i64 inputWidth = input.size(-1);

      const i64 outputHeight = pooling_output_shape<i64>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
      const i64 outputWidth = pooling_output_shape<i64>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

      pool2d_shape_check(
        input,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW,
        nInputPlane,
        inputHeight, inputWidth,
        outputHeight, outputWidth, memory_format);

      /* resize output and indices */
      &[Dimname] maybe_names = input.has_names() ? input.names() : &[Dimname]{};
      if (input.ndimension() == 3) {
        set_output(0, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
        /* indices will contain the locations for each output point */
        set_output(1, {nInputPlane, outputHeight, outputWidth}, {}, input.options().dtype(kLong), maybe_names);
      } else {
        set_output(0, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
        /* indices will contain the locations for each output point */
        set_output(1, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().dtype(kLong), maybe_names);
      }
    }
    */
}

lazy_static!{
    /*
    TORCH_META_FUNC(max_pool2d_with_indices_backward)
    (const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
      // #20866, #22032: Guarantee this for the official C++ API?
      TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
      const int kH = safe_downcast<int, i64>(kernel_size[0]);
      const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, i64>(kernel_size[1]);

      // NB: stride default is not expressible as an integer constant, so we accept
      // empty stride for this case
      TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
        "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
      const int dH = stride.empty() ? kH : safe_downcast<int, i64>(stride[0]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dH : safe_downcast<int, i64>(stride[1]);

      TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "max_pool2d: padding must be either be a single int, or a tuple of two ints");
      const int padH = safe_downcast<int, i64>(padding[0]);
      const int padW = padding.size() == 1 ? padH : safe_downcast<int, i64>(padding[1]);

      TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
        "max_pool2d: dilation must be either a single int, or a tuple of two ints");
      const int dilationH = safe_downcast<int, i64>(dilation[0]);
      const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, i64>(dilation[1]);

      TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");

      TORCH_CHECK(input.dtype() == gradOutput.dtype(),
        "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

      const auto memory_format = input.suggest_memory_format();
      if (memory_format == MemoryFormat::ChannelsLast) {
        TORCH_CHECK(input.ndimension() == 4,
          "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
      } else if (memory_format == MemoryFormat::Contiguous) {
        TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
          "non-empty 3D or 4D (batch mode) tensor expected for input");
      } else {
        TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
      }

      /* sizes */
      const i64 nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
      const i64 nInputPlane = input.size(-3);
      const i64 inputHeight = input.size(-2);
      const i64 inputWidth = input.size(-1);

      /* XXX preserve the existing shape check behavior */
      const i64 outputHeight_for_shape_check = pooling_output_shape<i64>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
      const i64 outputWidth_for_shape_check = pooling_output_shape<i64>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

      max_pool2d_backward_shape_check(
        input,
        gradOutput,
        indices,
        nbatch,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW,
        nInputPlane,
        inputHeight, inputWidth,
        outputHeight_for_shape_check, outputWidth_for_shape_check,
        memory_format);

      set_output(0, input.sizes(), {}, input.options().memory_format(memory_format),
                 input.has_names() ? input.names() : &[Dimname]{});
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(max_pool2d_with_indices_out_cpu)
    (const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& output,
    const Tensor& indices) {
      NoNamesGuard guard;

      const int kH = safe_downcast<int, i64>(kernel_size[0]);
      const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, i64>(kernel_size[1]);

      const int dH = stride.empty() ? kH : safe_downcast<int, i64>(stride[0]);
      const int dW = stride.empty() ? kW :
                     stride.size() == 1 ? dH : safe_downcast<int, i64>(stride[1]);

      const int padH = safe_downcast<int, i64>(padding[0]);
      const int padW = padding.size() == 1 ? padH : safe_downcast<int, i64>(padding[1]);

      const int dilationH = safe_downcast<int, i64>(dilation[0]);
      const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, i64>(dilation[1]);

      max_pool2d_kernel(
          kCPU, output, indices, input,
          kW, kH,
          dW, dH,
          padW, padH,
          dilationW, dilationH);
    }
    */
}

lazy_static!{
    /*
    TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_cpu)
    (const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    const Tensor& gradInput) {
      NoNamesGuard guard;

      gradInput.zero_();
      max_pool2d_backward_kernel(
          kCPU, const_cast<Tensor&>(gradInput),
          gradOutput, indices);
    }
    */
}

define_dispatch!{max_pool2d_kernel}
define_dispatch!{max_pool2d_backward_kernel}

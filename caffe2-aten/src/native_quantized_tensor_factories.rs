crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/TensorFactories.cpp]

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | We explicitly pass in scale and zero_point
  | because we don't have the infra ready to
  | support quantizer in python frontend, once that
  | is ready, we'll change to use quantizer
  |
  */
pub fn empty_affine_quantized(
        size:                   &[i32],
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        scale:                  f64,
        zero_point:             i64,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.");
      auto options = options_.merge_memory_format(optional_memory_format);
      TORCH_CHECK(
          options.has_dtype(),
          "Must provide data type for Tensor creation functions.");
      return new_qtensor(
          size,
          options,
          make_per_tensor_affine_quantizer(
              scale, zero_point, typeMetaToScalarType(options.dtype())));
        */
}

pub fn empty_per_channel_affine_quantized(
    size:                   &[i32],
    scales:                 &Tensor,
    zero_points:            &Tensor,
    axis:                   i64,
    dtype:                  Option<ScalarType>,
    layout:                 Option<Layout>,
    device:                 Option<Device>,
    pin_memory:             Option<bool>,
    optional_memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.");
      auto options = options_.merge_memory_format(optional_memory_format);
      TORCH_CHECK(
          options.has_dtype(),
          "Must provide data type for Tensor creation functions.");
      QuantizerPtr quantizer = make_per_channel_affine_quantizer(
              scales, zero_points, axis, typeMetaToScalarType(options.dtype()));
      return new_qtensor(
          size,
          options,
          quantizer);
        */
}

/**
  | Provide better error message if dtype
  | is wrong
  |
  */
pub fn empty_affine_quantized_other_backends_stub(
        _0: &[i32],
        _1: Option<ScalarType>,
        _2: Option<Layout>,
        _3: Option<Device>,
        _4: Option<bool>,
        _5: f64,
        _6: i64,
        _7: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
        */
}

pub fn empty_per_channel_affine_quantized_other_backends_stub(
    _0: &[i32],
    _1: &Tensor,
    _2: &Tensor,
    _3: i64,
    _4: Option<ScalarType>,
    _5: Option<Layout>,
    _6: Option<Device>,
    _7: Option<bool>,
    _8: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
        */
}

/**
  | Create an empty quantized Tensor with
  | size, based on the options and quantization
  | parameters of the input quantized Tensor
  |
  */
pub fn empty_quantized(
        size:    &[i32],
        qtensor: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor output;
      if (qtensor.qscheme() == kPerTensorAffine) {
        output = _empty_affine_quantized(size, qtensor.options(),
                                             qtensor.q_scale(),
                                             qtensor.q_zero_point());
      } else if (qtensor.qscheme() == kPerChannelAffine) {
        output = _empty_per_channel_affine_quantized(
            size,
            qtensor.q_per_channel_scales(),
            qtensor.q_per_channel_zero_points(),
            qtensor.q_per_channel_axis(),
            qtensor.options());
      } else {
        TORCH_CHECK(false,
                    "QScheme not supported by empty_quantized:",
                    toString(qtensor.qscheme()));
      }
      return output;
        */
}

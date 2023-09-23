crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ full ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | Performs dtype inference for full
  |
  */
pub fn infer_full_options(
        fill_value: &Scalar,
        options:    &TensorOptions) -> TensorOptions {
    
    todo!();
        /*
            if (!options.has_dtype()) {
        if (fill_value.isBoolean()) {
          return options.dtype(kBool);
        } else if (fill_value.isIntegral(false)) {
          return options.dtype(kLong);
        } else if (fill_value.isComplex()) {
          auto scalar_type = (get_default_dtype() == ScalarType::Double) ?
                                ScalarType::ComplexDouble :
                                ScalarType::ComplexFloat;
          return options.dtype(scalar_type);
        } else {
          return options.dtype(get_default_dtype());
        }
      }

      return options;
        */
}

pub fn full_a(
        size:       &[i32],
        fill_value: &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(options.layout() != kSparse,
        "full(...) is not implemented for sparse layout");

      auto result = empty(size, infer_full_options(fill_value, options));
      return result.fill_(fill_value);
        */
}

pub fn full_out<'a>(
        size:       &[i32],
        fill_value: &Scalar,
        result:     &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!result.is_sparse(),
        "full(...) is not implemented for sparse layout");

      result.resize_(size);
      return result.fill_(fill_value);
        */
}

pub fn full_like(
        self_:                  &Tensor,
        fill_value:             &Scalar,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty_like(self, options, optional_memory_format);
      return result.fill_(fill_value);
        */
}

pub fn new_full(
        self_:      &Tensor,
        size:       &[i32],
        fill_value: &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return full(size, fill_value, self.options().merge_in(options));
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ named tensor overloads ~~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
  | In the short term, these exist.
  |
  | In the long term, we should move &[Dimname]
  | into TensorOptions to avoid having these
  | overloads.
  |
  */
pub fn full_b(
        size:       &[i32],
        fill_value: &Scalar,
        names:      Option<&[Dimname]>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      TORCH_CHECK(options.layout() != kSparse,
        "full(...) is not implemented for sparse layout");

      auto result = empty(size, names, infer_full_options(fill_value, options));
      return result.fill_(fill_value);
        */
}


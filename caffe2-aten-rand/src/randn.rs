crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn randn_a(
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randn(size, static_cast<optional<dyn GeneratorInterface>>(nullopt), dtype, layout, device, pin_memory);
        */
}



pub fn randn_b(
        size:       &[i32],
        generator:  Option<dyn GeneratorInterface>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.normal_(0, 1, generator);
        */
}



pub fn randn_out_a<'a>(
        size:   &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::randn_out(size, nullopt, result);
        */
}

pub fn randn_out_b<'a>(
        size:      &[i32],
        generator: Option<dyn GeneratorInterface>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.normal_(0, 1, generator);
        */
}

pub fn randn_like(
    self_:                  &Tensor,
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
      return result.normal_(0, 1, nullopt);
        */
}

pub fn randn_c(
        size:       &[i32],
        names:      Option<&[Dimname]>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randn(size, nullopt, names, dtype, layout, device, pin_memory);
        */
}

pub fn randn_d(
        size:       &[i32],
        generator:  Option<dyn GeneratorInterface>,
        names:      Option<&[Dimname]>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, names, options);
      return result.normal_(0, 1, generator);
        */
}

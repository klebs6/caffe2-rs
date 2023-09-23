crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn randint_a(
        high:       i64,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(high, size, nullopt /* generator*/, dtype, layout, device, pin_memory);
        */
}



pub fn randint_b(
        high:       i64,
        size:       &[i32],
        generator:  Option<dyn GeneratorInterface>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(0, high, size, generator, dtype, layout, device, pin_memory);
        */
}


pub fn randint_c(
        low:        i64,
        high:       i64,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::randint(low, high, size, nullopt, dtype, layout, device, pin_memory);
        */
}

pub fn randint_d(
        low:        i64,
        high:       i64,
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
      return result.random_(low, high, generator);
        */
}



pub fn randint_out_a<'a>(
        high:   i64,
        size:   &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::randint_out(high, size, nullopt, result);
        */
}


pub fn randint_out_b<'a>(
        high:      i64,
        size:      &[i32],
        generator: Option<dyn GeneratorInterface>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.random_(0, high, generator);
        */
}


pub fn randint_out_c<'a>(
        low:    i64,
        high:   i64,
        size:   &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::randint_out(low, high, size, nullopt, result);
        */
}

pub fn randint_out_d<'a>(
        low:       i64,
        high:      i64,
        size:      &[i32],
        generator: Option<dyn GeneratorInterface>,
        result:    &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            result.resize_(size);
      return result.random_(low, high, generator);
        */
}


pub fn randint_like_a(
        self_:                  &Tensor,
        high:                   i64,
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
      return result.random_(0, high, nullopt);
        */
}

pub fn randint_like_b(
        self_:                  &Tensor,
        low:                    i64,
        high:                   i64,
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
      return result.random_(low, high, nullopt);
        */
}



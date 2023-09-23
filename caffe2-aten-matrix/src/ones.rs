crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn ones_a(
    size:       &[i32],
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
        /*
            return native::full(size, /*fill_value=*/1., dtype, layout, device, pin_memory);
        */
}

pub fn ones_out<'a>(
        size:   &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            return native::full_out(size, /*fill_value=*/1., result);
        */
}

pub fn ones_like(
        self_:                  &Tensor,
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto result = empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
      return result.fill_(1.);
        */
}

pub fn new_ones(
        self_:      &Tensor,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options =
          TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
              pin_memory);

      return ones(size, self.options().merge_in(options));
        */
}

pub fn ones_b(
        size:       &[i32],
        names:      Option<&[Dimname]>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]

      return native::full(
          size, /*fill_value=*/1., names, dtype, layout, device, pin_memory);
        */
}

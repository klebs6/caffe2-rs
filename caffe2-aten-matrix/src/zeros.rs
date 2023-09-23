crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ zeros ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn zeros_a(
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.zero_();
        */
}

pub fn zeros_out<'a>(
        size:   &[i32],
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            if (result.is_sparse()) {
        result.sparse_resize_and_clear_(size, size.size(), 0.);
        return result;
      } else {
        result.resize_(size);
      }
      return result.zero_();
        */
}

pub fn zeros_like(
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

      if (options.layout() == kSparse && self.is_sparse()) {
        TORCH_CHECK(
            !(optional_memory_format.has_value()),
            "memory format option is only supported by strided tensors");
        auto res = empty({0}, options); // to be resized
        res.sparse_resize_and_clear_(
            self.sizes(), self.sparse_dim(), self.dense_dim());
        return res;
      }
      auto result = empty_like(self, options, optional_memory_format);
      return result.zero_();
        */
}

pub fn new_zeros(
        self_:      &Tensor,
        size:       &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      return zeros(size, self.options().merge_in(options));
        */
}

pub fn zeros_b(
        size:       &[i32],
        names:      Option<&[Dimname]>,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::full(size, /*fill_value=*/0., names, dtype, layout, device, pin_memory);
        */
}



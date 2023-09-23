crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn range_a(
        start:      &Scalar,
        end:        &Scalar,
        step:       &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      Tensor result = empty({0}, options);
      return range_out(result, start, end, step);
        */
}

pub fn range_b(
        start:      &Scalar,
        end:        &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::range(start, end, 1, dtype, layout, device, pin_memory);
        */
}



crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn hann_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hann_window(window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn hann_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("hann_window", options, window_length);
      return native::hamming_window(
          window_length, periodic, /*alpha=*/0.5, /*beta=*/0.5, dtype, layout, device, pin_memory);
        */
}

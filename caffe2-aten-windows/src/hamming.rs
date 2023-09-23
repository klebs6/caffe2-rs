crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn hamming_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn hamming_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length,
          periodic,
          /*alpha=*/0.54,
          dtype,
          layout,
          device,
          pin_memory);
        */
}

pub fn hamming_window_c(
        window_length: i64,
        periodic:      bool,
        alpha:         f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::hamming_window(
          window_length, periodic, alpha, /*beta=*/0.46, dtype, layout, device, pin_memory);
        */
}

pub fn hamming_window_d(
        window_length: i64,
        periodic:      bool,
        alpha:         f64,
        beta:          f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("hamming_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      auto window = native::arange(window_length, dtype, layout, device, pin_memory);
      window.mul_(pi<double> * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

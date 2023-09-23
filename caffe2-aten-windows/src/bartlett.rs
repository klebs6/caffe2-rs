crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn bartlett_window_a(
    window_length: i64,
    dtype:         Option<ScalarType>,
    layout:        Option<Layout>,
    device:        Option<Device>,
    pin_memory:    Option<bool>

) -> Tensor {
    
    todo!();
        /*
            return native::bartlett_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn bartlett_window_b(
    window_length: i64,
    periodic:      bool,
    dtype:         Option<ScalarType>,
    layout:        Option<Layout>,
    device:        Option<Device>,
    pin_memory:    Option<bool>

) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("bartlett_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      auto window = native::arange(window_length, dtype, layout, device, pin_memory)
                        .mul_(2. / static_cast<double>(window_length - 1));
      const i64 first_half_size = ((window_length - 1) >> 1) + 1;
      window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

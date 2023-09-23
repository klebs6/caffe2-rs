crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn blackman_window_a(
    window_length: i64,
    dtype:         Option<ScalarType>,
    layout:        Option<Layout>,
    device:        Option<Device>,
    pin_memory:    Option<bool>

) -> Tensor {

    todo!();
        /*
            return native::blackman_window(
          window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
        */
}

pub fn blackman_window_b(
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

      window_function_checks("blackman_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return native::ones({1}, dtype, layout, device, pin_memory);
      }
      if (periodic) {
        window_length += 1;
      }
      // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
      auto window =
          native::arange(window_length, dtype, layout, device, pin_memory)
              .mul_(pi<double> / static_cast<double>(window_length - 1));
      window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

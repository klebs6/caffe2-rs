crate::ix!();

define_dispatch!{kaiser_window_stub}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kaiser_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn kaiser_window_a(
        window_length: i64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::kaiser_window(
          window_length,
          /*periodic=*/true,
          /*beta=*/12.0,
          dtype,
          layout,
          device,
          pin_memory);
        */
}

pub fn kaiser_window_b(
        window_length: i64,
        periodic:      bool,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            return native::kaiser_window(window_length, periodic, /*beta=*/12.0, dtype, layout, device, pin_memory);
        */
}

pub fn kaiser_window_c(
        window_length: i64,
        periodic:      bool,
        beta:          f64,
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      window_function_checks("kaiser_window", options, window_length);
      if (window_length == 0) {
        return empty({0}, options);
      }
      if (window_length == 1) {
        return ones({1}, options);
      }
      if (periodic) {
        window_length += 1;
      }
      auto initial = arange(window_length, options);
      auto window = empty(window_length, options);
      auto iter = TensorIterator::unary_op(window, initial);
      kaiser_window_stub(iter.device_type(), iter, window_length, beta);
      return periodic ? window.narrow(0, 0, window_length - 1) : window;
        */
}

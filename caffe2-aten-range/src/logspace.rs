crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn logspace(
    start:      &Scalar,
    end:        &Scalar,
    steps:      Option<i64>,
    base:       f64,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {

    todo!();
    /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      const auto steps_ = steps.value_or(100);
      TORCH_CHECK(steps_ >= 0, "number of steps must be non-negative");
      auto result_options = linspace_logspace_infer_options(start, end, options, "torch.logspace()");
      Tensor result = empty({steps_}, result_options);
      return logspace_out(result, start, end, steps, base);
        */
}




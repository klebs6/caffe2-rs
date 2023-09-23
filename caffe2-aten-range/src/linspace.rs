crate::ix!();

pub fn linspace_logspace_infer_options(
    start:   &Scalar,
    end:     &Scalar,
    options: &TensorOptions,
    fn_name: *const u8) -> TensorOptions {
    
    todo!();
        /*
            if (start.isComplex() || end.isComplex()) {
        const auto default_complex_dtype = get_default_complex_dtype();
        if (options.has_dtype()) {
          auto dtype = typeMetaToScalarType(options.dtype());
          TORCH_CHECK(isComplexType(dtype),
              fn_name, ": inferred dtype ", default_complex_dtype, " can't be safely cast to passed dtype ", dtype);
        } else {
          return options.dtype(default_complex_dtype);
        }
      }

      return options.has_dtype() ? options : options.dtype(get_default_dtype());
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn linspace(
        start:      &Scalar,
        end:        &Scalar,
        steps:      Option<i64>,
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
      auto result_options = linspace_logspace_infer_options(start, end, options, "torch.linspace()");
      Tensor result = empty({steps_}, result_options);
      return linspace_out(result, start, end, steps);
        */
}



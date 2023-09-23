crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn scalar_tensor(
        s:          &Scalar,
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      if (options.device() == kCPU) {
        // This is a fast track to skip device dispatch for making scalar tensor on CPU.
        // See https://github.com/pytorch/pytorch/pull/29915 for more detailed perf
        // difference.
        // In the future when we remove the overhead of device dispatch, we'll happily
        // revert this to following:
        //   auto result = empty({}, options);
        tracer::NoTracerDispatchMode tracer_guard;
        AutoDispatchBelowAutograd mode;
        auto result = empty_cpu({}, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
        native::fill_(result, s);
        return result;
      }
      return empty({}, options).fill_(s);
        */
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn tensor_cpu<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_cpu(values, options);
        */
}


pub fn tensor_backend<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_backend(values, options);
        */
}

pub fn tensor_complex_cpu<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_complex_cpu(values, options);
        */
}


pub fn tensor_complex_backend<T>(
    values:  &[T],
    options: &TensorOptions) -> Tensor {

    todo!();
        /*
            return tensor_complex_backend(values, options);
        */
}

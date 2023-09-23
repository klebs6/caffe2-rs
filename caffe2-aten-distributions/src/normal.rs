crate::ix!();

pub fn normal(
    mean:       f64,
    std:        f64,
    size:       &[i32],
    generator:  Option<dyn GeneratorInterface>,
    dtype:      Option<ScalarType>,
    layout:     Option<Layout>,
    device:     Option<Device>,
    pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      auto result = empty(size, options);
      return result.normal_(mean, std, generator);
        */
}

pub fn normal_out<'a>(
    mean:      f64,
    std:       f64,
    size:      &[i32],
    generator: Option<dyn GeneratorInterface>,
    result:    &mut Tensor) -> &'a mut Tensor {

    todo!();
        /*
            result.resize_(size);
      return result.normal_(mean, std, generator);
        */
}

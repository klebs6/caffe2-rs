crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Factory.cpp]

pub fn empty_memory_format(
        sizes:         &[i32],
        dtype:         Option<ScalarType>,
        layout:        Option<Layout>,
        device:        Option<Device>,
        pin_memory:    Option<bool>,
        memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            return convert(vTensor{
          api::context(),
          sizes,
          TensorOptions()
              .dtype(dtype)
              .layout(layout)
              .device(device)
              .pinned_memory(pin_memory)
              .memory_format(memory_format),
        });
        */
}

pub fn empty_strided(
        sizes:      &[i32],
        strides:    &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return empty_memory_format(
          sizes,
          dtype,
          layout,
          device,
          pin_memory,
          MemoryFormat::Contiguous);
        */
}

#[cfg(USE_VULKAN_API)]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
      m.impl("empty.memory_format", native::vulkan::ops::empty_memory_format);
      m.impl("empty_strided", TORCH_FN(native::vulkan::ops::empty_strided));
    }
    */
}

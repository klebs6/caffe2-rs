crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/TensorFactories.cpp]

#[cfg(AT_MKLDNN_ENABLED)]
pub fn empty_mkldnn(
        sizes:                  &[i32],
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
         !optional_memory_format.has_value(),
         "'memory_format' argument is incompatible with mkldnn tensor");
      // NOTE: i32 dims from ideep::tensor but sizes needs i64
      // TODO: support i64 dims in ideep::tensor to avoid extra conversion
      ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
      auto data_type = dtype.has_value() ? get_mkldnn_dtype(dtype.value()) : ideep::tensor::data_type::f32;
      ideep::tensor it {dst_dims, data_type};
      return new_with_itensor_mkldnn(move(it), dtype, device);
        */
}

#[cfg(not(AT_MKLDNN_ENABLED))]
pub fn empty_mkldnn(
        sizes:                  &[i32],
        dtype:                  Option<ScalarType>,
        layout:                 Option<Layout>,
        device:                 Option<Device>,
        pin_memory:             Option<bool>,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(false, "empty_mkldnn: MKL-DNN build is disabled");
        */
}

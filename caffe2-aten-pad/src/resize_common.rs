crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ResizeCommon.h]

#[inline] pub fn storage_size_for(
    size:   &[i32],
    stride: &[i32]) -> i64 {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(size.size() == stride.size(),
          "storage_size_for(size, stride) requires that size and stride ",
          "have the same size as a precondition.");
      i64 storage_size = 1;
      for (usize dim = 0; dim < size.size(); ++dim) {
        if (size[dim] == 0) {
          storage_size = 0;
          break;
        }
        storage_size += (size[dim] - 1) * stride[dim];
      }
      return storage_size;
        */
}

#[inline] pub fn resize_named_tensor(
    self_:                  &Tensor,
    size:                   &[i32],
    optional_memory_format: Option<MemoryFormat>) -> &Tensor {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT(self.has_names());
      TORCH_CHECK(
          self.sizes() == size,
          "Cannot resize named tensor with resize_ or resize_as_ (tried to resize "
          "Tensor",
          self.names(),
          " with size ",
          self.sizes(),
          " to ",
          size,
          "). This may be caused by passing a named tensor ",
          "as an `out=` argument; please ensure that the sizes are the same. ");
      TORCH_CHECK(
          !optional_memory_format.has_value(),
          "Unsupported memory format for named tensor resize ",
          optional_memory_format.value());
      return self;
        */
}

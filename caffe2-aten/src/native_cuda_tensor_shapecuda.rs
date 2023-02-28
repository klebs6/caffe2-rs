crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/TensorShapeCUDA.cpp]

/**
  | this needs to be split along CPU/CUDA lines
  | because we don't have a consistent way of
  | getting the allocator to use for a device
  | (GetAllocator is not the same as
  | getCUDADeviceAllocator().
  */
pub fn set_cuda(result: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TypeMeta dtype = result.dtype();
      Storage storage(
          Storage::use_byte_size_t(),
          0,
          getCUDADeviceAllocator(),
          true);
      result.set_(storage, 0, {0}, {});
      TORCH_INTERNAL_ASSERT(dtype == result.dtype());
      return result;
        */
}

/**
  | unify with cuda implementation? This
  | is not done to avoid a dispatch in resize_impl_cpu_
  |
  */
pub fn set_storage_cuda(
        result:         &mut Tensor,
        storage:        Storage,
        storage_offset: i64,
        size:           &[i32],
        stride:         &[i32]) -> &mut Tensor {
    
    todo!();
        /*
            checkSetStorage(result, storage, storage_offset, size, stride);

      result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
      optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                              optional<IntArrayRef>(stride) : nullopt;
      native::resize_impl_cuda_(result.unsafeGetTensorImpl(), size, stride_opt);
      return result;
        */
}

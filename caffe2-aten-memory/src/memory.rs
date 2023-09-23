crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Memory.cpp]

pub fn is_pinned(self_: &Tensor) -> bool {
    
    todo!();
        /*
            return getCUDAHooks().isPinnedPtr(self.storage().data());
        */
}

pub fn pin_memory(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            if (!self.device().is_cpu()) {
        AT_ERROR("cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
      }
      if (self.is_pinned()) {
        return self;
      }
      auto* allocator = getCUDAHooks().getPinnedMemoryAllocator();
      auto storage = Storage(
          Storage::use_byte_size_t(),
          computeStorageNbytes(
              self.sizes(), self.strides(), self.dtype().itemsize()),
          allocator,
          /*resizable=*/false);
      auto tensor = empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
      tensor.copy_(self);
      return tensor;
        */
}

/**
  | Exposes has_internal_overlap as an
  | operator for testing purposes
  |
  */
pub fn debug_has_internal_overlap(self_: &Tensor) -> i64 {
    
    todo!();
        /*
            return static_cast<i64>(has_internal_overlap(self));
        */
}

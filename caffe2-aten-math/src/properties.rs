crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorProperties.cpp]

pub fn is_same_size(
        self_: &Tensor,
        other: &Tensor) -> bool {
    
    todo!();
        /*
            return self.sizes().equals(other.sizes());
        */
}

pub fn size_a(
        self_: &Tensor,
        dim:   i64) -> i64 {
    
    todo!();
        /*
            return self.size(dim);
        */
}

pub fn stride_a(
        self_: &Tensor,
        dim:   i64) -> i64 {
    
    todo!();
        /*
            return self.stride(dim);
        */
}

pub fn size_b(
        self_: &Tensor,
        dim:   Dimname) -> i64 {
    
    todo!();
        /*
            usize pos_dim = dimname_to_position(self, dim);
      return self.sizes()[pos_dim];
        */
}

pub fn stride_b(
        self_: &Tensor,
        dim:   Dimname) -> i64 {
    
    todo!();
        /*
            usize pos_dim = dimname_to_position(self, dim);
      return self.strides()[pos_dim];
        */
}

pub fn cudnn_is_acceptable<'a>(self_: &Tensor) -> bool {
    
    todo!();
        /*
            if (!globalContext().userEnabledCuDNN()) return false;
      if (!self.is_cuda()) return false;
      auto st = self.scalar_type();
      if (!(st == kDouble || st == kFloat || st == kHalf)) return false;
      if (!getCUDAHooks().compiledWithCuDNN()) return false;
      // cuDNN functions like grid_sampler returns CUDNN_STATUS_BAD_PARAM on empty
      // tensors. Maybe some cuDNN functions actually support empty tensors, but
      // native/THNN kernels shouldn't be much slower because the output is also
      // likely empty.
      if (self.numel() == 0) return false;
      // NB: In the old Python code, there was also a test to see if the
      // cuDNN library was actually dynamically linked or not.  I'm not
      // sure if we can actually test this.
      return true;
        */
}


pub fn detach<'a>(self_: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // this just exists to give us a hook in VariableType and an entry in Declarations.yaml
      //AT_ERROR("detach_ is not implemented for Tensor");
      return self;
        */
}

pub fn contiguous_a(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return contiguous(self, MemoryFormat::Contiguous);
        */
}

pub fn contiguous_b(
        self_:         &Tensor,
        memory_format: MemoryFormat) -> Tensor {
    
    todo!();
        /*
            if (self.is_contiguous(memory_format)) {
        return self;
      }
      TORCH_CHECK(
          memory_format != MemoryFormat::Preserve,
          "preserve memory format is unsupported by the contiguous operator");

      auto result = empty_like(self, self.options(), memory_format);
      return result.copy_(self);
        */
}


pub fn is_set_to(
        self_: &Tensor,
        src:   &Tensor) -> bool {
    
    todo!();
        /*
            if (self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() &&
          self.storage_offset() == src.storage_offset() &&
          self.dim() == src.dim()) {
        for (i64 d = 0; d < self.dim(); ++d) {
          if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
            return false;
          }
        }
        return true;
      }
      return false;
        */
}

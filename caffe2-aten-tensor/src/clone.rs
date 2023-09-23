crate::ix!();

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn clone(
        src:                    &Tensor,
        optional_memory_format: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto memory_format =
          optional_memory_format.value_or(MemoryFormat::Preserve);
      if (memory_format == MemoryFormat::Preserve) {
        if (src.is_non_overlapping_and_dense()) {
          // Copy all strides
          auto self = empty_strided(src.sizes(), src.strides(), src.options());
          self.copy_(src);
          return self;
        } else {
          memory_format = src.suggest_memory_format();
        }
      }
      auto self = empty_like(src, src.options(), memory_format);
      self.copy_(src);
      return self;
        */
}

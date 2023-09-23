crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/utils/Factory.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/utils/Factory.cpp]

/**
  | TODO: Remove this function when native::empty()
  | is modified to accept a custom memory
  | allocator.
  |
  */
pub fn mobile_empty_with_tail_padding(
        size:          &[i32],
        dtype:         TypeMeta,
        memory_format: MemoryFormat,
        maybe_names:   &[Dimname]) -> Tensor {
    
    todo!();
        /*
            auto* const allocator_ptr = GetDefaultMobileCPUAllocator();
      const i64 nelements = multiply_integers(size);
      usize size_bytes = nelements * dtype.itemsize();

      Tensor tensor(make_intrusive<TensorImpl>(
          Storage{
              Storage::use_byte_Size(),
              size_bytes,
              allocator_ptr->allocate(size_bytes),
              allocator_ptr,
              /*resizable=*/true,
          },
          DispatchKeySet{DispatchKey::CPU},
          dtype));

      return namedinference::propagate_names_if_nonempty(
          tensor.resize_(size, memory_format),
          maybe_names);
        */
}

pub fn mobile_allocate_padded_contiguous_if_needed(
        input:         &Tensor,
        memory_format: MemoryFormat) -> Tensor {
    
    todo!();
        /*
            const auto* const allocator = input.storage().allocator();
      const auto* const mobile_allocator = GetDefaultMobileCPUAllocator();

      // If the allocators are the same and the memory is contiguous in the requested
      // format, then there is no need to reallocate the tensor.

      if ((allocator == mobile_allocator) && input.is_contiguous(memory_format)) {
        return input;
      }

      // If there is a need to reallocate the tensor on the other hand, either because
      // the allocators are not the same, or the allocators are the same but the input
      // is not contiguous in the requested format, then reallocate and directly copy
      // into destination.  There is no need to allocate a temporary contiguous memory
      // only to use it as the source of the copy operation onto our final destination.

      Tensor padded_input = empty_with_tail_padding(
          input.sizes(),
          input.options().dtype(),
          memory_format,
          input.names());

      return padded_input.copy_(input);
        */
}

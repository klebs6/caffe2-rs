crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/PinnedMemoryAllocator.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/PinnedMemoryAllocator.cpp]

pub fn get_pinned_memory_allocator() -> *mut Allocator {
    
    todo!();
        /*
            auto state = globalContext().lazyInitCUDA();
      return state->cudaHostAllocator;
        */
}

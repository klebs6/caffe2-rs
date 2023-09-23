crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Allocator.h]

/**
  | Do NOT include vk_mem_alloc.h directly.
  |
  | Always include this file (Allocator.h) instead.
  |
  */
#[cfg(USE_VULKAN_API)]
mod vulkan_api {
    use super::*;

    /*
       | Let this, then, be our first principle: — That
       | the citizen who does not know how to choose
       | between good and evil must not have authority,
       | although he possess great mental gifts, and
       | many accomplishments; for he is really a fool. 
       |
       | On the other hand, he who has this knowledge
       | may be unable either to read or swim;
       | nevertheless, he shall be counted wise and
       | permitted to rule. For how can there be wisdom
       | where there is no harmony? — the wise man is
       | the saviour, and he who is devoid of wisdom is
       | the destroyer of states and households.
       |
       | - Plato, Laws
       */
    pub const VMA_VULKAN_VERSION: usize = 1000000;

    #[cfg(USE_VULKAN_WRAPPER)]
    pub const VMA_STATIC_VULKAN_FUNCTIONS: usize = 0;

    #[cfg(not(USE_VULKAN_WRAPPER))]
    pub const VMA_DYNAMIC_VULKAN_FUNCTIONS: usize = 0;

    pub const VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE: usize = 64 * 1024 * 1024;
    pub const VMA_SMALL_HEAP_MAX_SIZE:           usize = 256 * 1024 * 1024;

    #[cfg(debug_assertions)]
    mod debug {

        pub const VMA_DEBUG_ALIGNMENT:                    usize = 4096;
        pub const VMA_DEBUG_ALWAYS_DEDICATED_MEMORY:      usize = 0;
        pub const VMA_DEBUG_DETECT_CORRUPTION:            usize = 1;
        pub const VMA_DEBUG_GLOBAL_MUTEX:                 usize = 1;
        pub const VMA_DEBUG_INITIALIZE_ALLOCATIONS:       usize = 1;
        pub const VMA_DEBUG_MARGIN:                       usize = 64;
        pub const VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY: usize = 256;
        pub const VMA_RECORDING_ENABLED:                  usize = 1;

        macro_rules! vma_debug_log {
            ($format:ident, $($arg:ident),*) => {
                /*

                   do {                              
                   printf(format, ##__VA_ARGS__);  
                   printf("\n");                   
                   } while (false)
                   */
            }
        }
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/api/Allocator.cpp]
pub const VMA_IMPLEMENTATION: bool = true;

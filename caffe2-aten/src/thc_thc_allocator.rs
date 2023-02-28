crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCAllocator.h]

/// IPC doesn't support (re)allocation
///
pub struct THCIpcDeleter {
    base_ptr: Arc<c_void>,
}

impl THCIpcDeleter {

    pub fn new(base_ptr: Arc<c_void>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn make_data_ptr(
        base_ptr: Arc<c_void>,
        data:     *mut c_void) -> DataPtr {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCAllocator.cpp]

pub fn delete_thc_ipc_deleter(ptr: *mut c_void)  {
    
    todo!();
        /*
            delete static_cast<THCIpcDeleter*>(ptr);
        */
}

impl THCIpcDeleter {
    
    /**
      | Refer to Note [CUDA IPC and the caching
      | allocator] for more details
      |
      | basePtr - device ptr of a single cudaMalloc
      |           allocation; this may be a large block
      |           of memory which is managed by the
      |           caching allocator.
      |
      | data    - ptr to where the storage (of a single
      | type) should start.
      |
      | Invariant: data must lie within the CUDA memory
      |   allocation represented by basePtr.
      |
      | Here basePtr should be saved in the struct,
      | while data should be used to construct the new
      | storage.
      |
      | Every time a storage referring to the IPC
      | memory region goes out of scope, the reference
      | count on the memory region will be decreased by
      | one, until it's zero, at which point IPC memory
      | region is closed (by calling
      | cudaIpcCloseMemHandle).
      |
      */
    pub fn make_data_ptr(&mut self, 
        base_ptr: Arc<c_void>,
        data:     *mut c_void) -> DataPtr {
        
        todo!();
        /*
      // The dynamic allocation here is a bit unfortunate
      int cur_device;
      THCudaCheck(cudaGetDevice(&cur_device));
      auto* context = new THCIpcDeleter(move(basePtr));
      return {data, context, &deleteTHCIpcDeleter, Device(DeviceType_CUDA, cur_device)};
        */
    }
    
    pub fn new(base_ptr: Arc<c_void>) -> Self {
    
        todo!();
        /*
        : base_ptr(move(basePtr)),
        */
    }
}

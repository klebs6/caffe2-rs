//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.cpp]

pub fn get() -> *mut Allocator {
    
    todo!();
        /*
            static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());
      return &allocator;
        */
}

pub fn record_stream_masquerading_ascuda(
        ptr:    &DataPtr,
        stream: HIPStreamMasqueradingAsCUDA)  {
    
    todo!();
        /*
            HIPCachingAllocator::recordStream(ptr, stream.hip_stream());
        */
}


/*!
  | Use of hip namespace here makes hipification
  | easier, because
  |
  | I don't have to also fix namespaces.  Sorry!
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h]

/**
  | Takes a valid HIPAllocator (of any sort) and
  | turns it into an allocator pretending to be
  | a CUDA allocator.  See Note [Masquerading as
  | CUDA]
  */
pub struct HIPAllocatorMasqueradingAsCUDA {
    base:      Allocator,
    allocator: *mut Allocator,
}

impl HIPAllocatorMasqueradingAsCUDA {
    
    pub fn new(allocator: *mut Allocator) -> Self {
    
        todo!();
        /*
        : allocator(allocator),

        
        */
    }
    
    pub fn allocate(&self, size: usize) -> DataPtr {
        
        todo!();
        /*
            DataPtr r = allocator_->allocate(size);
        r.unsafe_set_device(Device(DeviceType::Cuda, r.device().index()));
        return r;
        */
    }
    
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return allocator_->raw_deleter();
        */
    }
}

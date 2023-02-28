crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/StorageImpl.h]

pub struct UseByteSize {}

/**
  | A storage represents the underlying backing
  | data buffer for a tensor.
  |
  | This concept was inherited from the original
  | Torch7 codebase; we'd kind of like to get rid
  | of the concept (see
  | https://github.com/pytorch/pytorch/issues/14797)
  | but it's hard work and no one has gotten around
  | to doing it.
  |
  | NB: storage is supposed to uniquely own a data
  | pointer; e.g., two non-null data pointers alias
  | if and only if they are from the same storage.
  |
  | Technically you can violate this invariant
  | (e.g., you can create a non-owning StorageImpl
  | with from_blob) but a lot of things won't work
  | correctly, including:
  |
  | - An ordinary deleter on such a storage is
  |   wrong, because normal deleters assume unique
  |   ownership, but if you have two storages at
  |   the same data, that implies there is some
  |   sort of shared ownership. So your deleter
  |   would have to actually be internally doing
  |   some sort of refcount thing
  |
  | - Deepcopy in Python side relies on storage
  |   equality and not data pointer equality; so if
  |   there are two separate storages pointing to
  |   the same data, the data will actually get
  |   duplicated in that case (one data ptr before,
  |   two data ptrs after)
  |
  | - Version counts won't work correctly, because
  |   we do all VC tracking at the level of
  |   storages (unless you explicitly disconnect
  |   the VC with detach); mutation because data
  |   pointers are the same are totally untracked
  |
  */
pub struct StorageImpl {

    link:          LinkedListLink,

    data_ptr:      DataPtr,
    size_bytes:    usize,
    resizable:     bool,

    /**
      | Identifies that Storage was received
      | from another process and doesn't have
      | local to process cuda memory allocation
      |
      */
    received_cuda: bool,

    allocator:     *mut Allocator,
}

intrusive_adapter!(pub StorageImplAdapter = Box<StorageImpl>: StorageImpl { link: LinkedListLink });

//-------------------------------------------[.cpp/pytorch/c10/core/StorageImpl.cpp]
impl StorageImpl {

    pub fn new_with_data_ptr(
        use_byte_size: UseByteSize,
        size_bytes:    usize,
        data_ptr:      DataPtr,
        allocator:     *mut Allocator,
        resizable:     bool) -> Self {
    
        todo!();
        /*


            : data_ptr_(move(data_ptr)),
            size_bytes_(size_bytes),
            resizable_(resizable),
            received_cuda_(false),
            allocator_(allocator) 

        if (resizable) {
          TORCH_INTERNAL_ASSERT(
              allocator_, "For resizable storage, allocator must be provided");
        }
        */
    }
    
    pub fn new(
        use_byte_size: UseByteSize,
        size_bytes:    usize,
        allocator:     *mut Allocator,
        resizable:     bool) -> Self {
    
        todo!();
        /*


            : StorageImpl(
                use_byte_size_t(),
                size_bytes,
                allocator->allocate(size_bytes),
                allocator,
                resizable)
        */
    }
    
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            data_ptr_.clear();
        size_bytes_ = 0;
        */
    }
    
    
    #[inline] pub fn data_generic<T>(&self) -> *mut T {
    
        todo!();
        /*
            return unsafe_data<T>();
        */
    }
    
    
    #[inline] pub fn unsafe_data<T>(&self) -> *mut T {
    
        todo!();
        /*
            return static_cast<T*>(this->data_ptr_.get());
        */
    }
    
    pub fn release_resources(&mut self)  {
        
        todo!();
        /*
            data_ptr_.clear();
        */
    }
    
    pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
            return size_bytes_;
        */
    }

    /// TODO: remove later
    pub fn set_nbytes(&mut self, size_bytes: usize)  {
        
        todo!();
        /*
            size_bytes_ = size_bytes;
        */
    }
    
    
    pub fn resizable(&self) -> bool {
        
        todo!();
        /*
            return resizable_;
        */
    }
    
    pub fn data_ptr_mut(&mut self) -> &mut DataPtr {
        
        todo!();
        /*
            return data_ptr_;
        */
    }
    
    pub fn data_ptr(&self) -> &DataPtr {
        
        todo!();
        /*
            return data_ptr_;
        */
    }

    /// Returns the previous data_ptr
    pub fn set_data_ptr(&mut self, data_ptr: DataPtr) -> DataPtr {
        
        todo!();
        /*
            swap(data_ptr_, data_ptr);
        return move(data_ptr);
        */
    }
    
    pub fn set_data_ptr_noswap(&mut self, data_ptr: DataPtr)  {
        
        todo!();
        /*
            data_ptr_ = move(data_ptr);
        */
    }

    /// TODO: Return const ptr eventually if possible
    pub fn data_mut(&mut self)  {
        
        todo!();
        /*
            return data_ptr_.get();
        */
    }
    
    pub fn data(&self)  {
        
        todo!();
        /*
            return data_ptr_.get();
        */
    }
    
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return data_ptr_.device().type();
        */
    }
    
    pub fn allocator_mut(&mut self) -> *mut Allocator {
        
        todo!();
        /*
            return allocator_;
        */
    }
    
    pub fn allocator(&self) -> *const Allocator {
        
        todo!();
        /*
            return allocator_;
        */
    }

    /**
      | You generally shouldn't use this method, but
      | it is occasionally useful if you want to
      | override how a tensor will be reallocated,
      | after it was already allocated (and its
      | initial allocator was set)
      |
      */
    pub fn set_allocator(&mut self, allocator: *mut Allocator)  {
        
        todo!();
        /*
            allocator_ = allocator;
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return data_ptr_.device();
        */
    }
    
    pub fn set_resizable(&mut self, resizable: bool)  {
        
        todo!();
        /*
            if (resizable) {
          // We need an allocator to be resizable
          AT_ASSERT(allocator_);
        }
        resizable_ = resizable;
        */
    }

    /**
      | Can only be called when use_count is
      | 1
      |
      */
    pub fn unique_storage_share_external_pointer_with_deleter<A: std::alloc::Allocator>(&mut self, 
        src:        *mut c_void,
        size_bytes: usize,
        d:          Option<A>)  {

        todo!();
        /*
            UniqueStorageShareExternalPointer(
            DataPtr(src, src, d, data_ptr_.device()), size_bytes);
        */
    }

    /**
      | Can only be called when use_count is
      | 1
      |
      */
    pub fn unique_storage_share_external_pointer(&mut self, 
        data_ptr:   DataPtr,
        size_bytes: usize)  {
        
        todo!();
        /*
            data_ptr_ = move(data_ptr);
        size_bytes_ = size_bytes;
        allocator_ = nullptr;
        resizable_ = false;
        */
    }

    /**
      | This method can be used only after storage
      | construction and cannot be used to modify
      | storage status
      |
      */
    pub fn set_received_cuda(&mut self, received_cuda: bool)  {
        
        todo!();
        /*
            received_cuda_ = received_cuda;
        */
    }
    
    pub fn received_cuda(&mut self) -> bool {
        
        todo!();
        /*
            return received_cuda_;
        */
    }
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/Storage.h]

pub struct UseByteSize {}

pub struct Storage {
    storage_impl: StorageImplAdapter,
}

impl Storage {
    
    pub fn new_from_adapter(ptr: StorageImplAdapter) -> Self {
    
        todo!();
        /*


            : storage_impl_(move(ptr))
        */
    }

    /**
      | Allocates memory buffer using given
      | allocator and creates a storage with
      | it
      |
      */
    pub fn new(
        use_byte_size: UseByteSize,
        size_bytes:    usize,
        allocator:     Option<*mut Allocator>,
        resizable:     Option<bool>) -> Self {

        let resizable: bool = resizable.unwrap_or(false);

        todo!();
        /*


            : storage_impl_(make_intrusive<StorageImpl>(
                StorageImpl::use_byte_size_t(),
                size_bytes,
                allocator,
                resizable))
        */
    }

    /**
      | Creates storage with pre-allocated memory
      | buffer. Allocator is given for potential future
      | reallocations, however it can be nullptr if
      | the storage is non-resizable
      |
      */
    pub fn new_with_dataptr(
        use_byte_size: UseByteSize,
        size_bytes:    usize,
        data_ptr:      DataPtr,
        allocator:     Option<*mut Allocator>,
        resizable:     Option<bool>) -> Self {

        let resizable: bool = resizable.unwrap_or(false);

        todo!();
        /*
            : storage_impl_(make_intrusive<StorageImpl>(
                StorageImpl::use_byte_size_t(),
                size_bytes,
                move(data_ptr),
                allocator,
                resizable))
        */
    }

    /**
      | Legacy constructor for partially initialized
      | (dtype or memory) storages that can be
      | temporarily created with Caffe2 APIs. See
      | the note on top of TensorImpl.h for
      | details.
      */
    pub fn create_legacy(device: Device) -> Storage {
        
        todo!();
        /*
            auto allocator = GetAllocator(device.type());
        return Storage(make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            0,
            allocator->allocate(0), // materialize a non-default Device.
            allocator,
            true));
        */
    }
    
    pub fn data_t<T>(&self) -> *mut T {
    
        todo!();
        /*
            return storage_impl_->data<T>();
        */
    }
    
    pub fn unsafe_data<T>(&self) -> *mut T {
    
        todo!();
        /*
            return storage_impl_->unsafe_data<T>();
        */
    }

    // TODO: remove later
    pub fn set_nbytes(&self, size_bytes: usize)  {
        
        todo!();
        /*
            storage_impl_.get()->set_nbytes(size_bytes);
        */
    }
    
    pub fn resizable(&self) -> bool {
        
        todo!();
        /*
            return storage_impl_->resizable();
        */
    }
    
    pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
            return storage_impl_->nbytes();
        */
    }

    // get() use here is to get const-correctness
    
    pub fn data(&self)  {
        
        todo!();
        /*
            return storage_impl_.get()->data();
        */
    }
    
    pub fn data_ptr_mut(&mut self) -> &mut DataPtr {
        
        todo!();
        /*
            return storage_impl_->data_ptr();
        */
    }
    
    pub fn data_ptr(&self) -> &DataPtr {
        
        todo!();
        /*
            return storage_impl_->data_ptr();
        */
    }

    /// Returns the previous data_ptr
    pub fn set_data_ptr(&self, data_ptr: DataPtr) -> DataPtr {
        
        todo!();
        /*
            return storage_impl_.get()->set_data_ptr(move(data_ptr));
        */
    }
    
    pub fn set_data_ptr_noswap(&self, data_ptr: DataPtr)  {
        
        todo!();
        /*
            return storage_impl_.get()->set_data_ptr_noswap(move(data_ptr));
        */
    }
    
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return storage_impl_->device_type();
        */
    }
    
    pub fn allocator(&self) -> *mut Allocator {
        
        todo!();
        /*
            return storage_impl_.get()->allocator();
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return storage_impl_->device();
        */
    }
    
    pub fn unsafe_release_storage_impl(&mut self) -> *mut StorageImpl {
        
        todo!();
        /*
            return storage_impl_.release();
        */
    }
    
    pub fn unsafe_get_storage_impl(&self) -> *mut StorageImpl {
        
        todo!();
        /*
            return storage_impl_.get();
        */
    }
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return storage_impl_;
        */
    }
    
    pub fn use_count(&self) -> usize {
        
        todo!();
        /*
            return storage_impl_.use_count();
        */
    }
    
    #[inline] pub fn unique(&self) -> bool {
        
        todo!();
        /*
            return storage_impl_.unique();
        */
    }
    
    pub fn is_alias_of(&self, other: &Storage) -> bool {
        
        todo!();
        /*
            return storage_impl_ == other.storage_impl_;
        */
    }
    
    pub fn unique_storage_share_external_pointer_a<A: std::alloc::Allocator>(&mut self, 
        src:      *mut c_void,
        capacity: usize,
        d:        Option<A>)  {

        todo!();
        /*
            if (!storage_impl_.unique()) {
          TORCH_CHECK(
              false,
              "UniqueStorageShareExternalPointer can only be called when use_count == 1");
        }
        storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
        */
    }
    
    pub fn unique_storage_share_external_pointer_b(&mut self, 
        data_ptr: DataPtr,
        capacity: usize)  {
        
        todo!();
        /*
            if (!storage_impl_.unique()) {
          TORCH_CHECK(
              false,
              "UniqueStorageShareExternalPointer can only be called when use_count == 1");
        }
        storage_impl_->UniqueStorageShareExternalPointer(
            move(data_ptr), capacity);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/Storage.cpp]

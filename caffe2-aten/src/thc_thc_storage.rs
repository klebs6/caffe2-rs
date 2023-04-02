crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCStorage.h]

#[macro_export] macro_rules! thc_storage {
    ($NAME:ident) => {
        /*
                TH_CONCAT_4(TH,CReal,Storage_,NAME)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCStorage.hpp]
//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCStorage.cpp]

pub fn thc_storage_resize_bytes(
    state:      *mut THCState,
    self_:      *mut THCStorage,
    size_bytes: libc::ptrdiff_t)  {
    
    todo!();
        /*
            THArgCheck(size_bytes >= 0, 2, "invalid size");
      THAssert(self->allocator() != nullptr);
      int device;
      THCudaCheck(cudaGetDevice(&device));

      if (!self->resizable())
        THError("Trying to resize storage that is not resizable");

      if (size_bytes == 0) {
        self->set_data_ptr_noswap(DataPtr(nullptr, Device(DeviceType::Cuda, device)));
        self->set_nbytes(0);
      } else {
        DataPtr data = self->allocator()->allocate(size_bytes);

        if (self->data_ptr()) {
          // Enable p2p access when the memcpy is across devices
          THCState_getPeerToPeerAccess(state, device, THCStorage_getDevice(state, self));

          THCudaCheck(cudaMemcpyAsync(
              data.get(),
              self->data(),
              THMin(self->nbytes(), size_bytes),
              cudaMemcpyDeviceToDevice,
              getCurrentCUDAStream()));
        }

        // Destructively overwrite data_ptr
        self->set_data_ptr_noswap(move(data));
        self->set_nbytes(size_bytes);
      }
        */
}

pub fn thc_storage_get_device(
    state:   *mut THCState,
    storage: *const THCStorage) -> i32 {

    todo!();
        /*
            return storage->device().index();
        */
}

/// Should work with THStorageClass
///
pub fn thc_storage_new(state: *mut THCState) -> *mut THCStorage {
    
    todo!();
        /*
            THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_size_t(),
                               0,
                               CUDACachingAllocator::get(),
                               true)
                               .release();
      return storage;
        */
}

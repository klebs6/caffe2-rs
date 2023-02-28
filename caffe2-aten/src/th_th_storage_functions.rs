/*!
  | Note [Weak references for intrusive refcounting]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Here's the scheme:
  |
  |  - refcount == number of strong references to the object
  |    weakcount == number of weak references to the object,
  |      plus one more if refcount > 0
  |
  |  - THStorage stays live as long as there are any strong
  |    or weak pointers to it (weakcount > 0, since strong
  |    references count as a +1 to weakcount)
  |
  |  - finalizers are called and data_ptr is deallocated when refcount == 0
  |
  |  - Once refcount == 0, it can never again be > 0 (the transition
  |    from > 0 to == 0 is monotonic)
  |
  |  - When you access THStorage via a weak pointer, you must
  |    atomically increment the use count, if it is greater than 0.
  |    If it is not, you must report that the storage is dead.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THStorageFunctions.h]

#[macro_export] macro_rules! th_storage {
    ($NAME:ident) => {
        /*
                TH_CONCAT_4(TH,Real,Storage_,NAME)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THStorageFunctions.hpp]
//-------------------------------------------[.cpp/pytorch/aten/src/TH/THStorageFunctions.cpp]

pub fn th_storage_new() -> *mut THStorage {
    
    todo!();
    /*
    THStorage* storage = make_intrusive<StorageImpl>(
                       StorageImpl::use_byte_size_t(),
                       0,
                       getTHDefaultAllocator(),
                       true)
                       .release();
      return storage;
        */
}

/**
  | This exists to have a data-type independent
  | way of freeing (necessary for THPPointer).
  |
  | Free a non-weak pointer to THStorage
  |
  */
pub fn th_storage_free(storage: *mut THStorage)  {
    
    todo!();
        /*
            if (!storage) {
        return;
      }
      raw::intrusive_ptr::decref(storage);
        */
}


pub fn th_storage_retain(storage: *mut THStorage)  {
    
    todo!();
        /*
            if (storage) {
        raw::intrusive_ptr::incref(storage);
      }
        */
}

pub fn th_storage_resize_bytes(
    storage:    *mut THStorage,
    size_bytes: libc::ptrdiff_t)  {
    todo!();
        /*
            if (storage->resizable()) {
        /* case when the allocator does not have a realloc defined */
        DataPtr new_data;
        if (size_bytes != 0) {
          new_data = storage->allocator()->allocate(size_bytes);
        }
        DataPtr old_data = storage->set_data_ptr(move(new_data));
        ptrdiff_t old_capacity = storage->nbytes();
        storage->set_nbytes(size_bytes);
        if (old_data != nullptr) {
          ptrdiff_t copy_capacity = old_capacity;
          if (storage->nbytes() < copy_capacity) {
            copy_capacity = storage->nbytes();
          }
          if (copy_capacity > 0) {
            memcpy(storage->data(), old_data.get(), copy_capacity);
          }
        }
      } else {
        THError("Trying to resize storage that is not resizable");
      }
        */
}

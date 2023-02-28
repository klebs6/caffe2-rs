crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/UnsafeFromTH.h]

#[inline] pub fn unsafe_tensor_fromth(
        th_pointer: *mut c_void,
        retain:     bool) -> Tensor {
    
    todo!();
        /*
            auto tensor_impl = intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(static_cast<TensorImpl*>(th_pointer));
      if (retain && tensor_impl.get() != UndefinedTensorImpl::singleton()) {
        raw::intrusive_ptr::incref(tensor_impl.get());
      }
      return Tensor(move(tensor_impl));
        */
}

#[inline] pub fn unsafe_storage_fromth(
        th_pointer: *mut c_void,
        retain:     bool) -> Storage {
    
    todo!();
        /*
            if (retain && th_pointer) {
        raw::intrusive_ptr::incref(static_cast<StorageImpl*>(th_pointer));
      }
      return Storage(intrusive_ptr<StorageImpl>::reclaim(static_cast<StorageImpl*>(th_pointer)));
        */
}

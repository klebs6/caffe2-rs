crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THTensor.h]

#[macro_export] macro_rules! th_tensor {
    ($NAME:ident) => {
        /*
                TH_CONCAT_4(TH,Real,Tensor_,NAME)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THTensor.hpp]

/**
  | Returns a Tensor given a TensorImpl. The
  | TensorImpl remains valid after the the Tensor
  | is destroyed.
  |
  */
#[inline] pub fn th_tensor_wrap(tensor: *mut THTensor) -> Tensor {
    
    todo!();
        /*
            raw::intrusive_ptr::incref(tensor);
      return Tensor(intrusive_ptr<TensorImpl>::reclaim(tensor));
        */
}


#[inline] pub fn th_tensor_get_size_ptr(tensor: *mut THTensor) -> *const i64 {
    
    todo!();
        /*
            return tensor->sizes().data();
        */
}

#[inline] pub fn th_tensor_get_stride_ptr(tensor: *mut THTensor) -> *const i64 {
    
    todo!();
        /*
            return tensor->strides().data();
        */
}

/// NB: Non-retaining
#[inline] pub fn th_tensor_get_storage_ptr(tensor: *const THTensor) -> *mut THStorage {
    
    todo!();
        /*
            // Within PyTorch, the invariant is that storage_ is always
      // initialized; we never have tensors that don't have any storage.
      // However, for Caffe2, this is not true, because they have permitted
      // tensors to be allocated without specifying what scalar type
      // they should be, only to be filled when GetMutableData is called
      // for the first time (providing the necessary type).  It is an ERROR to
      // invoke any PyTorch operations on such a half-constructed storage,
      // and this check tests for that case.
      TORCH_CHECK(tensor->storage(), "Cannot use PyTorch operations on a half-constructed "
               "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
               "it first; otherwise, this is a bug, please report it.");
      return tensor->storage().unsafeGetStorageImpl();
        */
}

/**
  | [NOTE: nDimension vs nDimensionLegacyNoScalars
  | vs nDimensionLegacyAll]
  |
  | nDimension
  |
  | corresponds to the "true" ATen dimension.
  |
  | nDimensionLegacyNoScalars
  |
  | corresponds to the ATen dimension, except
  | scalars are viewed as 1-dimensional tensors.
  |
  | nDimensionLegacyAll
  |
  | corresponds to the ATen dimension, except
  | scalars are viewed as 1-dimensional tensors and
  | tensors with a dimension of size zero are
  | collapsed to 0-dimensional tensors.
  |
  | Eventually, everything should go through
  | nDimension or tensor->dim().
  |
  */
#[inline] pub fn th_tensor_n_dimension(tensor: *const THTensor) -> i32 {
    
    todo!();
        /*
            return tensor->dim();
        */
}


#[inline] pub fn th_tensor_n_dimension_legacy_no_scalars(tensor: *const THTensor) -> i32 {
    
    todo!();
        /*
            if (tensor->dim() == 0) {
        return 1;
      } else {
        return tensor->dim();
      }
        */
}


#[inline] pub fn th_tensor_n_dimension_legacy_all(tensor: *const THTensor) -> i32 {
    
    todo!();
        /*
            if (tensor->is_empty()) {
        return 0;
      } else if (tensor->dim() == 0) {
        return 1;
      } else {
        return tensor->dim();
      }
        */
}


#[inline] pub fn th_tensor_stride_legacy_no_scalars(
        self_: *const THTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
          dim, THTensor_nDimensionLegacyNoScalars(self));
      return self->dim() == 0 ? 1 : self->stride(dim);
        */
}


#[inline] pub fn th_tensor_size_legacy_no_scalars(
        self_: *const THTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
          dim, THTensor_nDimensionLegacyNoScalars(self));
      return self->dim() == 0 ? 1 : self->size(dim);
        */
}


#[inline] pub fn th_tensor_sizes_legacy_no_scalars(self_: *const THTensor) -> Vec<i64> {
    
    todo!();
        /*
            if (self->dim() == 0) {
        return {1};
      } else {
        return self->sizes().vec();
      }
        */
}


#[inline] pub fn th_tensor_strides_legacy_no_scalars(self_: *const THTensor) -> Vec<i64> {
    
    todo!();
        /*
            if (self->dim() == 0) {
        return {1};
      } else {
        return self->strides().vec();
      }
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THTensor.cpp]

/**
  | NB: This is NOT valid on UndefinedTensorImpl
  |
  */
pub fn th_tensor_free(self_: *mut THTensor)  {
    
    todo!();
        /*
            if (!self) return;
      raw::intrusive_ptr::decref(self);
        */
}

pub fn th_tensor_set_storage(
        self_:          *mut THTensor,
        storage:        *mut THStorage,
        storage_offset: libc::ptrdiff_t,
        size:           &[i32],
        stride:         &[i32])  {
    
    todo!();
        /*
            raw::intrusive_ptr::incref(storage_);
      THTensor_wrap(self).set_(Storage(intrusive_ptr<StorageImpl>::reclaim(storage_)), storageOffset_, size_, stride_);
        */
}


pub fn th_tensor_resize(
        self_:  *mut THTensor,
        size:   &[i32],
        stride: &[i32])  {
    
    todo!();
        /*
            if (stride.data()) {
        THArgCheck(stride.size() == size.size(), 3, "invalid stride");
      }

    #ifdef DEBUG
      THAssert(size.size() <= INT_MAX);
    #endif
      THTensor_resizeNd(self, size.size(), size.data(), stride.data());
        */
}


pub fn th_tensor_resize_nd(
        self_:       *mut THTensor,
        n_dimension: i32,
        size:        *const i64,
        stride:      *const i64)  {
    
    todo!();
        /*
            TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
      IntArrayRef sizes(size, nDimension);
      optional<IntArrayRef> strides;
      if (stride) {
        strides = IntArrayRef(stride, nDimension);
      }
      native::resize_impl_cpu_(self, sizes, strides);
        */
}

/// NB: Steals ownership of storage
///
pub fn th_tensor_steal_and_set_storage_ptr(
        tensor:  *mut THTensor,
        storage: *mut THStorage)  {
    
    todo!();
        /*
            // Caffe2 might have tensors whose storages are null, but we
      // don't allow it in PyTorch.
      AT_ASSERT(storage);

      // We used to allow this, but this breaks device caching.
      // Let's put an actual error message for this one.
      TORCH_CHECK(tensor->storage().device() == storage->device(),
                "Attempted to set the storage of a tensor on device \"", tensor->storage().device(),
                 "\" to a storage on different device \"", storage->device(),
                "\".  This is no longer allowed; the devices must match.");
      tensor->set_storage_keep_dtype(
          Storage(intrusive_ptr<THStorage>::reclaim(storage)));
        */
}

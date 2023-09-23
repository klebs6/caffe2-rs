crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Resize.h]

/**
  | These functions are called by native::resize_
  | as well as (legacy) TH resize.
  |
  | They are not in TH/THTensor.cpp because the at
  | namespace is easier to benchmark than TH;
  | I can't get gbenchmark to call fns from
  | THTensor.cpp
  |
  */
#[inline] pub fn maybe_resize_storage_cpu(
        self_:    *mut TensorImpl,
        new_size: u64)  {
    
    todo!();
        /*
            // It does not make sense to try to resize a storage
      // to hold 0 elements, and this can break
      // if storage_offset is positive but
      // new_size is 0, so just bail in that case
      // (same comment is in Resize.cuh)
      if (new_size == 0) {
        return;
      }
      if (!THTensor_getStoragePtr(self)) {
    #ifndef NDEBUG
        TypeMeta dtype = self->dtype();
    #endif
        THTensor_stealAndSetStoragePtr(self, THStorage_new());
    #ifndef NDEBUG
        // THTensor_stealAndSetStoragePtr guarantees this. Leave debug
        // assert in case of code changes.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dtype == self->dtype());
    #endif
      }
      u64 new_size_bytes =
          (new_size + self->storage_offset()) * self->dtype().itemsize();
      if (new_size_bytes > self->storage().nbytes()) {
        THStorage_resizeBytes(THTensor_getStoragePtr(self), new_size_bytes);
      }
        */
}

#[inline] pub fn resize_impl_cpu(
        self_:          *mut TensorImpl,
        size:           &[i32],
        stride:         Option<&[i32]>,
        resize_storage: bool) -> *mut TensorImpl {
    let resize_storage: bool = resize_storage.unwrap_or(true);

    todo!();
        /*
            if (self->sizes() == size && (!stride || self->strides() == stride)) {
        return self;
      }

      i64 storage_size = 1;
      if (stride) {
        self->set_sizes_and_strides(size, *stride);
        // NB: storage size can be different from numel.
        storage_size = storage_size_for(size, *stride);
      } else {
        self->set_sizes_contiguous(size);
        storage_size = self->numel();
      }
      if (resize_storage) {
        maybe_resize_storage_cpu(self, storage_size);
      }

      return self;
        */
}


#[inline] pub fn check_in_bounds_for_storage(
        size:           &[i32],
        stride:         &[i32],
        storage_offset: i64,
        data_type:      TypeMeta,
        new_storage:    &Storage)  {
    
    todo!();
        /*
            i64 storage_size_bytes =
          computeStorageNbytes(size, stride, data_type.itemsize());
      i64 storage_offset_bytes = storage_offset * data_type.itemsize();
      if (storage_size_bytes == 0) {
        // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
        return;
      }
      i64 new_storage_size_bytes = new_storage.nbytes();
      TORCH_CHECK(
          storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
          "setStorage: sizes ",
          size,
          ", strides ",
          stride,
          ","
          " storage offset ",
          storage_offset,
          ", and itemsize ",
          data_type.itemsize(),
          " requiring a storage size of ",
          storage_size_bytes + storage_offset_bytes,
          " are out of bounds for storage of size ",
          new_storage_size_bytes);
        */
}


#[inline] pub fn check_set_storage(
        result:         &mut Tensor,
        storage:        Storage,
        storage_offset: i64,
        size:           &[i32],
        stride:         &[i32])  {
    
    todo!();
        /*
            // FIXME: stride should be optional
      if (stride.data()) {
        TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                                  ") and stride length (", stride.size(), ")");
      }

    #ifdef DEBUG
      TORCH_CHECK(size.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
    #endif

      // storage: note this can't be replaced with result.set_(storage) as the semantics of that
      // function is to set the tensor size to be equal to the size of the storage.
      if (!result.storage().is_alias_of(storage)) {
        // Caffe2 might have tensors whose storages are null, but we
        // don't allow it in PyTorch.
        TORCH_INTERNAL_ASSERT(storage);
        TORCH_INTERNAL_ASSERT(result.storage());

        // We used to allow this, but this breaks device caching.
        // Let's put an actual error message for this one.
        TORCH_CHECK(result.storage().device() == storage.device(),
                    "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                    "\" to a storage on different device \"", storage.device(),
                    "\".  This is no longer allowed; the devices must match.");
        result.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);
      }

      // storageOffset
      TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
        */
}

/**
  | Set self's sizes, strides, and storage_offset.
  | (size, stride, storage_offset) must
  | be in bounds for self's storage.
  |
  */
#[inline] pub fn set_strided(
        self_:          &Tensor,
        size:           &[i32],
        stride:         &[i32],
        storage_offset: i64)  {
    
    todo!();
        /*
            TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
      auto* self_ = self.unsafeGetTensorImpl();
      checkInBoundsForStorage(
          size, stride, storage_offset, self_->dtype(), self_->storage());

      /* storage offset */
      TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
      self_->set_storage_offset(storage_offset);

      /* size and stride */
      if (self_->sizes() == size && self_->strides() == stride) {
        return;
      }
      for (auto val : stride) {
        TORCH_CHECK(val >= 0,
                    "as_strided: Negative strides are not supported at the moment, "
                    "got strides: ", stride);
      }
      self_->set_sizes_and_strides(size, stride);
        */
}



//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Resize.cpp]


/**
  | Returns true if resize is necessary
  |
  */
pub fn resize_output_check(
        output: &Tensor,
        shape:  &[i32]) -> bool {
    
    todo!();
        /*
            // Tests for resizing of tensors with one more elements
      if (output.sizes().equals(shape)) {
        return false;
      }
      if (output.numel() != 0) {
        TORCH_WARN(
          "An output with one or more elements was resized since it had ",
          "shape ", output.sizes(), ", which does not match the required ",
          "output shape ", shape, ".",
          "This behavior is deprecated, and in a future PyTorch release outputs ",
          "will not be resized unless they have zero elements. You can explicitly ",
          "reuse an out tensor t by resizing it, inplace, to zero elements with ",
          "t.resize_(0).");
      }
      return true;
        */
}

/**
  | TODO: make all operations that resize given
  |   outputs use this function for consistency and
  |   maintainability
  |
  | Resizes outputs
  |
  | Functions accepting output tensors, like with
  |   the "out" kwarg, should call this function to
  |   handle resizing their output tensor.
  |
  | Issues a warning if the output tensor has one
  |   or more elements and needs resizing
  |
  | NOTE: In the future the warning will become an
  | error
  |
  | Returns a bool saying whether or not the resize
  | actually happened or not
  */
pub fn resize_output(
    output: &Tensor,
    shape:  &[i32]) -> bool {
    
    todo!();
        /*
            if (resize_output_check(output, shape)) {
        // avoid a redispatch for cpu and cuda.
        // TODO: when resize_cuda_ is re-written to be unified with resize_,
        // we can provide the same benefit for cuda.
        if (output.is_cpu()) {
          native::resize_(output, shape);
        } else {
          output.resize_(shape);
        }
        return true;
      } else {
        return false;
      }
        */
}

/**
  | Call the sparse implementation in
  | SparseTensor.cpp directly.
  |
  | A dynamic dispatch here is NOT necessary, so
  | I didn't put this function in
  | native_functions.yaml
  |
  */
pub fn resize_as_sparse(
        self_: &Tensor,
        src:   &Tensor) -> &Tensor {
    
    todo!();
        /*
        
        */
}

/**
  | TODO(VitalyFedyunin): Move it to HTML docs.
  |
  | Strides of the output tensor of `resize_as_`
  | operator is defined by input tensor strides and
  | the value of memory_format argument.
  |
  | If memory_format is omitted and input tensor
  | have the same shape as output tensor, strides
  | of the output will remain unchanged. Strides
  | going to be set to contiguous if shapes are
  | different.
  |
  | If memory_format is equals to
  | MemoryFormat::Contiguous
  | (torch.contiguous_format) output tensor will
  | have contiguous strides.
  |
  | If memory_format is equal to
  | MemoryFormat::ChannelsLast
  | (torch.channels_last) and input tensor is 4D,
  | output tensor will have channels last memory
  | layout.
  |
  | If memory_format is equal to
  | MemoryFormat::Preserve (torch.preserve_format)
  | output tensor will be defined by strides of the
  | input tensor, following memory format
  | preservation rule:
  |
  |  - If input tensor strides are in channels last
  |    format, output tensor will have channels
  |    last memory layout.
  |
  |  - Otherwise, output tensor will have
  |  contiguous memory layout.
  |
  */
pub fn resize_as(
    self_:                  &Tensor,
    the_template:           &Tensor,
    optional_memory_format: Option<MemoryFormat>) -> &Tensor {

    todo!();
        /*
            if (self.is_sparse() && the_template.is_sparse()) {
        TORCH_CHECK(
            !optional_memory_format.has_value(),
            "Unsupported memory format for sparse tensor resize_as_ :",
            optional_memory_format.value());
        return native::resize_as_sparse_(self, the_template);
      }
      const Tensor& result = self.resize_(the_template.sizes());
      if (optional_memory_format.has_value()) {
        auto memory_format = optional_memory_format.value();
        if (memory_format == MemoryFormat::Preserve) {
          memory_format = the_template.suggest_memory_format();
        }
        self.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
      }
      namedinference::propagate_names(result, the_template);
      return result;
        */
}

pub fn resize(
    self_:                  &Tensor,
    size:                   &[i32],
    optional_memory_format: Option<MemoryFormat>) -> &Tensor {
    
    todo!();
        /*
            if (self.has_names()) {
        return resize_named_tensor_(self, size, optional_memory_format);
      }
      auto* self_ = self.unsafeGetTensorImpl();
      resize_impl_cpu_(self_, size, /*strides=*/nullopt);
      if (optional_memory_format.has_value()) {
        auto memory_format =
            optional_memory_format.value();
        TORCH_CHECK(
            memory_format != MemoryFormat::Preserve,
            "Unsupported memory format",
            memory_format);
        self_->empty_tensor_restride(memory_format);
      }
      return self;
        */
}

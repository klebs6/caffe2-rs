crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCTensor.h]

#[macro_export] macro_rules! thc_tensor {
    ($NAME:ident) => {
        /*
                TH_CONCAT_4(TH,CReal,Tensor_,NAME)
        */
    }
}

pub const THC_DESC_BUFF_LEN: usize = 64;

pub struct THCDescBuff {
    str_: [u8; THC_DESC_BUFF_LEN],
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCTensor.hpp]
//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCTensor.cpp]


// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
//
pub fn thc_tensor_n_dimension(
        state: *mut THCState,
        self_: *const THCTensor) -> i32 {
    
    todo!();
        /*
            return THTensor_nDimension(self);
        */
}


pub fn thc_tensor_n_dimension_legacy_no_scalars(
        state: *mut THCState,
        self_: *const THCTensor) -> i32 {
    
    todo!();
        /*
            return THTensor_nDimensionLegacyNoScalars(self);
        */
}


pub fn thc_tensor_n_dimension_legacy_all(
        state: *mut THCState,
        self_: *const THCTensor) -> i32 {
    
    todo!();
        /*
            return THTensor_nDimensionLegacyAll(self);
        */
}


pub fn thc_tensor_size(
        state: *mut THCState,
        self_: *const THCTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
      return self->size(dim);
        */
}


pub fn thc_tensor_size_legacy_no_scalars(
        state: *mut THCState,
        self_: *const THCTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            return THTensor_sizeLegacyNoScalars(self, dim);
        */
}


pub fn thc_tensor_stride(
        state: *mut THCState,
        self_: *const THCTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
      return self->stride(dim);
        */
}


pub fn thc_tensor_stride_legacy_no_scalars(
        state: *mut THCState,
        self_: *const THCTensor,
        dim:   i32) -> i64 {
    
    todo!();
        /*
            return THTensor_strideLegacyNoScalars(self, dim);
        */
}


pub fn thc_tensor_new(
        state:     *mut THCState,
        type_meta: TypeMeta) -> *mut THCTensor {
    
    todo!();
        /*
            auto scalar_type = typeMetaToScalarType(type_meta);
      switch (scalar_type) {
        case ScalarType::Byte:
          return THCudaByteTensor_new(state);
        case ScalarType::Char:
          return THCudaCharTensor_new(state);
        case ScalarType::Short:
          return THCudaShortTensor_new(state);
        case ScalarType::Int:
          return THCudaIntTensor_new(state);
        case ScalarType::Long:
          return THCudaLongTensor_new(state);
        case ScalarType::Half:
          return THCudaHalfTensor_new(state);
        case ScalarType::Float:
          return THCudaTensor_new(state);
        case ScalarType::Double:
          return THCudaDoubleTensor_new(state);
        case ScalarType::Bool:
          return THCudaBoolTensor_new(state);
        case ScalarType::BFloat16:
          return THCudaBFloat16Tensor_new(state);
        case ScalarType::ComplexFloat:
          return THCudaComplexFloatTensor_new(state);
        case ScalarType::ComplexDouble:
          return THCudaComplexDoubleTensor_new(state);
        default:
          AT_ERROR("unexpected ScalarType: ", toString(scalar_type));
      }
        */
}


pub fn thc_tensor_resize(
        state:  *mut THCState,
        self_:  *mut THCTensor,
        size:   &[i32],
        stride: &[i32])  {
    
    todo!();
        /*
            if(stride.data()) {
        THArgCheck(stride.size() == size.size(), 3, "invalid stride");
      }

    #ifdef DEBUG
      THAssert(size.size() <= INT_MAX);
    #endif
      THCTensor_resizeNd(state, self, size.size(), size.data(), stride.data());
        */
}



pub fn thc_tensor_resize_as(
        state: *mut THCState,
        self_: *mut THCTensor,
        src:   *mut THCTensor)  {
    
    todo!();
        /*
            int isSame = 0;
      int d;
      if(self->dim() == src->dim())
      {
        isSame = 1;
        for(d = 0; d < self->dim(); d++)
        {
          if(self->size(d) != src->size(d))
          {
            isSame = 0;
            break;
          }
        }
      }

      if(!isSame)
        THCTensor_resizeNd(state, self, src->dim(), THTensor_getSizePtr(src), NULL);
        */
}


pub fn thc_tensor_resize_nd(
        state:       *mut THCState,
        self_:       *mut THCTensor,
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
      native::resize_impl_cuda_(self, sizes, strides, /*device_guard=*/false);
        */
}


pub fn thc_tensor_set(
        state: *mut THCState,
        self_: *mut THCTensor,
        src:   *mut THCTensor)  {
    
    todo!();
        /*
            if(self != src)
        THCTensor_setStorage(state,
                             self,
                             THTensor_getStoragePtr(src),
                             src->storage_offset(),
                             src->sizes(),
                             src->strides());
        */
}



pub fn thc_tensor_set_storage(
        state:          *mut THCState,
        self_:          *mut THCTensor,
        storage:        *mut THCStorage,
        storage_offset: libc::ptrdiff_t,
        size:           &[i32],
        stride:         &[i32])  {
    
    todo!();
        /*
            raw::intrusive_ptr::incref(storage_);
      THTensor_wrap(self).set_(Storage(intrusive_ptr<StorageImpl>::reclaim(storage_)),
                               storageOffset_, size_, stride_);
        */
}


pub fn thc_tensor_squeeze1d(
        state:     *mut THCState,
        self_:     *mut THCTensor,
        src:       *mut THCTensor,
        dimension: i32)  {
    
    todo!();
        /*
            int d;

      if(!src)
        src = self;

      THArgCheck(dimension < src->dim(), 3, "dimension out of range");

      THCTensor_set(state, self, src);

      if(src->size(dimension) == 1)
      {
        DimVector newSize(static_cast<usize>(self->dim() - 1));
        DimVector newStride(static_cast<usize>(self->dim() - 1));
        for (d = 0; d < dimension; d++)
        {
          newSize[d] = self->size(d);
          newStride[d] = self->stride(d);
        }

        for(d = dimension; d < self->dim()-1; d++)
        {
          newSize[d] = self->size(d+1);
          newStride[d] = self->stride(d+1);
        }
        self->set_sizes_and_strides(newSize, newStride);
      }
        */
}


pub fn thc_tensor_unsqueeze1d(
        state:     *mut THCState,
        self_:     *mut THCTensor,
        src:       *mut THCTensor,
        dimension: i32)  {
    
    todo!();
        /*
            int d;

      if(!src)
        src = self;

      THArgCheck((dimension >= 0) && (dimension <= src->dim()), 3, "dimension out of range");

      THCTensor_set(state, self, src);

      DimVector newSize(static_cast<usize>(/* size */ self->dim()+1));
      DimVector newStride(static_cast<usize>(/* size */ self->dim()+1));

      for(d = self->dim(); d > dimension; d--)
      {
        newSize[d] = self->size(d-1);
        newStride[d] = self->stride(d-1);
      }
      if (dimension < self->dim())
      {
        newStride[dimension] = self->size(dimension) * self->stride(dimension);
      }
      else
      {
        newStride[dimension] = 1;
      }
      newSize[dimension] = 1;
      for(d = dimension - 1; d >= 0; d--)
      {
        newSize[d] = self->size(d);
        newStride[d] = self->stride(d);
      }
      self->set_sizes_and_strides(newSize, newStride);
        */
}


pub fn thc_tensor_all_contiguous(
        state:      *mut THCState,
        inputs:     *mut *mut THCTensor,
        num_inputs: i32) -> bool {
    
    todo!();
        /*
            THAssert(numInputs > 0);
      for (int i = 0; i < numInputs; ++i) {
        if (!inputs[i]->is_contiguous()) {
          return false;
        }
      }
      return true;
        */
}


pub fn thc_tensor_n_element(
        state: *mut THCState,
        self_: *const THCTensor) -> libc::ptrdiff_t {
    
    todo!();
        /*
            if(THTensor_nDimensionLegacyAll(self) == 0) {
        return 0;
      } else {
        return self->numel();
      }
        */
}

/**
  | NB: It is INVALID to call this on an UndefinedTensor
  |
  */
pub fn thc_tensor_retain(
        state: *mut THCState,
        self_: *mut THCTensor)  {
    
    todo!();
        /*
            raw::intrusive_ptr::incref(self);
        */
}


pub fn thc_tensor_free(
        state: *mut THCState,
        self_: *mut THCTensor)  {
    
    todo!();
        /*
            THTensor_free(self);
        */
}


pub fn thc_tensor_get_device(
        state:  *mut THCState,
        tensor: *const THCTensor) -> i32 {
    
    todo!();
        /*
            if (!THTensor_getStoragePtr(tensor)) return -1;
      return THCStorage_getDevice(state, THTensor_getStoragePtr(tensor));
        */
}


pub fn thc_tensor_all_same_device(
        state:      *mut THCState,
        inputs:     *mut *mut THCTensor,
        num_inputs: i32) -> bool {
    
    todo!();
        /*
            THAssert(numInputs > 0);
      int device = THCTensor_getDevice(state, inputs[0]);
      for (int i = 1; i < numInputs; ++i) {
        if (THCTensor_getDevice(state, inputs[i]) != device) {
          return false;
        }
      }
      return true;
        */
}

/**
  | Can we use 32 bit math for indexing?
  |
  */
pub fn thc_tensor_can_use_32bit_index_math(
        state:    *mut THCState,
        t:        *const THCTensor,
        max_elem: libc::ptrdiff_t) -> bool {

    let max_elem: libc::ptrdiff_t = max_elem.unwrap_or(INT32_MAX);
    
    todo!();
        /*
            ptrdiff_t elements = THCTensor_nElement(state, t);
      if (elements >= max_elem) {
        return false;
      }
      if (t->dim() == 0) {
        return true;
      }

      ptrdiff_t offset = 0;
      ptrdiff_t linearId = elements - 1;

      for (int i = THCTensor_nDimensionLegacyAll(state, t) - 1; i >= 0; --i) {
        ptrdiff_t curDimIndex =
          linearId % THCTensor_size(state, t, i);
        ptrdiff_t curDimOffset = curDimIndex *
          THCTensor_stride(state, t, i);
        offset += curDimOffset;
        linearId /= THCTensor_size(state, t, i);
      }

      if (offset >= max_elem) {
        return false;
      }

      return true;
        */
}

/**
  | Are all tensors 32-bit indexable?
  |
  */
pub fn thc_tensor_all_32bit_indexable(
        state:      *mut THCState,
        inputs:     *mut *mut THCTensor,
        num_inputs: i32) -> bool {
    
    todo!();
        /*
            for (int i = 0; i < numInputs; ++i) {
        if (!THCTensor_canUse32BitIndexMath(state, inputs[i])) {
          return false;
        }
      }
      return true;
        */
}

/**
  | Due to the resize semantics of ops with
  | `out=` keywords, if the output `tensor`
  | has the same shape as the output of the
  | reduction operation, then any noncontiguities
  | in the output `tensor` should be preserved.
  | 
  | This needs to be special cased b/c otherwise,
  | when keepdim=False, the implementations
  | of reduction ops resize `tensor` to
  | the reduced size with keepdim=True,
  | and then later squeeze `tensor` to the
  | correct output size, breaking the contiguity
  | guarantees of the resize semantics.
  |
  */
pub fn thc_tensor_preserve_reduce_dim_semantics(
    state:     *mut THCState,
    tensor:    *mut THCTensor,
    in_dims:   i32,
    dimension: i64,
    keepdim:   i32)  {

    todo!();
    /*
       int out_dims = THCTensor_nDimensionLegacyAll(state, tensor);
      if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {
        THCTensor_unsqueeze1d(state, tensor, tensor, dimension);
      }
        */
}

pub struct SizeAndStride {
    size:   i64,
    stride: i64,
}

/**
  | A comparator that will sort SizeAndStride
  | structs by stride, in ascending order.
  |
  */
pub fn compare_size_and_stride(
    a: *const c_void,
    b: *const c_void) -> i32 {
    
    todo!();
        /*
            const SizeAndStride* aS = (const SizeAndStride*) a;
      const SizeAndStride* bS = (const SizeAndStride*) b;

      if (aS->stride < bS->stride) return -1;
      if (aS->stride == bS->stride) return 0;
      return 1;
        */
}

/**
  | Returns false if there is no possibility
  | that the tensor has "overlapping" indices
  | and true otherwise.
  | 
  | "Overlapping" indices are two+ valid
  | indices that specify the same offset
  | within the tensor.
  | 
  | The function does this by checking for
  | a sufficient but not necessary condition
  | of no overlap.
  | 
  | In particular, that that there exists
  | an ordering of the tensor's dimensions
  | that is nicely "nested," with each dimension
  | contained within the next one.
  |
  | Returns false if there is no possibility
  | that the tensor has more than one index
  | that references the same datapoint,
  | true otherwise.
  |
  */
pub fn thc_tensor_maybe_overlapping_indices(
    state: *mut THCState,
    t:     *const THCTensor) -> bool {
    
    todo!();
        /*
            /* Extract size/stride arrays; only consider size >1 dims. */
      SizeAndStride info[MAX_CUTORCH_DIMS];

      int dims = THCTensor_nDimensionLegacyAll(state, t);
      int nonSize1Dims = 0;
      for (int i = 0; i < dims; ++i) {
        i64 size = THCTensor_sizeLegacyNoScalars(state, t, i);

        if (size > 1) {
          info[nonSize1Dims].size = size;
          info[nonSize1Dims].stride =
            THCTensor_stride(state, t, i);

          if (info[nonSize1Dims].stride < 1) {
            return true;
          }

          ++nonSize1Dims;
        }
      }

      /* Short-circuits if tensor is a single element.             */
      if (nonSize1Dims == 0) {
        return false;
      }

      /* Ascending order (innermost dimension in sorted view is at [0]) */
      qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

      for (int i = 0; i < (nonSize1Dims - 1); ++i) {
        if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
          return true;
        }
      }

      return false;
        */
}

// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCTensor.h]
lazy_static!{
    /*
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCTensor.h"
    #else

    #define THCTensor THTensor

    // These used to be distinct types; for some measure of backwards compatibility and documentation
    // alias these to the single THCTensor type.
    #define THCudaTensor                THCTensor
    #define THCudaDoubleTensor          THCTensor
    #define THCudaHalfTensor            THCTensor
    #define THCudaByteTensor            THCTensor
    #define THCudaCharTensor            THCTensor
    #define THCudaShortTensor           THCTensor
    #define THCudaIntTensor             THCTensor
    #define THCudaLongTensor            THCTensor
    #define THCudaBoolTensor            THCTensor
    #define THCudaBFloat16Tensor        THCTensor
    #define THCudaComplexFloatTensor    THCTensor
    #define THCudaComplexDoubleTensor   THCTensor

    /**** access methods ****/
    TORCH_CUDA_CU_API THCStorage* THCTensor_(
        storage)(THCState* state, const THCTensor* self);
    TORCH_CUDA_CU_API ptrdiff_t
        THCTensor_(storageOffset)(THCState* state, const THCTensor* self);

    // See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
    TORCH_CUDA_CU_API int THCTensor_(
        nDimension)(THCState* state, const THCTensor* self);
    TORCH_CUDA_CU_API int THCTensor_(
        nDimensionLegacyNoScalars)(THCState* state, const THCTensor* self);
    TORCH_CUDA_CU_API int THCTensor_(
        nDimensionLegacyAll)(THCState* state, const THCTensor* self);

    TORCH_CUDA_CU_API i64
        THCTensor_(size)(THCState* state, const THCTensor* self, int dim);
    TORCH_CUDA_CU_API i64 THCTensor_(
        sizeLegacyNoScalars)(THCState* state, const THCTensor* self, int dim);
    TORCH_CUDA_CU_API i64
        THCTensor_(stride)(THCState* state, const THCTensor* self, int dim);
    TORCH_CUDA_CU_API i64 THCTensor_(
        strideLegacyNoScalars)(THCState* state, const THCTensor* self, int dim);
    TORCH_CUDA_CU_API Scalar* THCTensor_(
        data)(THCState* state, const THCTensor* self);

    TORCH_CUDA_CU_API void THCTensor_(
        setFlag)(THCState* state, THCTensor* self, const char flag);
    TORCH_CUDA_CU_API void THCTensor_(
        clearFlag)(THCState* state, THCTensor* self, const char flag);

    /**** creation methods ****/
    TORCH_CUDA_CU_API THCTensor* THCTensor_(new)(THCState* state);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(
        newWithTensor)(THCState* state, THCTensor* tensor);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(newWithStorage1d)(
        THCState* state,
        THCStorage* storage_,
        ptrdiff_t storageOffset_,
        i64 size0_,
        i64 stride0_);

    /* stride might be NULL */
    TORCH_CUDA_CU_API THCTensor* THCTensor_(
        newWithSize1d)(THCState* state, i64 size0_);

    TORCH_CUDA_CU_API THCTensor* THCTensor_(
        newClone)(THCState* state, THCTensor* self);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(
        newContiguous)(THCState* state, THCTensor* tensor);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(newSelect)(
        THCState* state,
        THCTensor* tensor,
        int dimension_,
        i64 sliceIndex_);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(newNarrow)(
        THCState* state,
        THCTensor* tensor,
        int dimension_,
        i64 firstIndex_,
        i64 size_);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(newTranspose)(
        THCState* state,
        THCTensor* tensor,
        int dimension1_,
        int dimension2_);
    TORCH_CUDA_CU_API THCTensor* THCTensor_(
        newFoldBatchDim)(THCState* state, THCTensor* input);

    // resize* methods simply resize the storage. So they may not retain the current data at current indices.
    // This is especially likely to happen when the tensor is not contiguous. In general, if you still need the
    // values, unless you are doing some size and stride tricks, do not use resize*.
    TORCH_CUDA_CU_API void THCTensor_(resizeNd)(
        THCState* state,
        THCTensor* tensor,
        int nDimension,
        const i64* size,
        const i64* stride);
    TORCH_CUDA_CU_API void THCTensor_(
        resizeAs)(THCState* state, THCTensor* tensor, THCTensor* src);
    TORCH_CUDA_CU_API void THCTensor_(resize0d)(THCState* state, THCTensor* tensor);
    TORCH_CUDA_CU_API void THCTensor_(
        resize1d)(THCState* state, THCTensor* tensor, i64 size0_);
    TORCH_CUDA_CU_API void THCTensor_(resize2d)(
        THCState* state,
        THCTensor* tensor,
        i64 size0_,
        i64 size1_);
    TORCH_CUDA_CU_API void THCTensor_(resize3d)(
        THCState* state,
        THCTensor* tensor,
        i64 size0_,
        i64 size1_,
        i64 size2_);
    TORCH_CUDA_CU_API void THCTensor_(resize4d)(
        THCState* state,
        THCTensor* tensor,
        i64 size0_,
        i64 size1_,
        i64 size2_,
        i64 size3_);
    TORCH_CUDA_CU_API void THCTensor_(resize5d)(
        THCState* state,
        THCTensor* tensor,
        i64 size0_,
        i64 size1_,
        i64 size2_,
        i64 size3_,
        i64 size4_);

    TORCH_CUDA_CU_API void THCTensor_(
        set)(THCState* state, THCTensor* self, THCTensor* src);

    TORCH_CUDA_CU_API void THCTensor_(narrow)(
        THCState* state,
        THCTensor* self,
        THCTensor* src,
        int dimension_,
        i64 firstIndex_,
        i64 size_);
    TORCH_CUDA_CU_API void THCTensor_(select)(
        THCState* state,
        THCTensor* self,
        THCTensor* src,
        int dimension_,
        i64 sliceIndex_);
    TORCH_CUDA_CU_API void THCTensor_(transpose)(
        THCState* state,
        THCTensor* self,
        THCTensor* src,
        int dimension1_,
        int dimension2_);

    TORCH_CUDA_CU_API void THCTensor_(squeeze1d)(
        THCState* state,
        THCTensor* self,
        THCTensor* src,
        int dimension_);
    TORCH_CUDA_CU_API void THCTensor_(unsqueeze1d)(
        THCState* state,
        THCTensor* self,
        THCTensor* src,
        int dimension_);

    TORCH_CUDA_CU_API int THCTensor_(
        isContiguous)(THCState* state, const THCTensor* self);
    TORCH_CUDA_CU_API int THCTensor_(
        isSameSizeAs)(THCState* state, const THCTensor* self, const THCTensor* src);
    TORCH_CUDA_CU_API ptrdiff_t
        THCTensor_(nElement)(THCState* state, const THCTensor* self);

    TORCH_CUDA_CU_API void THCTensor_(retain)(THCState* state, THCTensor* self);
    TORCH_CUDA_CU_API void THCTensor_(free)(THCState* state, THCTensor* self);
    TORCH_CUDA_CU_API void THCTensor_(
        freeCopyTo)(THCState* state, THCTensor* self, THCTensor* dst);

    /* Slow access methods [check everything] */
    TORCH_CUDA_CU_API void THCTensor_(
        set0d)(THCState* state, THCTensor* tensor, Scalar value);
    TORCH_CUDA_CU_API void THCTensor_(
        set1d)(THCState* state, THCTensor* tensor, i64 x0, Scalar value);
    TORCH_CUDA_CU_API void THCTensor_(set2d)(
        THCState* state,
        THCTensor* tensor,
        i64 x0,
        i64 x1,
        Scalar value);
    TORCH_CUDA_CU_API void THCTensor_(set3d)(
        THCState* state,
        THCTensor* tensor,
        i64 x0,
        i64 x1,
        i64 x2,
        Scalar value);
    TORCH_CUDA_CU_API void THCTensor_(set4d)(
        THCState* state,
        THCTensor* tensor,
        i64 x0,
        i64 x1,
        i64 x2,
        i64 x3,
        Scalar value);

    TORCH_CUDA_CU_API Scalar
        THCTensor_(get0d)(THCState* state, const THCTensor* tensor);
    TORCH_CUDA_CU_API Scalar
        THCTensor_(get1d)(THCState* state, const THCTensor* tensor, i64 x0);
    TORCH_CUDA_CU_API Scalar THCTensor_(
        get2d)(THCState* state, const THCTensor* tensor, i64 x0, i64 x1);
    TORCH_CUDA_CU_API Scalar THCTensor_(get3d)(
        THCState* state,
        const THCTensor* tensor,
        i64 x0,
        i64 x1,
        i64 x2);
    TORCH_CUDA_CU_API Scalar THCTensor_(get4d)(
        THCState* state,
        const THCTensor* tensor,
        i64 x0,
        i64 x1,
        i64 x2,
        i64 x3);

    /* CUDA-specific functions */
    TORCH_CUDA_CU_API int THCTensor_(
        getDevice)(THCState* state, const THCTensor* self);
    TORCH_CUDA_CU_API int THCTensor_(
        checkGPU)(THCState* state, unsigned int nTensors, ...);

    /* debug methods */
    TORCH_CUDA_CU_API THCDescBuff
        THCTensor_(sizeDesc)(THCState* state, const THCTensor* tensor);

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCTensor.hpp]
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCTensor.hpp"
    #else

    // STOP!!! Thinking of including this header directly?  Please
    // read Note [TH abstraction violation]

    // NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
    // new functions in here, they should probably be un-genericized.

    TORCH_CUDA_CU_API void THCTensor_(setStorage)(
        THCState* state,
        THCTensor* self,
        THCStorage* storage_,
        ptrdiff_t storageOffset_,
        IntArrayRef size_,
        IntArrayRef stride_);

    TORCH_CUDA_CU_API void THCTensor_(resize)(
        THCState* state,
        THCTensor* self,
        IntArrayRef size,
        IntArrayRef stride);

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCTensor.cpp]
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCTensor.cpp"
    #else


    /**** access methods ****/
    THCStorage *THCTensor_(storage)(THCState *state, const THCTensor *self)
    {
      return THTensor_getStoragePtr(self);
    }

    ptrdiff_t THCTensor_(storageOffset)(THCState *state, const THCTensor *self)
    {
      return self->storage_offset();
    }

    int THCTensor_(nDimension)(THCState *state, const THCTensor *self)
    {
      return THCTensor_nDimension(state, self);
    }

    int THCTensor_(nDimensionLegacyNoScalars)(THCState *state, const THCTensor *self)
    {
      return THCTensor_nDimensionLegacyNoScalars(state, self);
    }

    int THCTensor_(nDimensionLegacyAll)(THCState *state, const THCTensor *self)
    {
      return THCTensor_nDimensionLegacyAll(state, self);
    }

    i64 THCTensor_(size)(THCState *state, const THCTensor *self, int dim)
    {
      return THCTensor_size(state, self, dim);
    }

    i64 THCTensor_(sizeLegacyNoScalars)(THCState *state, const THCTensor *self, int dim)
    {
      return THTensor_sizeLegacyNoScalars(self, dim);
    }

    i64 THCTensor_(stride)(THCState *state, const THCTensor *self, int dim)
    {
      return THCTensor_stride(state, self, dim);
    }

    i64 THCTensor_(strideLegacyNoScalars)(THCState *state, const THCTensor *self, int dim)
    {
      return THTensor_strideLegacyNoScalars(self, dim);
    }

    Scalar *THCTensor_(data)(THCState *state, const THCTensor *self)
    {
      if(THTensor_getStoragePtr(self))
        return (THCStorage_(data)(state, THTensor_getStoragePtr(self))+self->storage_offset());
      else
        return NULL;
    }

    /**** creation methods ****/

    /* Empty init */
    THCTensor *THCTensor_(new)(THCState *state)
    {
      return make_intrusive<TensorImpl, UndefinedTensorImpl>(
                 intrusive_ptr<StorageImpl>::reclaim(
                     THCStorage_(new)(state)),
                 DispatchKey::CUDA,
                 TypeMeta::Make<Scalar>())
          .release();
    }

    /* Pointer-copy init */
    THCTensor *THCTensor_(newWithTensor)(THCState *state, THCTensor *tensor)
    {
      return native::alias(THTensor_wrap(tensor)).unsafeReleaseTensorImpl();
    }

    THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                                   i64 size0, i64 stride0)
    {
      raw::intrusive_ptr::incref(storage);
      THTensor* self = make_intrusive<TensorImpl, UndefinedTensorImpl>(
                           intrusive_ptr<StorageImpl>::reclaim(storage),
                           DispatchKey::CUDA,
                           TypeMeta::Make<Scalar>())
                           .release();
      THCTensor_(setStorage)(state, self, storage, storageOffset, {size0}, {stride0});

      return self;
    }

    THCTensor *THCTensor_(newWithSize)(THCState *state, IntArrayRef size, IntArrayRef stride)
    {
      TORCH_INTERNAL_ASSERT(false, "this function should not be called and is in the process of being removed");
    }

    THCTensor *THCTensor_(newWithSize1d)(THCState *state, i64 size0)
    {
      THCStorage *new_storage = THCStorage_(new)(state);
      THCTensor* self =
          make_intrusive<TensorImpl, UndefinedTensorImpl>(
              intrusive_ptr<StorageImpl>::reclaim(new_storage),
              DispatchKey::CUDA,
              TypeMeta::Make<Scalar>())
              .release();
      THCTensor_(setStorage)(state, self, new_storage, 0, {size0}, {});

      return self;
    }

    THCTensor *THCTensor_(newClone)(THCState *state, THCTensor *self)
    {
      // already available in Aten as clone()
      THCTensor *tensor = THCTensor_(new)(state);
      Tensor tensor_wrap = THTensor_wrap(tensor);
      Tensor self_wrap = THTensor_wrap(self);
      tensor_wrap.resize_as_(self_wrap);
      THCTensor_(copy)(state, tensor, self);
      return tensor;
    }

    THCTensor *THCTensor_(newContiguous)(THCState *state, THCTensor *self)
    {
      if(!THCTensor_(isContiguous)(state, self)) {
        return THCTensor_(newClone)(state, self);
      } else {
        THCTensor_(retain)(state, self);
        return self;
      }
    }

    THCTensor *THCTensor_(newSelect)(THCState *state, THCTensor *tensor, int dimension_, i64 sliceIndex_)
    {
      THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(select)(state, self, NULL, dimension_, sliceIndex_);
      return self;
    }

    THCTensor *THCTensor_(newNarrow)(THCState *state, THCTensor *tensor, int dimension_, i64 firstIndex_, i64 size_)
    {
      THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(narrow)(state, self, NULL, dimension_, firstIndex_, size_);
      return self;
    }

    THCTensor *THCTensor_(newTranspose)(THCState *state, THCTensor *tensor, int dimension1_, int dimension2_)
    {
      THCTensor *self = THCTensor_(newWithTensor)(state, tensor);
      THCTensor_(transpose)(state, self, NULL, dimension1_, dimension2_);
      return self;
    }

    // Collapses the first two dimensions of a tensor.
    // Assumes the input tensor is contiguous.
    THCTensor *THCTensor_(newFoldBatchDim)(THCState *state, THCTensor *input) {
      int in_dims = THCTensor_(nDimensionLegacyAll)(state, input);
      THArgCheck(in_dims >= 2, 1, "Tensor needs to have at least two dimensions");
      THArgCheck(THCTensor_(isContiguous)(state, input), 1,
                 "Tensor must be contiguous");
      vector<i64> new_size(in_dims - 1);
      new_size[0] = THCTensor_(size)(state, input, 0) * THCTensor_(size)(state, input, 1);
      for (int i = 2; i < in_dims; i++) {
        new_size[i - 1] = THCTensor_(size)(state, input, i);
      }
      THCTensor *output = native::view(THTensor_wrap(input), new_size).unsafeReleaseTensorImpl();
      return output;
    }

    /* Resize */
    void THCTensor_(resize)(THCState *state, THCTensor *self, IntArrayRef size, IntArrayRef stride)
    {
      THCTensor_resize(state, self, size, stride);
    }

    void THCTensor_(resizeAs)(THCState *state, THCTensor *self, THCTensor *src)
    {
      // already available in Aten as resize_as_()
      THCTensor_resizeAs(state, self, src);
    }

    void THCTensor_(resize0d)(THCState *state, THCTensor *tensor)
    {
      THCTensor_resizeNd(state, tensor, 0, {}, nullptr);
    }

    void THCTensor_(resize1d)(THCState *state, THCTensor *tensor, i64 size0)
    {
      i64 size[1] = {size0};
      THCTensor_resizeNd(state, tensor, 1, size, nullptr);
    }

    void THCTensor_(resize2d)(THCState *state, THCTensor *tensor, i64 size0, i64 size1)
    {
      i64 size[2] = {size0, size1};
      THCTensor_resizeNd(state, tensor, 2, size, nullptr);
    }

    void THCTensor_(resize3d)(THCState *state, THCTensor *tensor, i64 size0, i64 size1, i64 size2)
    {
      i64 size[3] = {size0, size1, size2};
      THCTensor_resizeNd(state, tensor, 3, size, nullptr);
    }

    void THCTensor_(resize4d)(THCState *state, THCTensor *self, i64 size0, i64 size1, i64 size2, i64 size3)
    {
      i64 size[4] = {size0, size1, size2, size3};
      THCTensor_resizeNd(state, self, 4, size, nullptr);
    }

    void THCTensor_(resize5d)(THCState *state, THCTensor *self, i64 size0, i64 size1, i64 size2, i64 size3, i64 size4)
    {
      i64 size[5] = {size0, size1, size2, size3, size4};
      THCTensor_resizeNd(state, self, 5, size, nullptr);
    }

    void THCTensor_(set)(THCState *state, THCTensor *self, THCTensor *src)
    {
      THCTensor_set(state, self, src);
    }

    void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, IntArrayRef size_, IntArrayRef stride_) {
      THCTensor_setStorage(state, self, storage_, storageOffset_, size_, stride_);
    }

    void THCTensor_(narrow)(THCState *state, THCTensor *self, THCTensor *src, int dimension, i64 firstIndex, i64 size)
    {
      if(!src)
        src = self;

      THArgCheck( (dimension >= 0) && (dimension < src->dim()), 3, "out of range");
      THArgCheck( firstIndex >= 0, 4, "out of range");
      THArgCheck( size >= 0, 5, "out of range");
      THArgCheck(firstIndex+size <= src->size(dimension), 5, "out of range");

      THCTensor_(set)(state, self, src);

      if (firstIndex > 0) {
        self->set_storage_offset(self->storage_offset() + firstIndex*self->stride(dimension));
      }

      self->set_size(dimension, size);
    }

    void THCTensor_(select)(THCState *state, THCTensor *self, THCTensor *src, int dimension, i64 sliceIndex)
    {
      int d;

      if(!src)
        src = self;

      THArgCheck(src->dim() > 0, 1, "cannot select on a 0-dim tensor");
      THArgCheck((dimension >= 0) && (dimension < src->dim()), 3, "out of range");
      THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size(dimension)), 4, "out of range");

      THCTensor_(set)(state, self, src);
      THCTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);

      vector<i64> newSize(self->dim()-1);
      vector<i64> newStride(self->dim()-1);

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

    void THCTensor_(transpose)(THCState *state, THCTensor *self, THCTensor *src, int dimension1, int dimension2)
    {
      i64 z;

      if(!src)
        src = self;

      THArgCheck( (dimension1 >= 0) && (dimension1 < THTensor_nDimensionLegacyNoScalars(src)), 1, "out of range");
      THArgCheck( (dimension2 >= 0) && (dimension2 < THTensor_nDimensionLegacyNoScalars(src)), 2, "out of range");

      THCTensor_(set)(state, self, src);

      if(dimension1 == dimension2)
        return;

      z = self->stride(dimension1);
      self->set_stride(dimension1, self->stride(dimension2));
      self->set_stride(dimension2, z);
      z = self->size(dimension1);
      self->set_size(dimension1, self->size(dimension2));
      self->set_size(dimension2, z);
    }

    void THCTensor_(squeeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
    {
      THCTensor_squeeze1d(state, self, src, dimension);
    }

    void THCTensor_(unsqueeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension)
    {
      THCTensor_unsqueeze1d(state, self, src, dimension);
    }

    int THCTensor_(isContiguous)(THCState *state, const THCTensor *self)
    {
      return self->is_contiguous();
    }

    int THCTensor_(isSameSizeAs)(THCState *state, const THCTensor *self, const THCTensor* src)
    {
      int d;
      if (self->dim() != src->dim())
        return 0;
      for(d = 0; d < self->dim(); ++d)
      {
        if(self->size(d) != src->size(d))
          return 0;
      }
      return 1;
    }

    ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self)
    {
      return THCTensor_nElement(state, self);
    }

    void THCTensor_(retain)(THCState *state, THCTensor *self)
    {
      THCTensor_retain(state, self);
    }

    void THCTensor_(free)(THCState *state, THCTensor *self)
    {
      THCTensor_free(state, self);
    }

    void THCTensor_(freeCopyTo)(THCState *state, THCTensor *self, THCTensor *dst)
    {
      if(self != dst)
        THCTensor_(copy)(state, dst, self);

      THCTensor_(free)(state, self);
    }

    /*******************************************************************************/

    void THCTensor_(resizeNd)(THCState *state, THCTensor *self, int nDimension, const i64 *size, const i64 *stride)
    {
      THCTensor_resizeNd(state, self, nDimension, size, stride);
    }

    void THCTensor_(set0d)(THCState *state, THCTensor *tensor, Scalar value)
    {
      THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions");
      THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset(), value);
    }

    Scalar THCTensor_(get0d)(THCState *state, const THCTensor *tensor)
    {
      THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions dimension");
      return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset());
    }

    void THCTensor_(set1d)(THCState *state, THCTensor *tensor, i64 x0, Scalar value)
    {
      THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
      THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
      THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0), value);
    }

    Scalar THCTensor_(get1d)(THCState *state, const THCTensor *tensor, i64 x0)
    {
      THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
      THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
      return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0));
    }

    void THCTensor_(set2d)(THCState *state, THCTensor *tensor, i64 x0, i64 x1, Scalar value)
    {
      THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
      THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
      THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1), value);
    }

    Scalar THCTensor_(get2d)(THCState *state, const THCTensor *tensor, i64 x0, i64 x1)
    {
      THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
      THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
      return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1));
    }

    void THCTensor_(set3d)(THCState *state, THCTensor *tensor, i64 x0, i64 x1, i64 x2, Scalar value)
    {
      THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
      THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
      THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2), value);
    }

    Scalar THCTensor_(get3d)(THCState *state, const THCTensor *tensor, i64 x0, i64 x1, i64 x2)
    {
      THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
      THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
      return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2));
    }

    void THCTensor_(set4d)(THCState *state, THCTensor *tensor, i64 x0, i64 x1, i64 x2, i64 x3, Scalar value)
    {
      THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
      THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
      THCStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3), value);
    }

    Scalar THCTensor_(get4d)(THCState *state, const THCTensor *tensor, i64 x0, i64 x1, i64 x2, i64 x3)
    {
      THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
      THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
      return THCStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3));
    }

    int THCTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...)
    {
      int curDev = -1;
      THCudaCheck(cudaGetDevice(&curDev));
      va_list args;
      va_start(args, nTensors);
      int valid = 1;
      for (unsigned int i = 0; i < nTensors; i++) {
        THCTensor* tensor = va_arg(args, THCTensor*);
        if (tensor == NULL) {
          continue;
        }

        const int tensorDev = THCTensor_(getDevice)(state, tensor);

        // Skips CPU tensors
        if (tensorDev == -1) { continue; }

        // Checks all tensors are on the same device
        if (tensorDev != curDev) {
          valid = 0;
          break;
        }
      }

      va_end(args);
      return valid;
    }

    THCDescBuff THCTensor_(sizeDesc)(THCState *state, const THCTensor *tensor) {
      const int L = THC_DESC_BUFF_LEN;
      THCDescBuff buf;
      char *str = buf.str;
      int n = 0;
      n += snprintf(str, L-n, "[");
      int i;
      for(i = 0; i < tensor->dim(); i++) {
        if(n >= L) break;
        n += snprintf(str+n, L-n, "%" PRId64, tensor->size(i));
        if(i < tensor->dim()-1) {
          n += snprintf(str+n, L-n, " x ");
        }
      }
      if(n < L - 2) {
        snprintf(str+n, L-n, "]");
      } else {
        snprintf(str+L-5, 5, "...]");
      }
      return buf;
    }

    #endif
    */
}


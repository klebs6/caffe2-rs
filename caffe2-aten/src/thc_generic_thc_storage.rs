// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCStorage.h]
lazy_static!{
    /*
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCStorage.h"
    #else

    #define THCStorage THStorage

    // These used to be distinct types; for some measure of backwards compatibility and documentation
    // alias these to the single THCStorage type.
    #define THCudaStorage                       THCStorage
    #define THCudaDoubleStorage                 THCStorage
    #define THCudaHalfStorage                   THCStorage
    #define THCudaByteStorage                   THCStorage
    #define THCudaCharStorage                   THCStorage
    #define THCudaShortStorage                  THCStorage
    #define THCudaIntStorage                    THCStorage
    #define THCudaLongStorage                   THCStorage
    #define THCudaBoolStorage                   THCStorage
    #define THCudaBFloat16Storage               THCStorage
    #define THCudaComplexFloatStorage           THCStorage
    #define THCudaComplexDoubleStorage          THCStorage

    TORCH_CUDA_CU_API Scalar* THCStorage_(
        data)(THCState* state, const THCStorage*);
    TORCH_CUDA_CU_API int THCStorage_(elementSize)(THCState* state);

    /* slow access -- checks everything */
    TORCH_CUDA_CU_API void THCStorage_(
        set)(THCState* state, THCStorage*, ptrdiff_t, Scalar);
    TORCH_CUDA_CU_API Scalar
        THCStorage_(get)(THCState* state, const THCStorage*, ptrdiff_t);

    TORCH_CUDA_CU_API THCStorage* THCStorage_(new)(THCState* state);
    TORCH_CUDA_CU_API THCStorage* THCStorage_(
        newWithSize)(THCState* state, ptrdiff_t size);
    TORCH_CUDA_CU_API THCStorage* THCStorage_(
        newWithSize1)(THCState* state, Scalar);
    TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithMapping)(
        THCState* state,
        const char* filename,
        ptrdiff_t size,
        int shared);

    TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithAllocator)(
        THCState* state,
        ptrdiff_t size,
        Allocator* allocator);
    TORCH_CUDA_CU_API THCStorage* THCStorage_(newWithDataAndAllocator)(
        THCState* state,
        DataPtr&& data,
        ptrdiff_t size,
        Allocator* allocator);

    TORCH_CUDA_CU_API void THCStorage_(
        setFlag)(THCState* state, THCStorage* storage, const char flag);
    TORCH_CUDA_CU_API void THCStorage_(
        clearFlag)(THCState* state, THCStorage* storage, const char flag);
    TORCH_CUDA_CU_API void THCStorage_(
        retain)(THCState* state, THCStorage* storage);

    TORCH_CUDA_CU_API void THCStorage_(free)(THCState* state, THCStorage* storage);
    TORCH_CUDA_CU_API void THCStorage_(
        resizeBytes)(THCState* state, THCStorage* storage, ptrdiff_t size_bytes);
    TORCH_CUDA_CU_API void THCStorage_(
        fill)(THCState* state, THCStorage* storage, Scalar value);

    TORCH_CUDA_CU_API int THCStorage_(
        getDevice)(THCState* state, const THCStorage* storage);

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCStorage.cpp]
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCStorage.cpp"
    #else


    #ifdef __HIP_PLATFORM_HCC__

    #endif

    Scalar* THCStorage_(data)(THCState *state, const THCStorage *self)
    {
      return self->data<Scalar>();
    }

    int THCStorage_(elementSize)(THCState *state)
    {
      return sizeof(Scalar);
    }

    void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, Scalar value)
    {
      THArgCheck(
          (index >= 0) && (index < (self->nbytes() / sizeof(Scalar))),
          2,
          "index out of bounds");
      cudaStream_t stream = getCurrentCUDAStream();
    #if HIP_VERSION >= 301
      THCudaCheck(hipMemcpyWithStream(THCStorage_(data)(state, self) + index, &value, sizeof(Scalar),
                                      cudaMemcpyHostToDevice,
                                      stream));
    #else
      THCudaCheck(cudaMemcpyAsync(THCStorage_(data)(state, self) + index, &value, sizeof(Scalar),
                                  cudaMemcpyHostToDevice,
                                  stream));
      THCudaCheck(cudaStreamSynchronize(stream));
    #endif
    }

    Scalar THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
    {
      THArgCheck(
          (index >= 0) && (index < (self->nbytes() / sizeof(Scalar))),
          2,
          "index out of bounds");
      Scalar value;
      cudaStream_t stream = getCurrentCUDAStream();
    #if HIP_VERSION >= 301
      THCudaCheck(hipMemcpyWithStream(&value, THCStorage_(data)(state, self) + index, sizeof(Scalar),
                                      cudaMemcpyDeviceToHost, stream));
    #else
      THCudaCheck(cudaMemcpyAsync(&value, THCStorage_(data)(state, self) + index, sizeof(Scalar),
                                  cudaMemcpyDeviceToHost, stream));
      THCudaCheck(cudaStreamSynchronize(stream));
    #endif
      return value;
    }

    THCStorage* THCStorage_(new)(THCState *state)
    {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_size_t(),
                               0,
                               CUDACachingAllocator::get(),
                               true)
                               .release();
      return storage;
    }

    THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
    {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_size_t(),
                               size * sizeof(Scalar),
                               CUDACachingAllocator::get(),
                               true)
                               .release();
      return storage;
    }

    THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                              Allocator* allocator)
    {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_size_t(),
                               size * sizeof(Scalar),
                               allocator,
                               true)
                               .release();
      return storage;
    }

    THCStorage* THCStorage_(newWithSize1)(THCState *state, Scalar data0)
    {
      THCStorage *self = THCStorage_(newWithSize)(state, 1);
      THCStorage_(set)(state, self, 0, data0);
      return self;
    }

    THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
    {
      THError("not available yet for THCStorage");
      return NULL;
    }

    THCStorage* THCStorage_(newWithDataAndAllocator)(
        THCState* state,
        DataPtr&& data,
        ptrdiff_t size,
        Allocator* allocator) {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_size_t(),
                               size * sizeof(Scalar),
                               move(data),
                               allocator,
                               allocator != nullptr)
                               .release();
      return storage;
    }

    void THCStorage_(retain)(THCState *state, THCStorage *self)
    {
      THStorage_retain(self);
    }

    void THCStorage_(free)(THCState *state, THCStorage *self)
    {
      THStorage_free(self);
    }
    #endif
    */
}


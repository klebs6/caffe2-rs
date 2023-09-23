// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCStorageCopy.h]
lazy_static!{
    /*
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCStorageCopy.h"
    #else

    /* Support for copy between different Storage types */

    TORCH_CUDA_CU_API void THCStorage_(
        copy)(THCState* state, THCStorage* storage, THCStorage* src);
    #if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
    TORCH_CUDA_CU_API void THCStorage_(
        copyByte)(THCState* state, THCStorage* storage, struct THByteStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyChar)(THCState* state, THCStorage* storage, struct THCharStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyShort)(
        THCState* state,
        THCStorage* storage,
        struct THShortStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyInt)(THCState* state, THCStorage* storage, struct THIntStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyLong)(THCState* state, THCStorage* storage, struct THLongStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyFloat)(
        THCState* state,
        THCStorage* storage,
        struct THFloatStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyDouble)(
        THCState* state,
        THCStorage* storage,
        struct THDoubleStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyHalf)(THCState* state, THCStorage* storage, struct THHalfStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyBool)(THCState* state, THCStorage* storage, struct THBoolStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyBFloat16)(
        THCState* state,
        THCStorage* storage,
        struct THBFloat16Storage* src);
    #else
    TORCH_CUDA_CU_API void THCStorage_(copyComplexFloat)(
        THCState* state,
        THCStorage* storage,
        struct THComplexFloatStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyComplexDouble)(
        THCState* state,
        THCStorage* storage,
        struct THComplexDoubleStorage* src);
    #endif

    #if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
    TORCH_CUDA_CU_API void THCStorage_(copyCudaByte)(
        THCState* state,
        THCStorage* storage,
        struct THCudaByteStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaChar)(
        THCState* state,
        THCStorage* storage,
        struct THCudaCharStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaShort)(
        THCState* state,
        THCStorage* storage,
        struct THCudaShortStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaInt)(
        THCState* state,
        THCStorage* storage,
        struct THCudaIntStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaLong)(
        THCState* state,
        THCStorage* storage,
        struct THCudaLongStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaFloat)(
        THCState* state,
        THCStorage* storage,
        struct THCudaStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaDouble)(
        THCState* state,
        THCStorage* storage,
        struct THCudaDoubleStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaHalf)(
        THCState* state,
        THCStorage* storage,
        struct THCudaHalfStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaBool)(
        THCState* state,
        THCStorage* storage,
        struct THCudaBoolStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaBFloat16)(
        THCState* state,
        THCStorage* storage,
        struct THCudaBFloat16Storage* src);
    #else
    TORCH_CUDA_CU_API void THCStorage_(copyCudaComplexFloat)(
        THCState* state,
        THCStorage* storage,
        struct THCudaComplexFloatStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(copyCudaComplexDouble)(
        THCState* state,
        THCStorage* storage,
        struct THCudaComplexDoubleStorage* src);
    #endif

    #if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THByteStorage_copyCuda,
        Real)(THCState* state, THByteStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THCharStorage_copyCuda,
        Real)(THCState* state, THCharStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THShortStorage_copyCuda,
        Real)(THCState* state, THShortStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THIntStorage_copyCuda,
        Real)(THCState* state, THIntStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THLongStorage_copyCuda,
        Real)(THCState* state, THLongStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THFloatStorage_copyCuda,
        Real)(THCState* state, THFloatStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THDoubleStorage_copyCuda,
        Real)(THCState* state, THDoubleStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THHalfStorage_copyCuda,
        Real)(THCState* state, THHalfStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THBoolStorage_copyCuda,
        Real)(THCState* state, THBoolStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THBFloat16Storage_copyCuda,
        Real)(THCState* state, THBFloat16Storage* self, struct THCStorage* src);
    #else
    TORCH_CUDA_CU_API void TH_CONCAT_2(
        THComplexFloatStorage_copyCuda,
        Real)(THCState* state, THComplexFloatStorage* self, struct THCStorage* src);
    TORCH_CUDA_CU_API void TH_CONCAT_2(THComplexDoubleStorage_copyCuda, Real)(
        THCState* state,
        THComplexDoubleStorage* self,
        struct THCStorage* src);
    #endif

    TORCH_CUDA_CU_API void THStorage_(
        copyCuda)(THCState* state, THStorage* self, THCStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyCuda)(THCState* state, THCStorage* self, THCStorage* src);
    TORCH_CUDA_CU_API void THCStorage_(
        copyCPU)(THCState* state, THCStorage* self, THStorage* src);

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/THC/generic/THCStorageCopy.cpp]
    #ifndef THC_GENERIC_FILE
    #define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cpp"
    #else

    #ifdef __HIP_PLATFORM_HCC__

    #endif

    void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
    {
      THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
      cudaStream_t stream = getCurrentCUDAStream();
    #if HIP_VERSION >= 301
      THCudaCheck(hipMemcpyWithStream(
          THCStorage_(data)(state, self),
          THStorage_(data)(src),
          self->nbytes(),
          cudaMemcpyHostToDevice,
          stream));
    #else
      THCudaCheck(cudaMemcpyAsync(
          THCStorage_(data)(state, self),
          THStorage_(data)(src),
          self->nbytes(),
          cudaMemcpyHostToDevice,
          stream));
      THCudaCheck(cudaStreamSynchronize(stream));
    #endif
    }

    #define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                                 \
      void THCStorage_(copy##TYPEC)(                                              \
          THCState * state, THCStorage * self, struct TH##TYPEC##Storage * src) { \
        THCTensor* selfTensor = THCTensor_(newWithStorage1d)(                     \
            state, self, 0, src->nbytes() / sizeof(Scalar), 1);                 \
        struct TH##TYPEC##Tensor* srcTensor = TH##TYPEC##Tensor_newWithStorage1d( \
            src, 0, src->nbytes() / sizeof(Scalar), 1);                         \
        THCTensor_(copy)(state, selfTensor, srcTensor);                           \
        TH##TYPEC##Tensor_free(srcTensor);                                        \
        THCTensor_(free)(state, selfTensor);                                      \
      }

    // TODO: Add cross-dtype storage copy for complex storage
    #if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Float)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Half)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(Bool)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(BFloat16)
    #else
      TH_CUDA_STORAGE_IMPLEMENT_COPY(ComplexFloat)
      TH_CUDA_STORAGE_IMPLEMENT_COPY(ComplexDouble)
    #endif

    void THStorage_(copyCuda)(THCState *state, THStorage *self, struct THCStorage *src)
    {
      THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
      cudaStream_t stream = getCurrentCUDAStream();
    #if HIP_VERSION >= 301
      THCudaCheck(hipMemcpyWithStream(
          THStorage_(data)(self),
          THCStorage_(data)(state, src),
          self->nbytes(),
          cudaMemcpyDeviceToHost,
          stream));
    #else
      THCudaCheck(cudaMemcpyAsync(
          THStorage_(data)(self),
          THCStorage_(data)(state, src),
          self->nbytes(),
          cudaMemcpyDeviceToHost,
          stream));
      THCudaCheck(cudaStreamSynchronize(stream));
    #endif
    }

    #define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                               \
      void TH_CONCAT_4(TH, TYPEC, Storage_copyCuda, Real)(                        \
          THCState * state, TH##TYPEC##Storage * self, struct THCStorage * src) { \
        TH##TYPEC##Tensor* selfTensor = TH##TYPEC##Tensor_newWithStorage1d(       \
            self, 0, self->nbytes() / sizeof(Scalar), 1);                       \
        struct THCTensor* srcTensor = THCTensor_(newWithStorage1d)(               \
            state, src, 0, src->nbytes() / sizeof(Scalar), 1);                  \
        THCTensor_(copy)(state, selfTensor, srcTensor);                           \
        THCTensor_(free)(state, srcTensor);                                       \
        TH##TYPEC##Tensor_free(selfTensor);                                       \
      }

    // TODO: Add cross-dtype storage copy for complex storage
    #if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Float)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Half)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Bool)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(BFloat16)
    #else
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(ComplexFloat)
      TH_CUDA_STORAGE_IMPLEMENT_COPYTO(ComplexDouble)
    #endif

    #undef TH_CUDA_STORAGE_IMPLEMENT_COPY
    #undef TH_CUDA_STORAGE_IMPLEMENT_COPYTO

    #endif
    */
}


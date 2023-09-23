crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/MiscUtils.h]

/**
  | RAII for a MAGMA Queue
  |
  | Default constructor without a device will cause
  |   destroying a queue which has not been
  |   initialized.
  */
#[cfg(USE_MAGMA)]
pub struct MAGMAQueue {
    magma_queue: MagmaQueue,

    #[cfg(CUDA_VERSION_GTE_11000)]
    original_math_mode: CuBlasMath,
}

#[cfg(USE_MAGMA)]
impl Drop for MAGMAQueue {
    fn drop(&mut self) {
        todo!();
        /*
            #if CUDA_VERSION >= 11000
        // We've manually set the math mode to CUBLAS_DEFAULT_MATH, now we
        // should restore the original math mode back
        cublasHandle_t handle = magma_queue_get_cublas_handle(magma_queue_);
        cublasSetMathMode(handle, original_math_mode);
    #endif
        magma_queue_destroy(magma_queue_);
        */
    }
}

#[cfg(USE_MAGMA)]
impl MAGMAQueue {

    pub fn new(device_id: i64) -> Self {
    
        todo!();
        /*


            cublasHandle_t handle = getCurrentCUDABlasHandle();
    #if CUDA_VERSION >= 11000
        // Magma operations is numerically sensitive, so TF32 should be off
        // regardless of the global flag.
        TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_math_mode));
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    #endif
        magma_queue_create_from_cuda(
          device_id,
          getCurrentCUDAStream(),
          handle,
          getCurrentCUDASparseHandle(),
          &magma_queue_);
        */
    }
    
    pub fn get_queue(&self) -> MagmaQueue {
        
        todo!();
        /*
            return magma_queue_;
        */
    }
}

#[cfg(USE_MAGMA)]
#[inline] pub fn magma_int_cast(
        value:   i64,
        varname: *const u8) -> MagmaInt {
    
    todo!();
        /*
            auto result = static_cast<magma_int_t>(value);
      if (static_cast<i64>(result) != value) {
        AT_ERROR("magma: The value of ", varname, "(", (long long)value,
                 ") is too large to fit into a magma_int_t (", sizeof(magma_int_t), " bytes)");
      }
      return result;
        */
}

/**
  | MAGMA functions that don't take a magma_queue_t
  | aren't stream safe
  |
  | Work around this by synchronizing with the
  | default stream
  */
#[cfg(USE_MAGMA)]
pub struct MagmaStreamSyncGuard {

}

#[cfg(USE_MAGMA)]
impl Default for MagmaStreamSyncGuard {
    
    fn default() -> Self {
        todo!();
        /*


            auto stream = getCurrentCUDAStream();
        if (stream != getDefaultCUDAStream()) {
          AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        */
    }
}

#[cfg(USE_MAGMA)]
impl Drop for MagmaStreamSyncGuard {
    fn drop(&mut self) {
        todo!();
        /*
            auto default_stream = getDefaultCUDAStream();
        if (getCurrentCUDAStream() != default_stream) {
          AT_CUDA_CHECK(cudaStreamSynchronize(default_stream));
        }
        */
    }
}

#[inline] pub fn cuda_int_cast(
        value:   i64,
        varname: *const u8) -> i32 {
    
    todo!();
        /*
            auto result = static_cast<int>(value);
      TORCH_CHECK(static_cast<i64>(result) == value,
                  "cuda_int_cast: The value of ", varname, "(", (long long)value,
                  ") is too large to fit into a int (", sizeof(int), " bytes)");
      return result;
        */
}

/**
  | Creates an array of size elements of
  | type T, backed by pinned memory wrapped
  | in a Storage
  |
  */
#[inline] pub fn pin_memory<T>(size: i64) -> Storage {

    todo!();
        /*
            auto* allocator = getPinnedMemoryAllocator();
      i64 adjusted_size = size * sizeof(T);
      return Storage(
          Storage::use_byte_size_t(),
          adjusted_size,
          allocator,
          /*resizable=*/false);
        */
}

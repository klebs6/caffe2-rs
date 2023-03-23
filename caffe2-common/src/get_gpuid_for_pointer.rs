crate::ix!();

/**
  | Gets the GPU id that the current pointer
  | is located at.
  |
  */
#[inline] pub fn get_gpuid_for_pointer(ptr: *const c_void) -> i32 {
    
    todo!();
    /*
        cudaPointerAttributes attr;
      cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

      if (err == cudaErrorInvalidValue) {
        // Occurs when the pointer is in the CPU address space that is
        // unmanaged by CUDA; make sure the last error state is cleared,
        // since it is persistent
        err = cudaGetLastError();
        CHECK(err == cudaErrorInvalidValue);
        return -1;
      }

      // Otherwise, there must be no error
      CUDA_ENFORCE(err);

      if (attr.CAFFE2_CUDA_PTRATTR_MEMTYPE == cudaMemoryTypeHost) {
        return -1;
      }

      return attr.device;
    */
}

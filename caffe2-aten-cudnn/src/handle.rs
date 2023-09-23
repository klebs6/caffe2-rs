crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Handle.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cudnn/Handle.cpp]

pub fn create_cu_dnn_handle(handle: *mut CuDnnHandle)  {
    
    todo!();
        /*
            AT_CUDNN_CHECK(cudnnCreate(handle));
        */
}

pub fn destroy_cu_dnn_handle(handle: CuDnnHandle)  {
    
    todo!();
        /*
            // this is because of something dumb in the ordering of
    // destruction. Sometimes atexit, the cuda context (or something)
    // would already be destroyed by the time this gets destroyed. It
    // happens in fbcode setting. @colesbury and I decided to not destroy
    // the handle as a workaround.
    //   - @soumith
    //
    // Further note: this is now disabled globally, because we are seeing
    // the same issue as mentioned above in CUDA 11 CI.
    //   - @zasdfgbnm
    //
    // #ifdef NO_CUDNN_DESTROY_HANDLE
    // #else
    //   cudnnDestroy(handle);
    // #endif
        */
}

pub type CudnnPoolType = DeviceThreadHandlePool<CuDnnHandle,CreateCudnnHandle,DestroyCudnnHandle>;

pub fn get_cudnn_handle() -> CuDnnHandle {
    
    todo!();
        /*
            int device;
      AT_CUDA_CHECK(cudaGetDevice(&device));

      // Thread local PoolWindows are lazily-initialized
      // to avoid initialization issues that caused hangs on Windows.
      // See: https://github.com/pytorch/pytorch/pull/22405
      // This thread local unique_ptrs will be destroyed when the thread terminates,
      // releasing its reserved handles back to the pool.
      static auto pool = make_shared<CudnnPoolType>();
      thread_local unique_ptr<CudnnPoolType::PoolWindow> myPoolWindow(
          pool->newPoolWindow());

      auto handle = myPoolWindow->reserve(device);
      AT_CUDNN_CHECK(cudnnSetStream(handle, getCurrentCUDAStream()));
      return handle;
        */
}

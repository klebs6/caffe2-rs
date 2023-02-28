crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CusolverDnHandlePool.cpp]

#[cfg(CUDART_VERSION)]
pub fn create_cusolver_dn_handle(handle: *mut CuSolverDnHandle)  {
    
    todo!();
        /*
            TORCH_CUSOLVER_CHECK(cusolverDnCreate(handle));
        */
}

#[cfg(CUDART_VERSION)]
pub fn destroy_cusolver_dn_handle(handle: CuSolverDnHandle)  {
    
    todo!();
        /*
            // this is because of something dumb in the ordering of
    // destruction. Sometimes atexit, the cuda context (or something)
    // would already be destroyed by the time this gets destroyed. It
    // happens in fbcode setting. @colesbury and @soumith decided to not destroy
    // the handle as a workaround.
    //   - Comments of @soumith copied from cuDNN handle pool implementation
    #ifdef NO_CUDNN_DESTROY_HANDLE
    #else
        cusolverDnDestroy(handle);
    #endif
        */
}

#[cfg(CUDART_VERSION)]
pub type CuSolverDnPoolType = DeviceThreadHandlePool<CuSolverDnHandle,CreateCuSolverDnHandle,DestroyCuSolverDnHandle>;

#[cfg(CUDART_VERSION)]
pub fn get_current_cuda_solver_dn_handle() -> CuSolverDnHandle {
    
    todo!();
        /*
            int device;
      AT_CUDA_CHECK(cudaGetDevice(&device));

      // Thread local PoolWindows are lazily-initialized
      // to avoid initialization issues that caused hangs on Windows.
      // See: https://github.com/pytorch/pytorch/pull/22405
      // This thread local unique_ptrs will be destroyed when the thread terminates,
      // releasing its reserved handles back to the pool.
      static auto pool = std::make_shared<CuSolverDnPoolType>();
      thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
          pool->newPoolWindow());

      auto handle = myPoolWindow->reserve(device);
      auto stream = c10::cuda::getCurrentCUDAStream();
      TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
      return handle;
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CuSparseHandlePool.cpp]

pub fn create_cusparse_handle(handle: *mut CuSparseHandle)  {
    
    todo!();
        /*
            TORCH_CUDASPARSE_CHECK(cusparseCreate(handle));
        */
}

pub fn destroy_cusparse_handle(handle: CuSparseHandle)  {
    
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
        cusparseDestroy(handle);
    #endif
        */
}

pub type CuSparsePoolType = DeviceThreadHandlePool<CuSparseHandle,CreateCuSparseHandle,DestroyCuSparseHandle>;

pub fn get_current_cuda_sparse_handle() -> CuSparseHandle {
    
    todo!();
        /*
            int device;
      AT_CUDA_CHECK(cudaGetDevice(&device));

      // Thread local PoolWindows are lazily-initialized
      // to avoid initialization issues that caused hangs on Windows.
      // See: https://github.com/pytorch/pytorch/pull/22405
      // This thread local unique_ptrs will be destroyed when the thread terminates,
      // releasing its reserved handles back to the pool.
      static auto pool = make_shared<CuSparsePoolType>();
      thread_local unique_ptr<CuSparsePoolType::PoolWindow> myPoolWindow(
          pool->newPoolWindow());

      auto handle = myPoolWindow->reserve(device);
      cusparseSetStream(handle, cuda::getCurrentCUDAStream());
      return handle;
        */
}

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Handle.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/miopen/Handle.cpp]

pub fn create_mio_pen_handle(handle: *mut MiOpenHandle)  {
    
    todo!();
        /*
            MIOPEN_CHECK(miopenCreate(handle));
        */
}

pub fn destroy_mio_pen_handle(handle: MiOpenHandle)  {
    
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
    // #ifdef NO_MIOPEN_DESTROY_HANDLE
    // #else
    //   miopenDestroy(handle);
    // #endif
        */
}

pub type MIOpenPoolType = DeviceThreadHandlePool<MiOpenHandle,CreateMIOpenHandle,DestroyMIOpenHandle>;

pub fn get_miopen_handle() -> MiOpenHandle {
    
    todo!();
        /*
            int device;
      HIP_CHECK(hipGetDevice(&device));

      // Thread local PoolWindows are lazily-initialized
      // to avoid initialization issues that caused hangs on Windows.
      // See: https://github.com/pytorch/pytorch/pull/22405
      // This thread local unique_ptrs will be destroyed when the thread terminates,
      // releasing its reserved handles back to the pool.
      static auto pool = make_shared<MIOpenPoolType>();
      thread_local unique_ptr<MIOpenPoolType::PoolWindow> myPoolWindow(
          pool->newPoolWindow());

      auto handle = myPoolWindow->reserve(device);
      MIOPEN_CHECK(miopenSetStream(handle, hip::getCurrentHIPStream()));
      return handle;
        */
}

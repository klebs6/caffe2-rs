crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelThreadPoolNative.cpp]

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub const NOT_SET:  i32 = -1;

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub const CONSUMED: i32 = -2;

/**
  | Number of inter-op threads set by the user;
  |
  | NOT_SET -> positive value -> CONSUMED
  | (CONSUMED - thread pool is initialized)
  | or
  | NOT_SET -> CONSUMED
  */
#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
lazy_static!{
    /*
    atomic<int> num_interop_threads{NOT_SET};
    */
}

/**
  | thread pool global instance is hidden,
  | users should use launch and get/set_num_interop_threads
  | interface
  |
  */
#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn get_pool() -> &mut TaskThreadPoolBase {
    
    todo!();
        /*
            static shared_ptr<TaskThreadPoolBase> pool =
          ThreadPoolRegistry()->Create(
              "C10",
              /* device_id */ 0,
              /* pool_size */ num_interop_threads.exchange(CONSUMED),
              /* create_new */ true);
      return *pool;
        */
}

/**
  | Factory function for ThreadPoolRegistry
  |
  */
#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn create_c10_threadpool(
        device_id:  i32,
        pool_size:  i32,
        create_new: bool) -> Arc<TaskThreadPoolBase> {
    
    todo!();
        /*
            // For now, the only accepted device id is 0
      TORCH_CHECK(device_id == 0);
      // Create new thread pool
      TORCH_CHECK(create_new);
      return make_shared<PTThreadPool>(pool_size);
        */
}

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
C10_REGISTER_CREATOR!(ThreadPoolRegistry, C10, create_c10_threadpool);

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn set_num_interop_threads(nthreads: i32)  {
    
    todo!();
        /*
            TORCH_CHECK(nthreads > 0, "Expected positive number of threads");

      int no_value = NOT_SET;
      TORCH_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
          "Error: cannot set number of interop threads after parallel work "
          "has started or set_num_interop_threads called");
        */
}

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn get_num_interop_threads() -> i32 {
    
    todo!();
        /*
            int nthreads = num_interop_threads.load();
      if (nthreads > 0) {
        return nthreads;
      } else if (nthreads == NOT_SET) {
        // return default value
        return TaskThreadPoolBase::defaultNumThreads();
      } else {
        return get_pool().size();
      }
        */
}

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn launch_no_thread_state(fn_: fn() -> ())  {
    
    todo!();
        /*
            #if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
      intraop_launch(move(fn));
    #else
      get_pool().run(move(fn));
    #endif
        */
}

#[cfg(any(AT_PARALLEL_OPENMP,AT_PARALLEL_NATIVE,AT_PARALLEL_NATIVE_TBB))]
pub fn launch(func: fn() -> ())  {
    
    todo!();
        /*
      internal::launch_no_thread_state(bind([](
        function<void()> f, ThreadLocalState thread_locals) {
          ThreadLocalStateGuard guard(move(thread_locals));
          f();
        },
        move(func),
        ThreadLocalState()
      ));
        */
}

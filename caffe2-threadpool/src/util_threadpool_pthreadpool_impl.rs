crate::ix!();


// External API
#[inline] pub fn legacy_pthreadpool_compute_1d(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_1d_t,
    argument:   *mut c_void,
    range:      usize)  {
    
    todo!();
    /*
        if (threadpool == nullptr) {
        /* No thread pool provided: execute function sequentially on the calling
         * thread */
        for (size_t i = 0; i < range; i++) {
          function(argument, i);
        }
        return;
      }
    #ifdef USE_PTHREADPOOL
      if (caffe2::using_new_threadpool) {
        pthreadpool_parallelize_1d(threadpool, function, argument, range, 0u);
      } else {
        reinterpret_cast<caffe2::ThreadPool*>(threadpool)
            ->run(
                [function, argument](int threadId, size_t workId) {
                  function(argument, workId);
                },
                range);
      }
    #else
      reinterpret_cast<caffe2::ThreadPool*>(threadpool)
          ->run(
              [function, argument](int threadId, size_t workId) {
                function(argument, workId);
              },
              range);
    #endif
    */
}

#[inline] pub fn legacy_pthreadpool_parallelize_1d(
    threadpool: legacy_pthreadpool_t,
    function:   legacy_pthreadpool_function_1d_t,
    argument:   *const c_void,
    range:      usize,
    unused_0:   u32)  {
    
    todo!();
    /*
        legacy_pthreadpool_compute_1d(threadpool, function, argument, range);
    */
}

#[inline] pub fn legacy_pthreadpool_get_threads_count(threadpool: legacy_pthreadpool_t) -> usize {
    
    todo!();
    /*
        // The current fix only useful when XNNPACK calls legacy_pthreadpool_get_threads_count with nullptr.
      if (threadpool == nullptr) {
        return 1;
      }
      return reinterpret_cast<caffe2::ThreadPool*>(threadpool)->getNumThreads();
    */
}

#[inline] pub fn legacy_pthreadpool_create(threads_count: usize) -> legacy_pthreadpool_t {
    
    todo!();
    /*
        std::mutex thread_pool_creation_mutex_;
      std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);

      return reinterpret_cast<legacy_pthreadpool_t>(new caffe2::ThreadPool(threads_count));
    */
}

#[inline] pub fn legacy_pthreadpool_destroy(pthreadpool: legacy_pthreadpool_t)  {
    
    todo!();
    /*
        if (pthreadpool) {
        caffe2::ThreadPool* threadpool =
            reinterpret_cast<caffe2::ThreadPool*>(pthreadpool);
        delete threadpool;
      }
    */
}


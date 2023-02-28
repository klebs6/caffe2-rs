crate::ix!();

pub struct PThreadPool {
    mutex:      parking_lot::RawMutex,
    threadpool: Box<threadpool::ThreadPool>,
}

/**
  | Exposes the underlying implementation of
  | PThreadPool.
  |
  | Only for use in external libraries so as to
  | unify threading across internal (i.e. ATen,
  | etc.) and external (e.g. NNPACK, QNNPACK,
  | XNNPACK) use cases.
  */
#[inline] pub fn pthreadpool() -> threadpool::ThreadPool {
    
    todo!();
    /*
    
    */
}

impl PThreadPool {
    
    pub fn new(thread_count: usize) -> Self {
        todo!();
        /*
            : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy)
        */
    }
    
    #[inline] pub fn get_thread_count(&self) -> usize {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock{mutex_};

      TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
      return pthreadpool_get_threads_count(threadpool_.get());
        */
    }
    
    #[inline] pub fn set_thread_count(&mut self, thread_count: usize)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock{mutex_};

      // As it stands, pthreadpool is an entirely data parallel framework with no
      // support for task parallelism.  Hence, all functions are blocking, and no
      // user-provided tasks can be in flight when the control is returned to the
      // user of the API, which means re-initializing the library, without the
      // need to wait on any pending tasks, is all one needs to do to re-adjust
      // the thread count.
      threadpool_.reset(pthreadpool_create(thread_count));
        */
    }
    
    /**
      | Run, in parallel, function fn(task_id)
      | over task_id in range [0, range).
      |
      | This function is blocking.  All input is
      | processed by the time it returns.
      */
    #[inline] pub fn run(&mut self, f: fn(x: usize) -> (), range: usize)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock{mutex_};

      TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");

      struct Context final {
        const std::function<void(size_t)>& fn;
      } context{
          fn,
      };

      pthreadpool_parallelize_1d(
          threadpool_.get(),
          // Note: pthreadpool_parallelize_1d() is a blocking function.  The
          // function pointer to this lambda passed on to
          // pthreadpool_parallelize_1d() cannot go out of scope until
          // pthreadpool_parallelize_1d() returns.
          [](void* const context, const size_t item) {
            reinterpret_cast<Context*>(context)->fn(item);
          },
          &context,
          range,
          0u);
        */
    }
}

#[inline] pub fn get_default_num_threads() -> usize {
    
    todo!();
    /*
    
    */
}

/**
  | Return a singleton instance of PThreadPool
  | for ATen/TH multithreading.
  |
  */
#[inline] pub fn pthreadpool() -> *mut PThreadPool {
    
    todo!();
    /*
        static std::unique_ptr<PThreadPool> threadpool =
          std::make_unique<PThreadPool>(getDefaultNumThreads());
      return threadpool.get();
    */
}

#[inline] pub fn pthreadpool() -> threadpool::ThreadPool {
    
    todo!();
    /*
        if (caffe2::_NoPThreadPoolGuard::is_enabled()) {
        return nullptr;
      }
      PThreadPool* const threadpool = pthreadpool();
      TORCH_INTERNAL_ASSERT(
          threadpool, "Failed to acquire an instance of PThreadPool!");
      return threadpool->threadpool_.get();
    */
}

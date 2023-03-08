crate::ix!();

impl<Context> PrefetchOperator<Context> {
    
    #[inline] pub fn prefetch_worker(&mut self)  {
        
        todo!();
        /*
            context_.SwitchToDevice();
        std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
        while (prefetched_)
          producer_.wait(lock);
        while (!finalize_) {
          // We will need to run a FinishDeviceComputation() call because the
          // prefetcher thread and the main thread are potentially using different
          // streams (like on GPU).
          try {
            prefetch_success_ = Prefetch();
            context_.FinishDeviceComputation();
          } catch (const std::exception& e) {
            // TODO: propagate exception_ptr to the caller side
            LOG(ERROR) << "Prefetching error " << e.what();
            prefetch_success_ = false;
          }
          prefetched_ = true;
          consumer_.notify_one();
          while (prefetched_)
            producer_.wait(lock);
        }
        */
    }
}

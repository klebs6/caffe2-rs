crate::ix!();

impl<Context> PrefetchOperator<Context> {

    #[inline] pub fn finalize(&mut self)  {
        
        todo!();
        /*
            if (prefetch_thread_.get()) {
          {
            std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
            while (!prefetched_)
              consumer_.wait(lock);
            finalize_ = true;
            prefetched_ = false;
          }
          producer_.notify_one();
          prefetch_thread_->join();
          prefetch_thread_.reset();
        } else {
          // If we never initialized the prefetch thread, just set
          // finalize anyway.
          finalize_ = true;
        }
        */
    }
}

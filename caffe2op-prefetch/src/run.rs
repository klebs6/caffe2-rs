crate::ix!();

impl<Context> PrefetchOperator<Context> {

    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            if (no_prefetch_) {
          context_.SwitchToDevice();
          bool result = Prefetch() && CopyPrefetched();
          context_.FinishDeviceComputation();
          return result;
        }
        // Note(jiayq): We only start the prefetch_thread at the Run() function
        // instead of in the constructor, because the prefetch_thread needs to start
        // after all derived classes' constructors finish.
        if (!prefetch_thread_) {
          prefetch_thread_.reset(
              new std::thread([this] { this->PrefetchWorker(); }));
        }
        context_.SwitchToDevice();
        std::unique_lock<std::mutex> lock(prefetch_access_mutex_);
        while (!prefetched_)
          consumer_.wait(lock);
        if (!prefetch_success_) {
          LOG(ERROR) << "Prefetching failed.";
          return false;
        }
        if (!CopyPrefetched()) {
          LOG(ERROR) << "Error when copying prefetched data.";
          return false;
        }
        prefetched_ = false;
        context_.FinishDeviceComputation();
        producer_.notify_one();
        return true;
        */
    }
}

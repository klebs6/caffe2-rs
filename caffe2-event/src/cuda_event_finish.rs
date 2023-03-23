crate::ix!();

#[inline] pub fn event_finishCUDA(event: *const Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
        while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
          wrapper->cv_recorded_.wait(lock);
        }
      }

      if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
        // ok, even if event is already completed and status was not yet updated
        CUDAGuard g(wrapper->device_id_);
        auto cudaResult = cudaEventSynchronize(wrapper->cuda_event_);
        if (cudaResult == cudaSuccess) {
          wrapper->status_ = EventStatus::EVENT_SUCCESS;
        } else {
          const auto& err_msg = cudaGetErrorString(cudaResult);

          std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
    */
}

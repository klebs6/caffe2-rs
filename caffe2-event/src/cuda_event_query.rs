crate::ix!();

#[inline] pub fn event_queryCUDA(event: *const Event) -> EventStatus {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
        auto cudaResult = cudaEventQuery(wrapper->cuda_event_);
        if (cudaResult == cudaSuccess) {
          wrapper->status_ = EventStatus::EVENT_SUCCESS;
        } else if (cudaResult != cudaErrorNotReady) {
          const auto& err_msg = cudaGetErrorString(cudaResult);

          std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
      return static_cast<EventStatus>(wrapper->status_.load());
    */
}

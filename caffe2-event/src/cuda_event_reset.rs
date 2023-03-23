crate::ix!();

#[inline] pub fn event_resetCUDA(event: *mut Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->status_ = EventStatus::EVENT_INITIALIZED;
      wrapper->err_msg_ = "";
      wrapper->cuda_stream_ = nullptr;
    */
}

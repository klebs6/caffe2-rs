crate::ix!();

#[inline] pub fn event_set_finishedCUDA(event: *const Event, err_msg: *const u8)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

        CAFFE_ENFORCE_EQ(
            wrapper->status_,
            EventStatus::EVENT_INITIALIZED,
            "Calling SetFinished on recorded CUDA event");

        if (!err_msg) {
          wrapper->status_ = EventStatus::EVENT_SUCCESS;
        } else {
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
      wrapper->cv_recorded_.notify_all();
    */
}

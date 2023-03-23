crate::ix!();

#[inline] pub fn event_finishcpu(event: *const Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);
      while (wrapper->status_ != EventStatus::EVENT_SUCCESS &&
             wrapper->status_ != EventStatus::EVENT_FAILED) {
        wrapper->cv_completed_.wait(lock);
      }
    */
}

#[inline] pub fn event_set_finishedcpu(event: *const Event, err_msg: *const u8)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      if (wrapper->status_ == EventStatus::EVENT_FAILED) {
        LOG(WARNING) << "SetFinished called on a finished event. "
                     << "Most likely caused by an external cancellation. "
                     << "old message: " << wrapper->err_msg_ << ", "
                     << "new message: " << err_msg;
        return;
      }

      CAFFE_ENFORCE(
          wrapper->status_ == EventStatus::EVENT_INITIALIZED ||
              wrapper->status_ == EventStatus::EVENT_SCHEDULED,
          "Calling SetFinished on finished event");

      if (!err_msg) {
        wrapper->status_ = EventStatus::EVENT_SUCCESS;
      } else {
        wrapper->err_msg_ = err_msg;
        wrapper->status_ = EventStatus::EVENT_FAILED;
      }

      for (auto& callback : wrapper->callbacks_) {
        callback();
      }

      wrapper->cv_completed_.notify_all();
    */
}

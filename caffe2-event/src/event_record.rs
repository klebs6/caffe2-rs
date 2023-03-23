crate::ix!();

#[inline] pub fn event_recordcpu(
    event:   *mut Event,
    unused:  *const c_void,
    err_msg: *const u8) {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      // Possible state changes:
      //  INITIALIZED -> SCHEDULED or SUCCESS/FAILED
      //  SCHEDULED -> SUCCESS/FAILED
      //  SUCCESS/FAILED - terminal, no further changes to status_/err_msg_

      CAFFE_ENFORCE(
          wrapper->status_ != EventStatus::EVENT_SCHEDULED,
          "Calling Record multiple times");

      // Event might be in SUCCESS/FAILED state in case an op has
      // finished async execution part first
      if (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
        if (!err_msg) {
          wrapper->status_ = EventStatus::EVENT_SCHEDULED;
        } else {
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
          wrapper->cv_completed_.notify_all();
        }
      }
    */
}

crate::ix!();

#[inline] pub fn event_set_callbackcpu(event: *mut Event, callback: EventCallbackFunction)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      wrapper->callbacks_.push_back(callback);
      if (wrapper->status_ == EventStatus::EVENT_SUCCESS ||
          wrapper->status_ == EventStatus::EVENT_FAILED) {
        callback();
      }
    */
}

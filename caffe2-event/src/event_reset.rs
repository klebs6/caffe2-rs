crate::ix!();

#[inline] pub fn event_resetcpu(event: *mut Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);
      wrapper->status_ = EventStatus::EVENT_INITIALIZED;
      wrapper->err_msg_ = "";
      wrapper->callbacks_.clear();
    */
}

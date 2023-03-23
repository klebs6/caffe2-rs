crate::ix!();

pub const kNoError: &'static str = "No error";

#[inline] pub fn event_error_messagecpu<'a>(event: *const Event) -> &'a String {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      if (wrapper->status_ == EventStatus::EVENT_FAILED) {
        // Failed is a terminal state, not synchronizing,
        // err_msg_ should not be changed anymore
        return wrapper->err_msg_;
      } else {
        return kNoError;
      }
    */
}

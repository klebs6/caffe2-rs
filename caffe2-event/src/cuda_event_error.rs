crate::ix!();

pub const kNoError: &'static str = "No error";

#[inline] pub fn event_error_messageCUDA<'a>(event: *const Event) -> &'a String {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      // supposed to be called after EventQueryCUDA to update status first
      if (wrapper->status_ == EventStatus::EVENT_FAILED) {
        return wrapper->err_msg_;
      } else {
        return kNoError;
      }
    */
}

crate::ix!();

#[inline] pub fn event_recordCUDA(
    event:   *mut Event,
    context: *const c_void,
    err_msg: *const u8)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

        // Possible state changes:
        //  INITIALIZED -> SCHEDULED/FAILED
        //  SCHEDULED -> SUCCESS/FAILED
        //  SUCCESS/FAILED - terminal
        //
        // No further changes to cuda_event_ and cuda_stream_ after transitioning
        // from INITIALIZED
        // No further changes to err_msg_ after transitioning into FAILED

        CAFFE_ENFORCE_EQ(
            wrapper->status_,
            EventStatus::EVENT_INITIALIZED,
            "Calling Record multiple times");

        if (!err_msg) {
          // When recording, one needs to make sure that the current gpu id is
          // correct.
          // TODO(jiayq): move the enforce logic to the caller?
          const auto& current_device = CaffeCudaGetDevice();
          CAFFE_ENFORCE_EQ(
              current_device,
              wrapper->device_id_,
              "When you call EventRecordCUDA, your current device should be the same "
              "as the device specified by the event.");
          CAFFE_ENFORCE_EQ(
              current_device,
              static_cast<const CUDAContext*>(context)->device_id());
          CUDA_ENFORCE(cudaEventRecord(
              wrapper->cuda_event_,
              static_cast<const CUDAContext*>(context)->cuda_stream()));
          wrapper->cuda_stream_ =
              static_cast<const CUDAContext*>(context)->cuda_stream();
          wrapper->status_ = EventStatus::EVENT_SCHEDULED;
        } else {
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
      wrapper->cv_recorded_.notify_all();
    */
}

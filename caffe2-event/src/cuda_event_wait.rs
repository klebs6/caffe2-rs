crate::ix!();

/// Both waiter and event are CUDA. Non-blocking
#[inline] pub fn event_waitCUDACUDA(event: *const Event, context: *mut c_void)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
        while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
          wrapper->cv_recorded_.wait(lock);
        }
      }

      if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
        // ok, even if event is already completed and status was not yet updated
        auto context_stream = static_cast<CUDAContext*>(context)->cuda_stream();
        auto event_stream = wrapper->cuda_stream_;
        if (context_stream != event_stream) {
          // CAFFE_ENFORCE_EQ(
          //    CaffeCudaGetDevice(),
          //    static_cast<const CUDAContext*>(context)->device_id());
          CUDA_CHECK(cudaStreamWaitEvent(context_stream, wrapper->cuda_event_, 0));
        }
      }
    */
}

/// Waiter is CPU, event is CUDA
#[inline] pub fn event_waitCPUCUDA(event: *const Event, context: *mut c_void)  {
    
    todo!();
    /*
        EventFinishCUDA(event);
    */
}

/// Waiter is CUDA, event is CPU
#[inline] pub fn event_waitCUDACPU(event: *const Event, context: *mut c_void)  {
    
    todo!();
    /*
        event->Finish(); // calls EventFinishCPU
    */
}

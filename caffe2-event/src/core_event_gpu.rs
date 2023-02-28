crate::ix!();

pub struct CudaEventWrapper {
    cuda_event:         CudaEvent,
    cuda_stream:        CudaStream,
    device_id:          i32,

    status:             Atomic<i32>,
    mutex_recorded:     parking_lot::RawMutex,
    cv_recorded:        std::sync::Condvar,
    err_msg:            String,
}

impl CudaEventWrapper {

    pub fn new(option: &DeviceOption) -> Self {
    
        todo!();
        /*
            : cuda_stream_(nullptr),
            device_id_(option.device_id()),
            status_(EventStatus::EVENT_INITIALIZED) 

        CAFFE_ENFORCE(option.device_type(), PROTO_CUDA);
        CUDAGuard g(device_id_);
        try {
          CUDA_ENFORCE(cudaEventCreateWithFlags(
              &cuda_event_, cudaEventDefault | cudaEventDisableTiming));
        } catch (const Error&) {
          std::cerr << "ERROR: Failed to load CUDA.\n"
                    << "HINT: Check that this binary contains GPU code."
                    << std::endl;
          throw;
        }
        */
    }
}

impl Drop for CudaEventWrapper {
    fn drop(&mut self) {
        todo!();
        /* 
        CUDAGuard g(device_id_);
        CUDA_CHECK(cudaEventDestroy(cuda_event_));
       */
    }
}


pub const kNoError: &'static str = "No error";

#[inline] pub fn event_createCUDA(option: &DeviceOption, event: *mut Event)  {
    
    todo!();
    /*
        event->event_ = std::make_shared<CudaEventWrapper>(option);
    */
}

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


#[inline] pub fn event_finishCUDA(event: *const Event)  {
    
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
        CUDAGuard g(wrapper->device_id_);
        auto cudaResult = cudaEventSynchronize(wrapper->cuda_event_);
        if (cudaResult == cudaSuccess) {
          wrapper->status_ = EventStatus::EVENT_SUCCESS;
        } else {
          const auto& err_msg = cudaGetErrorString(cudaResult);

          std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
    */
}

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


#[inline] pub fn event_queryCUDA(event: *const Event) -> EventStatus {
    
    todo!();
    /*
        auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
      if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
        auto cudaResult = cudaEventQuery(wrapper->cuda_event_);
        if (cudaResult == cudaSuccess) {
          wrapper->status_ = EventStatus::EVENT_SUCCESS;
        } else if (cudaResult != cudaErrorNotReady) {
          const auto& err_msg = cudaGetErrorString(cudaResult);

          std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
        }
      }
      return static_cast<EventStatus>(wrapper->status_.load());
    */
}

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

register_event_create_function![CUDA,          EventCreateCUDA];
register_event_record_function![CUDA,          EventRecordCUDA];
register_event_wait_function![CUDA,            CUDA, EventWaitCUDACUDA];
register_event_wait_function![CPU,             CUDA, EventWaitCPUCUDA];
register_event_wait_function![CUDA,            CPU, EventWaitCUDACPU];
register_event_finish_function![CUDA,          EventFinishCUDA];

register_event_query_function![CUDA,           EventQueryCUDA];
register_event_error_message_function![CUDA,   EventErrorMessageCUDA];
register_event_set_finished_function![CUDA,    EventSetFinishedCUDA];
register_event_reset_function![CUDA,           EventResetCUDA];

register_event_wait_function![MKLDNN,          CUDA, EventWaitCPUCUDA];
register_event_wait_function![CUDA,            MKLDNN, EventWaitCUDACPU];

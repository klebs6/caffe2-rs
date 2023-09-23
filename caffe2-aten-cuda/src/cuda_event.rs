crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CudaEvent.h]

/**
  | CUDAEvents are movable not copyable
  | wrappers around CUDA's events.
  | 
  | CUDAEvents are constructed lazily
  | when first recorded unless it is reconstructed
  | from a cudaIpcEventHandle_t. The event
  | has a device, and this device is acquired
  | from the first recording stream. However,
  | if reconstructed from a handle, the
  | device should be explicitly specified;
  | or if ipc_handle() is called before
  | the event is ever recorded, it will use
  | the current device.
  | 
  | Later streams that record the event
  | must match this device.
  |
  */
#[derive(Default)]
pub struct CudaEvent {
    flags:        u32, // default = cudaEventDisableTiming
    is_created:   bool, // default = false
    was_recorded: bool, // default = false
    device_index: DeviceIndex, // default = -1
    event:        cuda::Event,
}

impl Drop for CudaEvent {

    /// Note: event destruction done on creating
    /// device to avoid creating a CUDA context on
    /// other devices.
    ///
    fn drop(&mut self) {
        todo!();
        /*
            try {
          if (is_created_) {
            CUDAGuard guard(device_index_);
            cudaEventDestroy(event_);
          }
        } catch (...) { /* No throw */ }
        */
    }
}

impl Ord<CudaEvent> for CudaEvent {
    
    // Less than operator (to allow use in sets)
    #[inline] fn cmp(&self, other: &CudaEvent) -> Ordering {
        todo!();
        /*
            return left.event_ < right.event_;
        */
    }
}

impl PartialOrd<CudaEvent> for CudaEvent {
    #[inline] fn partial_cmp(&self, other: &CudaEvent) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl CudaEvent {

    // Constructors
    //
    // Default value for `flags` is specified below - it's cudaEventDisableTiming
    pub fn new(flags: u32) -> Self {
    
        todo!();
        /*


            : flags_{flags}
        */
    }
    
    pub fn new(
        device_index: DeviceIndex,
        handle:       *const CudaIpcEventHandle) -> Self {
    
        todo!();
        /*


            #ifndef __HIP_PLATFORM_HCC__
          device_index_ = device_index;
          CUDAGuard guard(device_index_);

          AT_CUDA_CHECK(cudaIpcOpenEventHandle(&event_, *handle));
          is_created_ = true;
        #else
          AT_ERROR("cuIpcOpenEventHandle with HIP is not supported");
        #endif
        */
    }
    
    pub fn new(other: CudaEvent) -> Self {
    
        todo!();
        /*


            moveHelper(move(other));
        */
    }
    
    pub fn assign_from(&mut self, other: CudaEvent) -> &mut CudaEvent {
        
        todo!();
        /*
            moveHelper(move(other));
        return *this;
        */
    }
    
    pub fn operator_cuda_event_t(&self) -> cuda::Event {
        
        todo!();
        /*
            return event();
        */
    }
    
    pub fn device(&self) -> Option<Device> {
        
        todo!();
        /*
            if (is_created_) {
          return Device(kCUDA, device_index_);
        } else {
          return {};
        }
        */
    }
    
    pub fn is_created(&self) -> bool {
        
        todo!();
        /*
            return is_created_;
        */
    }
    
    pub fn device_index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return device_index_;
        */
    }
    
    pub fn event(&self) -> cuda::Event {
        
        todo!();
        /*
            return event_;
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaEventQuery can be safely called
      | from any device
      |
      */
    pub fn query(&self) -> bool {
        
        todo!();
        /*
            if (!is_created_) {
          return true;
        }

        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) {
          return true;
        } else if (err != cudaErrorNotReady) {
          C10_CUDA_CHECK(err);
        }

        return false;
        */
    }
    
    pub fn record(&mut self)  {
        
        todo!();
        /*
            record(getCurrentCUDAStream());
        */
    }
    
    pub fn record_once(&mut self, stream: &CUDAStream)  {
        
        todo!();
        /*
            if (!was_recorded_) record(stream);
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaEventRecord must be called on the
      | same device as the event.
      |
      */
    pub fn record(&mut self, stream: &CUDAStream)  {
        
        todo!();
        /*
            if (!is_created_) {
          createEvent(stream.device_index());
        }

        TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
          " does not match recording stream's device ", stream.device_index(), ".");
        CUDAGuard guard(device_index_);
        AT_CUDA_CHECK(cudaEventRecord(event_, stream));
        was_recorded_ = true;
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaStreamWaitEvent must be called
      | on the same device as the stream.
      | 
      | The event has no actual GPU resources
      | associated with it.
      |
      */
    pub fn block(&mut self, stream: &CUDAStream)  {
        
        todo!();
        /*
            if (is_created_) {
          CUDAGuard guard(stream.device_index());
          AT_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
        }
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaEventElapsedTime can be safely
      | called from any device
      |
      */
    pub fn elapsed_time(&self, other: &CudaEvent) -> f32 {
        
        todo!();
        /*
            TORCH_CHECK(is_created_ && other.isCreated(),
          "Both events must be recorded before calculating elapsed time.");
        float time_ms = 0;
        // raise cudaErrorNotReady if either event is recorded but not yet completed
        AT_CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_, other.event_));
        return time_ms;
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaEventSynchronize can be safely
      | called from any device
      |
      */
    pub fn synchronize(&self)  {
        
        todo!();
        /*
            if (is_created_) {
          AT_CUDA_CHECK(cudaEventSynchronize(event_));
        }
        */
    }
    
    /**
      | -----------
      | @note
      | 
      | cudaIpcGetEventHandle must be called
      | on the same device as the event
      |
      */
    pub fn ipc_handle(&mut self, handle: *mut CudaIpcEventHandle)  {
        
        todo!();
        /*
            #ifndef __HIP_PLATFORM_HCC__
          if (!is_created_) {
            // this CudaEvent object was initially constructed from flags but event_
            // is not created yet.
            createEvent(getCurrentCUDAStream().device_index());
          }
          CUDAGuard guard(device_index_);
          AT_CUDA_CHECK(cudaIpcGetEventHandle(handle, event_));
        #else
          AT_ERROR("cuIpcGetEventHandle with HIP is not supported");
        #endif
        */
    }
    
    pub fn create_event(&mut self, device_index: DeviceIndex)  {
        
        todo!();
        /*
            device_index_ = device_index;
        CUDAGuard guard(device_index_);
        AT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags_));
        is_created_ = true;
        */
    }
    
    pub fn move_helper(&mut self, other: CudaEvent)  {
        
        todo!();
        /*
            swap(flags_, other.flags_);
        swap(is_created_, other.is_created_);
        swap(was_recorded_, other.was_recorded_);
        swap(device_index_, other.device_index_);
        swap(event_, other.event_);
        */
    }
}

crate::ix!();

/**
  | MIOpenState is the owner of the MIOpenWorkspace,
  | and serializes all executions of operations that
  | use the state onto it's own stream (so multiple
  | Net workers can reuse the same workspace from
  | different threads and HIP streams).
  */
pub struct MIOpenState {
    miopen_handle:  miopenHandle_t, // default = nullptr
    before:         hipEvent_t,     // default = nullptr
    after:          hipEvent_t,     // default = nullptr
    stream:         hipStream_t,    // default = nullptr
    workspace:      MIOpenWorkspace,
    gpu_id:         usize, // default = 0
}

impl Drop for MIOpenState {

    fn drop(&mut self) {
        todo!();
        /* 
            HIPGuard g(gpu_id_);
            MIOPEN_CHECK(miopenDestroy(miopen_handle_));
            HIP_CHECK(hipStreamDestroy(stream_));
            HIP_CHECK(hipEventDestroy(after_));
            HIP_CHECK(hipEventDestroy(before_));
         */
    }
}

impl MIOpenState {

    pub fn new(gpu_id: usize) -> Self {
    
        todo!();
        /*
            : gpu_id_(gpu_id)

            HIPGuard g(gpu_id_);
            MIOPEN_ENFORCE(miopenCreate(&miopen_handle_));
            HIP_ENFORCE(hipEventCreate(&before_));
            HIP_ENFORCE(hipEventCreate(&after_));
            HIP_ENFORCE(hipStreamCreate(&stream_));
            MIOPEN_ENFORCE(miopenSetStream(miopen_handle_, stream_));
        */
    }
    
    #[inline] pub fn miopen_handle(&mut self) -> &mut miopenHandle_t {
        
        todo!();
        /*
            return miopen_handle_;
        */
    }
    
    #[inline] pub fn workspace(&mut self) -> &mut MIOpenWorkspace {
        
        todo!();
        /*
            return workspace_;
        */
    }
    
    #[inline] pub fn execute<F>(&mut self, stream: hipStream_t, f: F)  {
    
        todo!();
        /*
            HIP_ENFORCE(hipEventRecord(before_, stream));
            HIP_ENFORCE(hipStreamWaitEvent(stream_, before_, 0));
            f(this);
            HIP_ENFORCE(hipEventRecord(after_, stream_));
            HIP_ENFORCE(hipStreamWaitEvent(stream, after_, 0));
        */
    }
}

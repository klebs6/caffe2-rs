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

impl Drop for CudaEventWrapper {

    fn drop(&mut self) {
        todo!();
        /* 
        CUDAGuard g(device_id_);
        CUDA_CHECK(cudaEventDestroy(cuda_event_));
       */
    }
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

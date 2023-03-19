crate::ix!();

pub struct SyncedCudnnState {
    mutex: parking_lot::RawMutex,
    state: Box<CudnnState>,
}

/**
  | CudnnState is the owner of the CudnnWorkspace,
  | and serializes all executions of operations
  | that use the state onto it's own stream
  | (so multiple Net workers can reuse the same workspace
  | from different threads and CUDA streams).
  |
  */
pub struct CudnnState {
    cudnn_handle:   CudnnHandle,  // default = nullptr
    before:         CudaEvent,    // default = nullptr
    after:          CudaEvent,    // default = nullptr
    stream:         CudaStream,   // default = nullptr
    workspace:      CudnnWorkspace,
    gpu_id:         usize,          // default = 0
}

impl CudnnState {
    
    pub fn new(gpu_id: usize) -> Self {
    
        todo!();
        /*
            : gpu_id_(gpu_id) 

                CUDAGuard g(gpu_id_);
                CUDNN_ENFORCE(cudnnCreate(&cudnn_handle_));
                CUDA_ENFORCE(cudaEventCreate(&before_));
                CUDA_ENFORCE(cudaEventCreate(&after_));
                CUDA_ENFORCE(cudaStreamCreate(&stream_));
                CUDNN_ENFORCE(cudnnSetStream(cudnn_handle_, stream_));
        */
    }
}

impl Drop for CudnnState {
    fn drop(&mut self) {
        todo!();
        /* 
                CUDAGuard g(gpu_id_);
                CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
                CUDA_CHECK(cudaStreamDestroy(stream_));
                CUDA_CHECK(cudaEventDestroy(after_));
                CUDA_CHECK(cudaEventDestroy(before_));
             */
    }
}

impl CudnnState {
    
    #[inline] pub fn cudnn_handle(&mut self) -> &mut CudnnHandle {
        
        todo!();
        /*
            return cudnn_handle_;
        */
    }
    
    #[inline] pub fn workspace(&mut self) -> &mut CudnnWorkspace {
        
        todo!();
        /*
            return workspace_;
        */
    }
    
    #[inline] pub fn execute<F>(&mut self, stream: CudaStream, f: F)  {
    
        todo!();
        /*
            CUDA_ENFORCE(cudaEventRecord(before_, stream));
                    CUDA_ENFORCE(cudaStreamWaitEvent(stream_, before_, 0));
                    f(this);
                    CUDA_ENFORCE(cudaEventRecord(after_, stream_));
                    CUDA_ENFORCE(cudaStreamWaitEvent(stream, after_, 0));
        */
    }
}

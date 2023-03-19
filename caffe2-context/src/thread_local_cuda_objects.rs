crate::ix!();

/**
  | A struct to host thread-local cuda objects.
  | 
  | In Caffe2, each thread has its own non-default
  | cuda stream as well as related objects
  | such as cublas and curand handles.
  | 
  | This is achieved by having the ThreadLocalCUDAObjects
  | wrapper that takes care of allocating
  | and deallocating these objects at the
  | thread scope.
  | 
  | This class is solely used inside CUDAContext
  | and should not be used externally.
  | 
  | This class manages the mapping from
  | logical stream ID (int stream_id passed
  | around in Caffe2) and CudaStream objects.
  | 
  | We intend to eventually deprecate the
  | logical stream ID interface, but not
  | for now.
  |
  */
pub struct ThreadLocalCUDAObjects {

    /**
      | WARNING: mapping from logical stream ID to
      | CudaStream is NOT bijective; multiple
      | logical stream IDs may map to the same
      | underlying stream ID.
      */
    cuda_streams:   [Vec<CudaStream>; COMPILE_TIME_MAX_GPUS],

    cublas_handles: HashMap<CudaStream, CublasHandle>,

    #[cfg(caffe2_use_cudnn)]
    cudnn_handles_: HashMap<CudaStream, CudnnHandle>,
}

impl Drop for ThreadLocalCUDAObjects {

    fn drop(&mut self) {
        /*
        for (auto element : cublas_handles_) {
          if (element.second) {
            CUBLAS_CHECK(cublasDestroy(element.second));
          }
        }
    #ifdef CAFFE2_USE_CUDNN
        for (auto element : cudnn_handles_) {
          if (element.second) {
            CUDNN_CHECK(cudnnDestroy(element.second));
          }
        }
    #endif // CAFFE2_USE_CUDNN

        */
    }
}

impl Default for ThreadLocalCUDAObjects {
    
    fn default() -> Self {
        todo!();
        /*
            for (DeviceIndex i = 0; i < C10_COMPILE_TIME_MAX_GPUS; ++i) {
          cuda_streams_[i] = vector<CudaStream>();
        
        */
    }
}

impl ThreadLocalCUDAObjects {
    
    /**
      | Record current stream id for the current
      | thread.
      |
      | This is the new API we're trying to
      | migrate use cases to and get rid of
      | explicit stream id passing.
      |
      | For now it's invoked in
      | CUDAContext::SwitchToDevice
      */
    #[inline] pub fn set_current_stream_id(
        &mut self, 
        gpu: DeviceIndex,
        stream_id: StreamId)  
    {
        todo!();
        /*
            // TODO: use current device id from thread local instead of passing gpu in
        setCurrentCudaStream(GetCudaStream(gpu, stream_id));
        */
    }
    
    /**
      | Uses the logical stream id from the thread
      | local to pick the stream
      |
      | We're going to migrate all usages to this
      | case API instead of passing the stream id
      | directly
      */
    #[inline] pub fn get_stream(&mut self, gpu: DeviceIndex) -> CudaStream {
        
        todo!();
        /*
            return getCurrentCudaStream(gpu).stream();
        */
    }
    
    #[inline] pub fn get_stream_with_stream_id(
        &mut self, 
        gpu:       DeviceIndex,
        stream_id: StreamId) -> CudaStream 
    {
        todo!();
        /*
            return GetCudaStream(gpu, stream_id).stream();
        */
    }
    
    /**
      | Uses the logical stream id from the thread
      | local to pick the stream
      |
      | We're going to migrate all usages to this
      | case API instead of passing the stream id
      | directly
      */
    #[inline] pub fn get_handle_with_device_index(&mut self, gpu: DeviceIndex) -> CublasHandle {
        
        todo!();
        /*
            return GetHandle(getCurrentCudaStream(gpu));
        */
    }
    
    /**
      | Retrieves the CudaStream corresponding to
      | a logical stream ID, ensuring that it
      | exists in cuda_streams_ if it has not been
      | allocated yet.
      */
    #[inline] pub fn get_cudastream(
        &mut self, 
        gpu:       DeviceIndex,
        stream_id: StreamId) -> CudaStream 
    {
        todo!();
        /*
            vector<CudaStream>& gpu_streams = cuda_streams_[gpu];
        while (gpu_streams.size() <= static_cast<size_t>(stream_id)) {
          // NB: This streams are not guaranteed to be unique; we'll
          // wrap around once we run out of streams in the pool.
          gpu_streams.emplace_back(getStreamFromPool(/* high priority */ false, gpu));
        }
        return gpu_streams[stream_id];
        */
    }
    
    #[inline] pub fn get_handle_with_cuda_stream(&mut self, 
        cuda_stream: CudaStream) -> CublasHandle {
        
        todo!();
        /*
            CUDAGuard guard(cuda_stream.device_index());
        // Default construct in the map if it doesn't exist, and return a mutable
        // reference to it.
        auto& r = cublas_handles_[cuda_stream];
        if (r == nullptr) {
          CUBLAS_ENFORCE(cublasCreate(&r));
          // The default is CUBLAS_POINTER_MODE_HOST. You can override
          // it after obtaining the cublas handle, but do that with
          // caution.
          CUBLAS_ENFORCE(cublasSetPointerMode(r, CUBLAS_POINTER_MODE_HOST));
          CUBLAS_ENFORCE(cublasSetStream(r, cuda_stream));
        }
        return r;
        */
    }

    /**
      | Uses the logical stream id from the thread
      | local to pick the stream
      |
      | We're going to migrate all usages to this
      | case API instead of passing the stream id
      | directly
      */
    #[inline] pub fn get_cudnn_handle(&mut self, gpu: DeviceIndex)  {
        
        todo!();
        /*
            return GetCudnnHandle(getCurrentCudaStream(gpu));
        */
    }
    
    #[cfg(caffe2_use_cudnn)]
    #[inline] pub fn get_cudnn_handle(&mut self, cuda_stream: CudaStream)  {
        
        todo!();
        /*
            CUDAGuard guard(cuda_stream.device_index());
        auto& r = cudnn_handles_[cuda_stream];
        if (r == nullptr) {
          CUDNN_ENFORCE(cudnnCreate(&r));
          CUDNN_ENFORCE(cudnnSetStream(r, cuda_stream));
        }
        return r;
        */
    }
}

/**
  | Gets the current memory pool type used
  | by Caffe2.
  | 
  | The memory pool is set up during caffe2's
  | global initialization time.
  |
  */
#[inline] pub fn get_cuda_memory_pool_type() -> CudaMemoryPoolType {
    
    todo!();
    /*
    
    */
}

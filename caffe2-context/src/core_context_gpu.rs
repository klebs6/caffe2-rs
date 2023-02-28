crate::ix!();

pub struct CUDAContext {
    gpu_id:           i32,
    random_seed:      i32,
    curand_generator: CuRandGenerator,
}

impl From<DeviceIndex> for CUDAContext {
    /// The default cuda context constructor.
    fn from(gpu_id: DeviceIndex) -> CUDAContext {

        todo!();
        /*

        */
    }
}

impl From<&DeviceOption> for CUDAContext {

    fn from(option: &DeviceOption) -> CUDAContext {
    
        todo!();
        /*
        
        */
    }
}

impl From<Device> for CUDAContext {

    fn from(device: Device) -> CUDAContext {
        todo!();
        /*
            : CUDAContext(DeviceToOption(device))
        */
    }
}

impl Default for CUDAContext {
    fn default() -> Self {
        let gpu_id: DeviceIndex = -1;
        Self::from(gpu_id)
    }
}
    
impl BaseContext for CUDAContext {

    #[inline] fn switch_to_device(&mut self, stream_id: StreamId)  {
        
        todo!();
        /*
            getCudaObjects().SetCurrentStreamId(gpu_id_, stream_id);
        CaffeCudaSetDevice(gpu_id_);
        */
    }
    
    #[inline] fn wait_event(&mut self, ev: &Event)  {
        
        todo!();
        /*
            ev.Wait(CUDA, this);
        */
    }
    
    #[inline] fn record(
        &self, 
        ev: *mut Event,
        err_msg: *const u8)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(ev, "Event must not be null.");
        ev->Record(CUDA, this, err_msg);
        */
    }

    /**
      | Note on current use cases:
      |
      | FinishDeviceComputation must be called on
      | the same cpu thread as SwitchToDevice()
      */
    #[inline] fn finish_device_computation(&mut self)  {
        
        todo!();
        /*
            CUDA_ENFORCE(cudaStreamSynchronize(getCudaObjects().GetStream(gpu_id_)));
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
          CAFFE_THROW("Encountered CUDA error: ", cudaGetErrorString(error));
        }
        */
    }
    
    #[inline] fn copy_bytes_same_device(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            CopyBytes<CUDAContext, CUDAContext>(nbytes, src, dst);
        */
    }
    
    #[inline] fn copy_bytes_toCPU(
        &mut self, 
        nbytes: usize,
        src: *const c_void,
        dst: *mut c_void)  
    {
        todo!();
        /*
            CopyBytes<CUDAContext, CPUContext>(nbytes, src, dst);
        */
    }
    
    #[inline] fn copy_bytes_fromCPU(
        &mut self, 
        nbytes: usize,
        src:    *const c_void,
        dst:    *mut c_void)  
    {
        todo!();
        /*
            CopyBytes<CPUContext, CUDAContext>(nbytes, src, dst);
        */
    }

    #[inline] fn device(&self) -> Device {
        
        todo!();
        /*
            return at::Device(CUDA, gpu_id_);
        */
    }
    
    #[inline] fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return CUDA;
        */
    }
    
    #[cfg(caffe2_use_cudnn)]
    #[inline] pub fn cudnn_handle(&mut self)  {


        
        todo!();
        /*
            return getCudaObjects().GetCudnnHandle(gpu_id_);
        */
    }
}

impl CUDAContext {

    #[inline] fn new(nbytes: usize) -> DataPtr {
        
        todo!();
        /*
            return GetAllocator(CUDA)->allocate(nbytes);
        */
    }

    #[inline] fn get_cuda_objects<'a>() -> &'a mut ThreadLocalCUDAObjects {
        
        todo!();
        /*
        
        */
    }

    /**
      | Get a mutex to lock out
      |
      | cudaMalloc / cudaFree
      |
      | calls when NCCL kernels are being
      | launched. Should remove threat of
      | deadlocks
      */
    #[inline] fn mutex<'a>() -> &'a mut parking_lot::RawMutex {
        
        todo!();
        /*
        
        */
    }

    #[inline] fn copy_bytes_sync(
        nbytes:     usize,
        src:        *const c_void,
        src_device: Device,
        dst:        *mut c_void,
        dst_device: Device)  {
        
        todo!();
        /*
        
        */
    }

    #[inline] fn copy_bytes_async(
        nbytes:     usize,
        src:        *const c_void,
        src_device: Device,
        dst:        *mut c_void,
        dst_device: Device)  {

        todo!();
        /*
        
        */
    }

    #[inline] fn max_memory_by_gpu() -> Vec<i64> {
        
        todo!();
        /*
        
        */
    }

    /**
      | Functions to query memory stats. Only
      | available if flag
      | 
      | --caffe2_gpu_memory_tracking is
      | enabled.
      |
      */
    #[inline] fn total_memory_by_gpu() -> Vec<i64> {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] fn get_device_type() -> DeviceType {
        
        todo!();
        /*
            return CUDA;
        */
    }

    #[inline] fn is_stream_free(
        option: &DeviceOption,
        stream_id: StreamId) -> bool 
    {
        todo!();
        /*
            auto stream = CUDAContext::cuda_stream(option.device_id(), stream_id);
        return cudaStreamQuery(stream) == cudaSuccess;
        */
    }

    #[inline] fn supports_async_scheduling() -> bool {
        
        todo!();
        /*
            return true;
        */
    }

    /// By default CUDA operators have async device parts
    #[inline] fn has_async_part_default() -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    #[inline] fn copy_items<SrcContext, DstContext>(
        &mut self,
        meta: TypeMeta,
        n:    usize,
        src:  *const c_void,
        dst:  *mut c_void) 
    {
        todo!();
        /*
            CAFFE_ENFORCE(!meta.copy(), "CUDAContext requires fundamental types.");
            CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
        */
    }
    

    #[inline] fn copy<T, SrcContext, DstContext>(
        &mut self,
        n:   i32,
        src: *const T,
        dst: *mut T) {
        todo!();
        /*
            CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                         static_cast<const void*>(src),
                                         static_cast<void*>(dst));
        */
    }

    #[inline] fn copy_bytes<SrcContext, DstContext>(
        &mut self,
        nbytes: usize,
        src:    *const c_void,
        dst:    *mut c_void) 
    {
        todo!();
        /*
            CUDA_ENFORCE(cudaMemcpyAsync(
                dst,
                src,
                nbytes,
                cudaMemcpyDefault,
                getCudaObjects().GetStream(gpu_id_)));
        */
    }
    
    #[inline] fn device_id(&self) -> i32 {
        
        todo!();
        /*
            return gpu_id_;
        */
    }
    
    #[inline] fn cuda_stream(&self) -> CudaStream {
        
        todo!();
        /*
            return getCudaObjects().GetStream(gpu_id_);
        */
    }

    #[inline] fn cuda_stream_from_ids(
        &mut self,
        gpu_id: DeviceIndex,
        stream_id: StreamId) -> CudaStream 
    {
        
        todo!();
        /*
            return getCudaObjects().GetStream(gpu_id, stream_id);
        */
    }

    #[inline] fn cublas_handle(&mut self) -> CublasHandle {
        
        todo!();
        /*
            return getCudaObjects().GetHandle(gpu_id_);
        */
    }
    
    #[inline] fn curand_generator(&mut self) -> &mut CuRandGenerator {
        
        todo!();
        /*
            if (!curand_generator_) {
          CUDAGuard guard(gpu_id_);
          CURAND_ENFORCE(
              curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
          CURAND_ENFORCE(
              curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed_));
          CHECK_NOTNULL(curand_generator_);
        }
        CURAND_ENFORCE(curandSetStream(curand_generator_, cuda_stream()));
        return curand_generator_;
        */
    }
}

pub enum CudaMemoryPoolType {
    NoMemoryPool,
    CUB,
    THC,
}

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



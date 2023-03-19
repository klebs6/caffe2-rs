crate::ix!();

pub enum CudaMemoryPoolType {
    NoMemoryPool,
    CUB,
    THC,
}

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


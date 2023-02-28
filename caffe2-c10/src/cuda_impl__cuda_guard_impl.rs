crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/impl/CUDAGuardImpl.h]

#[cfg(feature = "cuda")]
pub struct CUDAGuardImpl {
    base: dyn DeviceGuardImplInterface,
}

#[cfg(feature = "cuda")]
pub mod cuda_guard_impl {

    use super::*;

    pub const STATIC_TYPE: DeviceType = DeviceType::CUDA;
}

#[cfg(feature = "cuda")]
impl CUDAGuardImpl {
    
    pub fn new(t: DeviceType) -> Self {
    
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(t == DeviceType::CUDA);
        */
    }
    
    
    pub fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return DeviceType::CUDA;
        */
    }
    
    
    pub fn exchange_device(&self, d: Device) -> Device {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(d.is_cuda());
            Device old_device = getDevice();
            if (old_device.index() != d.index()) {
                C10_CUDA_CHECK(cudaSetDevice(d.index()));
            }
            return old_device;
        */
    }
    
    
    pub fn get_device(&self) -> Device {
        
        todo!();
        /*
            int device;
            C10_CUDA_CHECK(cudaGetDevice(&device));
            return Device(DeviceType::CUDA, device);
        */
    }
    
    
    pub fn unchecked_get_device(&self) -> Option<Device> {
        
        todo!();
        /*
            int device;
            auto err = cudaGetDevice(&device);
            C10_CUDA_CHECK_WARN(err);
            if (err != cudaSuccess) {
                return nullopt;
            }
            return Device(DeviceType::CUDA, device);
        */
    }
    
    
    pub fn set_device(&self, d: Device)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(d.is_cuda());
            Device current_device = getDevice();
            if (current_device != d) {
                C10_CUDA_CHECK(cudaSetDevice(d.index()));
            }
        */
    }
    
    
    pub fn unchecked_set_device(&self, d: Device)  {
        
        todo!();
        /*
            auto current_device = uncheckedGetDevice();
            if (!current_device.has_value() || current_device.value() != d) {
                C10_CUDA_CHECK_WARN(cudaSetDevice(d.index()));
            }
        */
    }
    
    
    pub fn get_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return getCurrentCudaStream(d.index()).unwrap();
        */
    }
    
    
    pub fn get_default_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return getDefaultCudaStream(d.index());
        */
    }
    
    
    pub fn get_stream_from_global_pool(&self, 
        d:                Device,
        is_high_priority: bool) -> Stream {
        let is_high_priority: bool = is_high_priority.unwrap_or(false);

        todo!();
        /*
            return getStreamFromPool(isHighPriority, d.index());
        */
    }

    /// NB: These do NOT set the current device
    pub fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            CudaStream cs(s);
            auto old_stream = getCurrentCudaStream(s.device().index());
            setCurrentCudaStream(cs);
            return old_stream.unwrap();
        */
    }
    
    
    pub fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            return device_count();
        */
    }

    /// Event-related functions
    pub fn create_event(&self, 
        cuda_event: *mut CudaEvent,
        flag:       EventFlag)  {
        
        todo!();
        /*
            // Maps PyTorch's Event::Flag to Cuda flag
            auto cuda_flag = cudaEventDefault;
            switch (flag) {
                case EventFlag::PYTORCH_DEFAULT:
                case EventFlag::CUDA_EVENT_DISABLE_TIMING:
                    cuda_flag = cudaEventDisableTiming;
                    break;
                case EventFlag::BACKEND_DEFAULT:
                case EventFlag::CUDA_EVENT_DEFAULT:
                    cuda_flag = cudaEventDefault;
                    break;
                default:
                    TORCH_CHECK(false, "Cuda event received unknown flag");
            }

            C10_CUDA_CHECK(cudaEventCreateWithFlags(cuda_event, cuda_flag));
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut c_void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
            if (!event)
                    return;
                auto cuda_event = static_cast<cudaEvent_t>(event);
                int orig_device;
                C10_CUDA_CHECK_WARN(cudaGetDevice(&orig_device));
                C10_CUDA_CHECK_WARN(cudaSetDevice(device_index));
                C10_CUDA_CHECK_WARN(cudaEventDestroy(cuda_event));
                C10_CUDA_CHECK_WARN(cudaSetDevice(orig_device));
        */
    }
    
    pub fn record(&self, 
        event:        *mut *mut c_void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            TORCH_CHECK(
                device_index == -1 || device_index == stream.device_index(),
                "Event device index ",
                device_index,
                " does not match recording stream's device index ",
                stream.device_index(),
                ".");

            cudaEvent_t cuda_event = static_cast<cudaEvent_t>(*event);
            CudaStream cuda_stream{stream};

            // Moves to stream's device to record
            const auto orig_device = getDevice();
            setDevice(stream.device());

            // Creates the event (lazily)
            if (!cuda_event)
                createEvent(&cuda_event, flag);
            C10_CUDA_CHECK(cudaEventRecord(cuda_event, cuda_stream));
            // Makes the void* point to the (possibly just allocated) Cuda event
            *event = cuda_event;

            // Resets device
            setDevice(orig_device);
        */
    }
    
    pub fn block(&self, 
        event:  *mut c_void,
        stream: &Stream)  {
        
        todo!();
        /*
            if (!event)
                return;
            cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
            CudaStream cuda_stream{stream};
            const auto orig_device = getDevice();
            setDevice(stream.device());
            C10_CUDA_CHECK(cudaStreamWaitEvent(
                    cuda_stream,
                    cuda_event,
                    /*flags (must be zero)=*/0));
            setDevice(orig_device);
        */
    }

    /// May be called from any device
    pub fn query_event(&self, event: *mut c_void) -> bool {
        
        todo!();
        /*
            if (!event)
                return true;
            cudaEvent_t cuda_event = static_cast<cudaEvent_t>(event);
            const cudaError_t err = cudaEventQuery(cuda_event);
            if (err != cudaErrorNotReady) {
                C10_CUDA_CHECK(err);
            }
            return (err == cudaSuccess);
        */
    }

    /// Stream-related functions
    pub fn query_stream(&self, stream: &Stream) -> bool {
        
        todo!();
        /*
            CudaStream cuda_stream{stream};
            return cuda_stream.query();
        */
    }
    
    pub fn synchronize_stream(&self, stream: &Stream)  {
        
        todo!();
        /*
            CudaStream cuda_stream{stream};
            cuda_stream.synchronize();
        */
    }
    
    pub fn record_data_ptr_on_stream(&self, 
        data_ptr: &DataPtr,
        stream:   &Stream)  {
        
        todo!();
        /*
            CudaStream cuda_stream{stream};
                CUDACachingAllocator::recordStream(data_ptr, cuda_stream);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/cuda/impl/CUDAGuardImpl.cpp]

#[cfg(feature = "cuda")]
lazy_static!{
    /*
    constexpr DeviceType CUDAGuardImpl::static_type;
    */
}

#[cfg(feature = "cuda")]
c10_register_guard_impl!{Cuda, CUDAGuardImpl}

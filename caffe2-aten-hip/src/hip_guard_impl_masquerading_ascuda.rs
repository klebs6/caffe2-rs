/**
  | Use of hip namespace here makes hipification
  | easier, because I don't have to also
  | fix namespaces. Sorry!
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h]

/**
  | Note [Masquerading as CUDA]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | c10_hip is very easy to understand: it is
  | HIPified from c10_cuda, and anywhere you said
  | CUDA, the source code now says HIP.
  |
  | HIPified PyTorch is much harder to understand:
  | it is HIPified from regular PyTorch, yes, but
  | NO source-to-source translation from CUDA to
  | HIP occurs; instead, anywhere we see "CUDA", it
  | actually means "HIP". For example, when you use
  | HIPified PyTorch, you say x.cuda() to move
  | a tensor onto ROCm device.
  |
  | We call this situation "HIP masquerading as
  | CUDA".
  |
  | This leads to a very awkward situation when we
  | want to call c10_hip code from PyTorch, since
  | c10_hip is expecting things to be called HIP,
  | but PyTorch is calling them CUDA (masquerading
  | as HIP).  To fix this impedance mismatch, we
  | have MasqueradingAsCUDA variants for all
  | c10_hip classes.
  |
  | These translate between the "HIP" and "CUDA
  | masquerading as HIP" worlds.
  |
  | For example, HIPGuardImplMasqueradingAsCUDA
  | (this file) provides something like
  | a HIPGuardImpl, but it reports its DeviceType
  | as CUDA (e.g., type() returns CUDA, getDevice()
  | reports the current HIP device as a CUDA
  | device.)
  |
  | We should be able to delete all of these
  | classes entirely once we switch PyTorch to
  | calling a HIP a HIP.
  |
  | When you add a new MasqueradingAsCUDA
  | class/function, you need to also update the
  | rewrite rules in
  | torch/utils/hipify/cuda_to_hip_mappings.py
  |
  | By the way, note that the cpp file associated
  | with this also *overwrites* the entry in the
  | DeviceGuardImpl registry for CUDA with this HIP
  | implementation.
  */
pub struct HIPGuardImplMasqueradingAsCUDA {
    base: DeviceGuardImplInterface,
}

pub mod hip_guarding_impl_masquerading_as_cuda {

    use super::*;

    pub const STATIC_TYPE: DeviceType = DeviceType::Cuda;
}

impl HIPGuardImplMasqueradingAsCUDA {
    
    pub fn new(t: DeviceType) -> Self {
    
        todo!();
        /*


            TORCH_INTERNAL_ASSERT(t == DeviceType::Cuda);
        */
    }
    
    pub fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return DeviceType::Cuda;
        */
    }
    
    pub fn exchange_device(&self, d: Device) -> Device {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(d.is_cuda());
        Device old_device = getDevice();
        if (old_device.index() != d.index()) {
          C10_HIP_CHECK(hipSetDevice(d.index()));
        }
        return old_device;
        */
    }
    
    pub fn get_device(&self) -> Device {
        
        todo!();
        /*
            int device;
        C10_HIP_CHECK(hipGetDevice(&device));
        return Device(DeviceType::Cuda, device);
        */
    }
    
    pub fn set_device(&self, d: Device)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(d.is_cuda());
        C10_HIP_CHECK(hipSetDevice(d.index()));
        */
    }
    
    pub fn unchecked_set_device(&self, d: Device)  {
        
        todo!();
        /*
            C10_HIP_CHECK_WARN(hipSetDevice(d.index()));
        */
    }
    
    pub fn get_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return getCurrentHIPStreamMasqueradingAsCUDA(d.index()).unwrap();
        */
    }
    
    pub fn get_default_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return getDefaultHIPStreamMasqueradingAsCUDA(d.index());
        */
    }
    
    pub fn get_stream_from_global_pool(&self, 
        d:                Device,
        is_high_priority: bool) -> Stream {
        let is_high_priority: bool = is_high_priority.unwrap_or(false);

        todo!();
        /*
            return getStreamFromPoolMasqueradingAsCUDA(isHighPriority, d.index());
        */
    }
    
    pub fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            HIPStreamMasqueradingAsCUDA cs(s);
        auto old_stream = getCurrentHIPStreamMasqueradingAsCUDA(s.device().index());
        setCurrentHIPStreamMasqueradingAsCUDA(cs);
        return old_stream.unwrap();
        */
    }
    
    pub fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            int deviceCnt;
        C10_HIP_CHECK(hipGetDeviceCount(&deviceCnt));
        return deviceCnt;
        */
    }

    /**
      | Event-related functions
      |
      | Note: hipEventCreateWithFlags should be
      |  called on the same device as the recording
      |  stream's device.
      */
    pub fn create_event(&self, 
        hip_event: *mut hip::Event,
        flag:      EventFlag)  {
        
        todo!();
        /*
            // Maps PyTorch's Event::Flag to HIP flag
        auto hip_flag = hipEventDefault;
        switch (flag) {
          case EventFlag::PYTORCH_DEFAULT:
          case EventFlag::HIP_EVENT_DISABLE_TIMING:
            hip_flag = hipEventDisableTiming;
            break;
          case EventFlag::BACKEND_DEFAULT:
          case EventFlag::HIP_EVENT_DEFAULT:
            hip_flag = hipEventDefault;
            break;
          default:
            TORCH_CHECK(false, "HIP event received unknown flag");
        }

        C10_HIP_CHECK(hipEventCreateWithFlags(hip_event, hip_flag));
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
            if (!event) return;
        auto hip_event = static_cast<hipEvent_t>(event);
        int orig_device;
        C10_HIP_CHECK_WARN(hipGetDevice(&orig_device));
        C10_HIP_CHECK_WARN(hipSetDevice(device_index));
        C10_HIP_CHECK_WARN(hipEventDestroy(hip_event));
        C10_HIP_CHECK_WARN(hipSetDevice(orig_device));
        */
    }
    
    pub fn record(&self, 
        event:        *mut *mut void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
          "Event device index ",
          device_index,
          " does not match recording stream's device index ",
          stream.device_index(),
          ".");

        hipEvent_t hip_event = static_cast<hipEvent_t>(*event);
        HIPStreamMasqueradingAsCUDA hip_stream{stream};

        // Moves to stream's device to record
        const auto orig_device = getDevice();
        setDevice(stream.device());

        // Creates the event (lazily)
        if (!hip_event) createEvent(&hip_event, flag);
        C10_HIP_CHECK(hipEventRecord(hip_event, hip_stream));
        // Makes the void* point to the (possibly just allocated) HIP event
        *event = hip_event;

        // Resets device
        setDevice(orig_device);
        */
    }
    
    pub fn block(&self, 
        event:  *mut void,
        stream: &Stream)  {
        
        todo!();
        /*
            if (!event) return;
        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        HIPStreamMasqueradingAsCUDA hip_stream{stream};
        const auto orig_device = getDevice();
        setDevice(stream.device());
        C10_HIP_CHECK(hipStreamWaitEvent(
          hip_stream,
          hip_event,
          /*flags (must be zero)=*/ 0));
        setDevice(orig_device);
        */
    }
    
    pub fn query_event(&self, event: *mut void) -> bool {
        
        todo!();
        /*
            if (!event) return true;
        hipEvent_t hip_event = static_cast<hipEvent_t>(event);
        const hipError_t err = hipEventQuery(hip_event);
        if (err != hipErrorNotReady) C10_HIP_CHECK(err);
        return (err == hipSuccess);
        */
    }
    
    /**
      | Stream-related functions
      |
      */
    pub fn query_stream(&self, stream: &Stream) -> bool {
        
        todo!();
        /*
            HIPStreamMasqueradingAsCUDA hip_stream{stream};
        return hip_stream.query();
        */
    }
    
    pub fn synchronize_stream(&self, stream: &Stream)  {
        
        todo!();
        /*
            HIPStreamMasqueradingAsCUDA hip_stream{stream};
        hip_stream.synchronize();
        */
    }
    
    pub fn record_data_ptr_on_stream(&self, 
        data_ptr: &DataPtr,
        stream:   &Stream)  {
        
        todo!();
        /*
            HIPStreamMasqueradingAsCUDA hip_stream{stream};
        HIPCachingAllocatorMasqueradingAsCUDA::recordStreamMasqueradingAsCUDA(data_ptr, hip_stream);
        */
    }
}

/**
  | All of the guards which have HIPGuardImpl
  | burned in need to also have variants using
  | HIPGuardImplMasqueradingAsCUDA.
  |
  | This code is all a direct copy from
  | c10/cuda/HIPGuardMasqueradingAsCUDA.h, but
  | with the correct InlineDeviceGuard burned in.
  | Sorry about the copy-pasting.
  */
pub struct HIPGuardMasqueradingAsCUDA {
    guard: InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA>,
}

impl HIPGuardMasqueradingAsCUDA {
    
    pub fn new(device_index: DeviceIndex) -> Self {
    
        todo!();
        /*
        : guard(device_index),

        
        */
    }
    
    pub fn new(device: Device) -> Self {
    
        todo!();
        /*
        : guard(device),

        
        */
    }
    
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.set_device(device);
        */
    }
    
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }
    
    pub fn set_index(&mut self, device_index: DeviceIndex)  {
        
        todo!();
        /*
            guard_.set_index(device_index);
        */
    }
    
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }
    
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
}

pub struct OptionalHIPGuardMasqueradingAsCUDA {
    guard: InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA>,
}

impl OptionalHIPGuardMasqueradingAsCUDA {

    pub fn new() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }
    
    pub fn new(device_opt: Option<Device>) -> Self {
    
        todo!();
        /*
        : guard(device_opt),

        
        */
    }
    
    pub fn new(device_index_opt: Option<DeviceIndex>) -> Self {
    
        todo!();
        /*
        : guard(device_index_opt),

        
        */
    }
    
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.set_device(device);
        */
    }
    
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }
    
    pub fn set_index(&mut self, device_index: DeviceIndex)  {
        
        todo!();
        /*
            guard_.set_index(device_index);
        */
    }
    
    pub fn original_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }
    
    pub fn current_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
    
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            guard_.reset();
        */
    }
}

pub struct HIPStreamGuardMasqueradingAsCUDA {
    guard: InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA>,
}

impl HIPStreamGuardMasqueradingAsCUDA {
    
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : guard(stream),
        */
    }
    
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            guard_.reset_stream(stream);
        */
    }
    
    pub fn original_stream(&self) -> HIPStreamMasqueradingAsCUDA {
        
        todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.original_stream());
        */
    }
    
    pub fn current_stream(&self) -> HIPStreamMasqueradingAsCUDA {
        
        todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, guard_.current_stream());
        */
    }
    
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
    
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }
}

pub struct OptionalHIPStreamGuardMasqueradingAsCUDA {
    guard: InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA>,
}

impl OptionalHIPStreamGuardMasqueradingAsCUDA {

    pub fn new() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }
    
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : guard(stream),

        
        */
    }
    
    pub fn new(stream_opt: Option<Stream>) -> Self {
    
        todo!();
        /*
        : guard(stream_opt),

        
        */
    }
    
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            guard_.reset_stream(stream);
        */
    }
    
    pub fn original_stream(&self) -> Option<HIPStreamMasqueradingAsCUDA> {
        
        todo!();
        /*
            auto r = guard_.original_stream();
        if (r.has_value()) {
          return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
        } else {
          return nullopt;
        }
        */
    }
    
    pub fn current_stream(&self) -> Option<HIPStreamMasqueradingAsCUDA> {
        
        todo!();
        /*
            auto r = guard_.current_stream();
        if (r.has_value()) {
          return make_optional(HIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA::UNCHECKED, r.value()));
        } else {
          return nullopt;
        }
        */
    }
    
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            guard_.reset();
        */
    }
}

pub struct HIPMultiStreamGuardMasqueradingAsCUDA {
    guard: InlineMultiStreamGuard<HIPGuardImplMasqueradingAsCUDA>,
}

impl HIPMultiStreamGuardMasqueradingAsCUDA {
    
    pub fn new(streams: &[HIPStreamMasqueradingAsCUDA]) -> Self {
    
        todo!();
        /*
        : guard(unwrapStreams(streams)),

        
        */
    }
    
    pub fn unwrap_streams(hip_streams: &[HIPStreamMasqueradingAsCUDA]) -> Vec<Stream> {
        
        todo!();
        /*
            vector<Stream> streams;
        streams.reserve(hipStreams.size());
        for (const HIPStreamMasqueradingAsCUDA& hipStream : hipStreams) {
          streams.push_back(hipStream);
        }
        return streams;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.cpp]

/**
  | THIS IS A MASSIVE HACK.  This will BREAK you
  | Caffe2 CUDA code if you load ATen_hip, even if
  | you don't ever actually use ATen_hip at
  | runtime.
  |
  | If you ever link ATen_hip statically into the
  | full library along with ATen_cuda (libomnibus),
  | the loading order of this versus the regular
  | ATen_cuda will be nondeterministic, and you'll
  | nondeterministically get one or the other.
  | (This will be obvious because all of your code
  | will fail.)
  |
  | This hack can be removed once PyTorch is
  | out-of-place HIPified, and doesn't pretend CUDA
  | is HIP.
  */
lazy_static!{
    /*
    c10_register_guard_impl!{
        CUDA, 
        HIPGuardImplMasqueradingAsCUDA
    }
    */
}

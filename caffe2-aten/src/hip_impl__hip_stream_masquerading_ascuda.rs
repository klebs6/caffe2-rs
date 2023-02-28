/*!
  | Use of hip namespace here makes hipification
  | easier, because I don't have to also
  | fix namespaces. Sorry!
  |
  */

crate::ix!();

pub enum Unchecked { UNCHECKED }

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h]

/// See Note [Masquerading as CUDA] for motivation
///
#[derive(Hash)]
pub struct HIPStreamMasqueradingAsCUDA {
    stream: HIPStream,
}

impl PartialEq<HIPStreamMasqueradingAsCUDA> for HIPStreamMAsqueradingAsCUDA {
    
    #[inline] fn eq(&self, other: &HIPStreamMasqueradingAsCUDA) -> bool {
        todo!();
        /*
            return stream_ == other.stream_;
        */
    }
}

impl HIPStreamMasqueradingAsCUDA {
    
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : hip_stream_masquerading_ascuda(UNCHECKED, stream),

            // We did the coercion unchecked; check that it was right.
        TORCH_CHECK(stream.device().is_cuda() /* !!! */);
        */
    }
    
    pub fn new(
        _0:     Unchecked,
        stream: Stream) -> Self {
    
        todo!();
        /*
        : stream(HIPStream(
                  Stream(
                    Stream::UNSAFE,
                    Device(DeviceType_HIP, stream.device_index()),
                    stream.id())
                )
              ),

            // Unsafely coerce the "CUDA" stream into a HIP stream
        */
    }

    /// New constructor, just for this.  Does NOT
    /// coerce.
    ///
    pub fn new(stream: HIPStream) -> Self {
    
        todo!();
        /*
        : stream(stream),

        
        */
    }
    
    pub fn operator_hip_stream_t(&self) -> HipStream {
        
        todo!();
        /*
            return stream_.stream();
        */
    }
    
    pub fn operator_stream(&self) -> Stream {
        
        todo!();
        /*
            // Unsafely coerce HIP stream into a "CUDA" stream
        return Stream(Stream::UNSAFE, device(), id());
        */
    }
    
    pub fn device_index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return stream_.device_index();
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            // Unsafely coerce HIP device into CUDA device
        return Device(DeviceType_CUDA, stream_.device_index());
        */
    }
    
    pub fn id(&self) -> StreamId {
        
        todo!();
        /*
            return stream_.id();
        */
    }
    
    pub fn query(&self) -> bool {
        
        todo!();
        /*
            return stream_.query();
        */
    }
    
    pub fn synchronize(&self)  {
        
        todo!();
        /*
            stream_.synchronize();
        */
    }
    
    pub fn priority(&self) -> i32 {
        
        todo!();
        /*
            return stream_.priority();
        */
    }
    
    pub fn stream(&self) -> hip::Stream {
        
        todo!();
        /*
            return stream_.stream();
        */
    }
    
    pub fn unwrap(&self) -> Stream {
        
        todo!();
        /*
            // Unsafely coerce HIP stream into "CUDA" stream
        return Stream(Stream::UNSAFE, device(), id());
        */
    }
    
    pub fn pack(&self) -> u64 {
        
        todo!();
        /*
            // Unsafely coerce HIP stream into "CUDA" stream before packing
        return unwrap().pack();
        */
    }
    
    pub fn unpack(bits: u64) -> HIPStreamMasqueradingAsCUDA {
        
        todo!();
        /*
            // NB: constructor manages CUDA->HIP translation for us
        return HIPStreamMasqueradingAsCUDA(Stream::unpack(bits));
        */
    }
    
    pub fn priority_range() -> (i32,i32) {
        
        todo!();
        /*
            return HIPStream::priority_range();
        */
    }
    
    /**
      | New method, gets the underlying HIPStream
      |
      */
    pub fn hip_stream(&self) -> HIPStream {
        
        todo!();
        /*
            return stream_;
        */
    }
}

#[inline] pub fn get_stream_from_pool_masquerading_ascuda(
        is_high_priority: bool,
        device:           DeviceIndex) -> HIPStreamMasqueradingAsCUDA {

    let is_high_priority: bool = is_high_priority.unwrap_or(false);

    let device: DeviceIndex = device.unwrap_or(-1);

    todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(getStreamFromPool(isHighPriority, device));
        */
}

#[inline] pub fn get_stream_from_external_masquerading_ascuda(
        ext_stream: hip::Stream,
        device:     DeviceIndex) -> HIPStreamMasqueradingAsCUDA {
    
    todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(getStreamFromExternal(ext_stream, device));
        */
}

#[inline] pub fn get_default_hip_stream_masquerading_ascuda(device_index: DeviceIndex) -> HIPStreamMasqueradingAsCUDA {
    let device_index: DeviceIndex = device_index.unwrap_or(-1);

    todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(getDefaultHIPStream(device_index));
        */
}

#[inline] pub fn get_current_hip_stream_masquerading_ascuda(device_index: DeviceIndex) -> HIPStreamMasqueradingAsCUDA {
    let device_index: DeviceIndex = device_index.unwrap_or(-1);

    todo!();
        /*
            return HIPStreamMasqueradingAsCUDA(getCurrentHIPStream(device_index));
        */
}

#[inline] pub fn set_current_hip_stream_masquerading_ascuda(stream: HIPStreamMasqueradingAsCUDA)  {
    
    todo!();
        /*
            setCurrentHIPStream(stream.hip_stream());
        */
}

impl fmt::Display for HIPStreamMasqueradingAsCUDA {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            stream << s.hip_stream() << " (masquerading as CUDA)";
      return stream;
        */
    }
}

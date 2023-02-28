/*!
  | This code is kind of boilerplatey. See
  | Note [Whither the DeviceGuard boilerplate]
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAGuard.h]

/**
  | A variant of DeviceGuard that is specialized
  | for Cuda.  It accepts integer indices
  | (interpreting them as Cuda devices) and is
  | a little more efficient than DeviceGuard (it
  | compiles to straight line
  | cudaSetDevice/cudaGetDevice calls);
  |
  | however, it can only be used from code that
  | links against Cuda directly.
  |
  | No default constructor; see Note [Omitted
  | default constructor from RAII]
  */
#[cfg(feature = "cuda")]
pub struct CUDAGuard {

    /**
      | The guard for the current device.
      |
      */
    guard: InlineDeviceGuard<CUDAGuardImpl>,
}

#[cfg(feature = "cuda")]
impl CUDAGuard {

    /// Set the current Cuda device to the passed
    /// device index.
    ///
    pub fn new(device_index: DeviceIndex) -> Self {
    
        todo!();
        /*
        : guard(device_index),

        
        */
    }

    /**
      | Sets the current Cuda device to the passed
      | device.  Errors if the passed device is not
      | a Cuda device.
      |
      */
    pub fn new(device: Device) -> Self {
    
        todo!();
        /*
        : guard(device),

        
        */
    }

    /**
      | Sets the Cuda device to the given device.
      | Errors if the given device is not a Cuda
      | device.
      |
      */
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.set_device(device);
        */
    }

    /**
      | Sets the Cuda device to the given device.
      | Errors if the given device is not a Cuda
      | device.  (This method is provided for
      | uniformity with DeviceGuard).
      */
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }

    /// Sets the Cuda device to the given device
    /// index.
    ///
    pub fn set_index(&mut self, device_index: DeviceIndex)  {
        
        todo!();
        /*
            guard_.set_index(device_index);
        */
    }

    /// Returns the device that was set upon
    /// construction of the guard
    ///
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }

    /**
      | Returns the last device that was set via
      | `set_device`, if any, otherwise the device
      | passed during construction.
      |
      */
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
}

/**
  | A variant of OptionalDeviceGuard that is
  | specialized for Cuda.
  |
  | See CUDAGuard for when you can use this.
  |
  */
#[cfg(feature = "cuda")]
pub struct OptionalCUDAGuard {

    guard: InlineOptionalDeviceGuard<CUDAGuardImpl>,
}

#[cfg(feature = "cuda")]
impl OptionalCUDAGuard {

    /// Create an uninitialized OptionalCUDAGuard.
    ///
    pub fn new() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }

    /**
      | Set the current Cuda device to the passed
      | Device, if it is not nullopt.
      |
      */
    pub fn new(device_opt: Option<Device>) -> Self {
    
        todo!();
        /*
        : guard(device_opt),

        
        */
    }

    /**
      | Set the current Cuda device to the passed
      | device index, if it is not nullopt
      |
      */
    pub fn new(device_index_opt: Option<DeviceIndex>) -> Self {
    
        todo!();
        /*
        : guard(device_index_opt),

        
        */
    }

    /**
      | Sets the Cuda device to the given device,
      | initializing the guard if it is not already
      | initialized.  Errors if the given device is
      | not a Cuda device.
      |
      */
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.set_device(device);
        */
    }

    /**
      | Sets the Cuda device to the given device,
      | initializing the guard if it is not already
      | initialized.
      |
      | Errors if the given device is not a Cuda
      | device. (This method is provided for
      | uniformity with OptionalDeviceGuard).
      |
      */
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }

    /**
      | Sets the Cuda device to the given device
      | index, initializing the guard if it is not
      | already initialized.
      |
      */
    pub fn set_index(&mut self, device_index: DeviceIndex)  {
        
        todo!();
        /*
            guard_.set_index(device_index);
        */
    }

    /**
      | Returns the device that was set immediately
      | prior to initialization of the guard, or
      | nullopt if the guard is uninitialized.
      |
      */
    pub fn original_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
      | construction, or via set_device, if the
      | guard is initialized, or nullopt if the
      | guard is uninitialized.
      |
      */
    pub fn current_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }

    /**
      | Restore the original Cuda device, resetting
      | this guard to uninitialized state.
      |
      */
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            guard_.reset();
        */
    }
}

/**
  | A variant of StreamGuard that is specialized
  | for Cuda.  See CUDAGuard for when you can use
  | this.
  |
  */
#[cfg(feature = "cuda")]
pub struct CudaStreamGuard {
    guard: InlineStreamGuard<CUDAGuardImpl>,
}

#[cfg(feature = "cuda")]
impl CudaStreamGuard {

    /**
      | Set the current Cuda device to the device
      | associated with the passed stream, and set
      | the current Cuda stream on that device to
      | the passed stream.
      |
      | Errors if the Stream is not a Cuda stream.
      */
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : guard(stream),

        
        */
    }

    /**
      | Resets the currently set stream to the
      | original stream and the currently set device
      | to the original device.
      |
      | Then, set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream.
      |
      | Errors if the stream passed is not a Cuda
      | stream.
      |
      | NOTE: this implementation may skip some
      | stream/device setting if it can prove that
      | it is unnecessary.
      |
      | WARNING: reset_stream does NOT preserve
      | previously set streams on different devices.
      | If you need to set streams on multiple
      | devices on Cuda, use CUDAMultiStreamGuard
      | instead.
      */
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            guard_.reset_stream(stream);
        */
    }

    /**
      | Returns the Cuda stream that was set
      | at the time the guard was constructed.
      |
      */
    #[cfg(feature = "cuda")]
    pub fn original_stream(&self) -> CudaStream {
        
        todo!();
        /*
            return CudaStream(CudaStream::UNCHECKED, guard_.original_stream());
        */
    }

    /**
      | Returns the most recent Cuda stream that was
      | set using this device guard, either from
      | construction, or via set_stream.
      |
      */
    #[cfg(feature = "cuda")]
    pub fn current_stream(&self) -> CudaStream {
        
        todo!();
        /*
            return CudaStream(CudaStream::UNCHECKED, guard_.current_stream());
        */
    }

    /**
      | Returns the most recent Cuda device that was
      | set using this device guard, either from
      | construction, or via
      | set_device/reset_device/set_index.
      |
      */
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }

    /**
      | Returns the Cuda device that was set at the
      | most recent reset_stream(), or otherwise the
      | device at construction time.
      |
      */
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }
}

/**
  | A variant of OptionalStreamGuard that is
  | specialized for Cuda.  See CUDAGuard for when
  | you can use this.
  |
  */
#[cfg(feature = "cuda")]
pub struct OptionalCudaStreamGuard {
    guard: InlineOptionalStreamGuard<CUDAGuardImpl>,
}

#[cfg(feature = "cuda")]
impl OptionalCudaStreamGuard {

    /// Create an uninitialized guard.
    pub fn new() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }

    /**
      | Set the current Cuda device to the device
      | associated with the passed stream, and set
      | the current Cuda stream on that device to
      | the passed stream.
      |
      | Errors if the Stream is not a Cuda stream.
      |
      */
    pub fn new(stream: Stream) -> Self {
    
        todo!();
        /*
        : guard(stream),

        
        */
    }

    /**
      | Set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream, if the passed stream is not
      | nullopt.
      |
      */
    pub fn new(stream_opt: Option<Stream>) -> Self {
    
        todo!();
        /*
        : guard(stream_opt),

        
        */
    }

    /**
      | Resets the currently set Cuda stream to
      | the original stream and the currently set
      | device to the original device.
      |
      | Then, set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream.
      |
      | Initializes the guard if it was not
      | previously initialized.
      */
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            guard_.reset_stream(stream);
        */
    }

    /**
      | Returns the Cuda stream that was set at the
      | time the guard was most recently
      | initialized, or nullopt if the guard is
      | uninitialized.
      |
      */
    pub fn original_stream(&self) -> Option<CudaStream> {
        
        todo!();
        /*
            auto r = guard_.original_stream();
        if (r.has_value()) {
          return make_optional(CudaStream(CudaStream::UNCHECKED, r.value()));
        } else {
          return nullopt;
        }
        */
    }

    /**
      | Returns the most recent Cuda stream that
      | was set using this stream guard, either
      | from construction, or via reset_stream, if
      | the guard is initialized, or nullopt if
      | the guard is uninitialized.
      |
      */
    pub fn current_stream(&self) -> Option<CudaStream> {
        
        todo!();
        /*
            auto r = guard_.current_stream();
        if (r.has_value()) {
          return make_optional(CudaStream(CudaStream::UNCHECKED, r.value()));
        } else {
          return nullopt;
        }
        */
    }

    /**
      | Restore the original Cuda device and
      | stream, resetting this guard to uninitialized
      | state.
      |
      */
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            guard_.reset();
        */
    }
}

/**
  | A variant of MultiStreamGuard that
  | is specialized for Cuda.
  |
  */
#[cfg(feature = "cuda")]
pub struct CUDAMultiStreamGuard {
    guard: InlineMultiStreamGuard<CUDAGuardImpl>,
}

#[cfg(feature = "cuda")]
impl CUDAMultiStreamGuard {
    
    pub fn new(streams: &[CudaStream]) -> Self {
    
        todo!();
        /*
        : guard(unwrapStreams(streams)),

        
        */
    }
    
    pub fn unwrap_streams(cuda_streams: &[CudaStream]) -> Vec<Stream> {
        
        todo!();
        /*
            vector<Stream> streams;
        streams.reserve(cudaStreams.size());
        for (const CudaStream& cudaStream : cudaStreams) {
          streams.push_back(cudaStream);
        }
        return streams;
        */
    }
}

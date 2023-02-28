crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/StreamGuard.h]

/**
  | A StreamGuard is an RAII class that changes
  | the current device to the device corresponding
  | to some stream, and changes the default
  | stream on that device to be this stream.
  | 
  | Use of StreamGuard is HIGHLY discouraged
  | in operator definitions. In a single
  | operator, you probably don't know enough
  | about the global state of the world to
  | profitably decide how to set streams.
  | Let the caller handle this appropriately,
  | and just use the current stream in your
  | operator code.
  | 
  | This StreamGuard does NOT have an uninitialized
  | state; it is guaranteed to reset the
  | stream and device on exit. If you are
  | in a situation where you *might* want
  | to setup a stream guard, see OptionalStreamGuard.
  |
  | Copy is disallowed
  |
  | Move is disallowed, as StreamGuard does not
  | have an uninitialized state, which is
  | required for moves on types with nontrivial
  | destructors.
  */
pub struct StreamGuard {
    guard: InlineStreamGuard<VirtualGuardImpl>,
}

impl StreamGuard {

    /**
      | Set the current device to the device
      | associated with the passed stream, and set
      | the current  stream on that device to the
      | passed stream.
      |
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
      | to the original device.  Then, set the
      | current device to the device associated with
      | the passed stream, and set the current
      | stream on that device to the passed stream.
      |
      | NOTE: this implementation may skip some
      | stream/device setting if it can prove that
      | it is unnecessary.
      |
      | WARNING: reset_stream does NOT preserve
      | previously set streams on different devices.
      | If you need to set streams on multiple
      | devices on , use MultiStreamGuard instead.
      */
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            guard_.reset_stream(stream);
        */
    }

    /**
      | Returns the stream that was set at the
      | time the guard was constructed.
      |
      */
    pub fn original_stream(&self) -> Stream {
        
        todo!();
        /*
            return guard_.original_stream();
        */
    }

    /**
      | Returns the most recent stream that was set
      | using this device guard, either from
      | construction, or via set_stream.
      |
      */
    pub fn current_stream(&self) -> Stream {
        
        todo!();
        /*
            return guard_.current_stream();
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
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
      | Returns the device that was set at the most
      | recent reset_stream(), or otherwise the
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
  | An OptionalStreamGuard is an RAII class
  | that sets a device to some value on initialization,
  | and resets the device to its original
  | value on destruction.
  | 
  | See OptionalDeviceGuard for more guidance
  | on how to use this class.
  |
  */
pub struct OptionalStreamGuard {
    guard: InlineOptionalStreamGuard<VirtualGuardImpl>,
}

impl Default for OptionalStreamGuard {

    /// Create an uninitialized guard.
    ///
    fn default() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }
}

impl From<Stream> for OptionalStreamGuard {

    /**
      | Set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream.
      |
      */
    fn from(x: Stream) -> Self {
        todo!();
        /*
        : guard(stream),
        */
    }
}

impl From<Option<Stream>> for OptionalStreamGuard {

    /**
      | Set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream, if the passed stream is not
      | nullopt.
      |
      */
    fn from(x: Option<Stream>) -> Self {

        todo!();
        /*
        : guard(stream_opt),
        */
    }
}

impl OptionalStreamGuard {

    /**
      | Resets the currently set stream to the
      | original stream and the currently set
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
      | Returns the stream that was set at the time
      | the guard was most recently initialized, or
      | nullopt if the guard is uninitialized.
      |
      */
    pub fn original_stream(&self) -> Option<Stream> {
        
        todo!();
        /*
            return guard_.original_stream();
        */
    }

    /**
      | Returns the most recent  stream that was
      | set using this stream guard, either from
      | construction, or via reset_stream, if the
      | guard is initialized, or nullopt if the
      | guard is uninitialized.
      |
      */
    pub fn current_stream(&self) -> Option<Stream> {
        
        todo!();
        /*
            return guard_.current_stream();
        */
    }

    /**
      | Restore the original  device and stream,
      | resetting this guard to uninitialized
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
  | A MultiStreamGuard is an RAII class
  | that sets the current streams of a set
  | of devices all at once, and resets them
  | to their original values on destruction.
  |
  */
pub struct MultiStreamGuard {
    guard: InlineMultiStreamGuard<VirtualGuardImpl>,
}

impl MultiStreamGuard {

    /**
      | Set the current streams to the passed
      | streams on each of their respective
      | devices.
      |
      */
    pub fn new(streams: &[Stream]) -> Self {
    
        todo!();
        /*
        : guard(streams),

        
        */
    }
}

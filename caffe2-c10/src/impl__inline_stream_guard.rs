crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/InlineStreamGuard.h]

/**
  | A StreamGuard is an RAII class that changes
  | the current device to the device corresponding
  | to some stream, and changes the default
  | stream on that device to be this stream.
  | 
  | InlineStreamGuard is a helper class
  | for implementing StreamGuards.
  | 
  | See InlineDeviceGuard for guidance
  | on how to use this class.
  |
  | Copy is disallowed
  |
  | Move is disallowed, as StreamGuard does not
  | have an uninitialized state, which is
  | required for moves on types with nontrivial
  | destructors.
  |
  */
pub struct InlineStreamGuard<T> {
    base: InlineDeviceGuard<T>,

    /**
      | what the user probably cares about
      |
      */
    original_stream_of_original_device: Stream,


    /**
      | what we need to restore
      |
      */
    original_stream_of_current_device:  Stream,

    current_stream:                     Stream,
}

impl<T> Drop for InlineStreamGuard<T> {

    fn drop(&mut self) {
        todo!();
        /*
            this->impl_.exchangeStream(original_stream_of_current_device_);
        */
    }
}

impl<T> InlineStreamGuard<T> {

    /**
      | Set the current device to the device
      | associated with the passed stream, and set
      | the current stream on that device to the
      | passed stream.
      |
      */
    pub fn new_from_stream(stream: Stream) -> Self {
    
        todo!();
        /*


            : InlineDeviceGuard<T>(stream.device()),
            original_stream_of_original_device_(
                this->impl_.getStream(original_device())),
            original_stream_of_current_device_(this->impl_.exchangeStream(stream)),
            current_stream_(stream)
        */
    }

    /**
      | This constructor exists purely for
      | testing
      |
      */
    /// template < typename U = T, typename = typename enable_if< is_same<U, VirtualGuardImpl>::value>::type>
    pub fn new_from_stream_with_guard(
        stream: Stream,
        impl_:  *const dyn DeviceGuardImplInterface) -> Self {
    
        todo!();
        /*


            : InlineDeviceGuard<T>(
                stream.device(),
                impl ? impl : getDeviceGuardImpl(stream.device_type())),
            original_stream_of_original_device_(
                this->impl_.getStream(original_device())),
            original_stream_of_current_device_(this->impl_.exchangeStream(stream)),
            current_stream_(stream)
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
      | devices use MultiStreamGuard instead.
      */
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            // TODO: make a version that takes an impl argument.  Unfortunately,
        // that will require SFINAE because impl is only valid for the
        // VirtualGuardImpl specialization.
        if (stream.device() == this->current_device()) {
          this->impl_.exchangeStream(stream);
          current_stream_ = stream;
        } else {
          // Destruct and reconstruct the StreamGuard in-place
          this->impl_.exchangeStream(original_stream_of_current_device_);
          this->reset_device(stream.device());
          original_stream_of_current_device_ = this->impl_.exchangeStream(stream);
          current_stream_ = stream;
        }
        */
    }

    /**
      | It's not clear if set_device should also
      | reset the current stream if the device is
      | unchanged; therefore, we don't provide it.
      |
      | The situation is somewhat clearer with
      | reset_device, but it's still a pretty weird
      | thing to do, so haven't added this either.
      |
      | Returns the stream of the original device
      | prior to this guard.  Subtly, the stream
      | returned here is the original stream of the
      | *original* device; i.e., it's the stream
      | that your computation *would* have been put
      | on, if it hadn't been for this meddling
      | stream guard.
      |
      | This is usually what you want.
      |
      */
    pub fn original_stream(&self) -> Stream {
        
        todo!();
        /*
            return original_stream_of_original_device_;
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
            return current_stream_;
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
            return InlineDeviceGuard<T>::current_device();
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
            return InlineDeviceGuard<T>::original_device();
        */
    }
}

/**
  | An OptionalStreamGuard is an RAII class
  | that sets a device to some value on initialization,
  | and resets the device to its original
  | value on destruction.
  | 
  | See InlineOptionalDeviceGuard for
  | more guidance on how to use this class.
  |
  */
pub struct InlineOptionalStreamGuard<T> {
    guard: Option<InlineStreamGuard<T>>,
}

impl<T> InlineOptionalStreamGuard<T> {

    /// Creates an uninitialized stream guard.
    pub fn new_default() -> Self {
    
        todo!();
        /*


            : guard_() // See Note [Explicit initialization of optional fields]
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
    pub fn new_from_maybe_stream(stream_opt: Option<Stream>) -> Self {
    
        todo!();
        /*
        : guard(),

            if (stream_opt.has_value()) {
          guard_.emplace(stream_opt.value());
        }
        */
    }

    /// All constructors of StreamGuard are valid
    /// for OptionalStreamGuard
    ///
    pub fn new_from_args<Args>(args: Args) -> Self {
    
        todo!();
        /*


            : guard_(in_place, forward<Args>(args)...)
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
      | Initializes the OptionalStreamGuard if it
      | was not previously initialized.
      |
      */
    pub fn reset_stream(&mut self, stream: Stream)  {
        
        todo!();
        /*
            if (guard_.has_value()) {
          guard_->reset_stream(stream);
        } else {
          guard_.emplace(stream);
        }
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
            return guard_.has_value() ? make_optional(guard_->original_stream())
                                  : nullopt;
        */
    }

    /**
      | Returns the most recent stream that was set
      | using this stream guard, either from
      | construction, or via reset_stream, if the
      | guard is initialized, or nullopt if the
      | guard is uninitialized.
      |
      */
    pub fn current_stream(&self) -> Option<Stream> {
        
        todo!();
        /*
            return guard_.has_value() ? make_optional(guard_->current_stream())
                                  : nullopt;
        */
    }

    /**
      | Restore the original device and stream,
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
  | Copy is disallowed
  |
  | Move is disallowed, as StreamGuard does not
  | have an uninitialized state, which is
  | required for moves on types with nontrivial
  | destructors.
  |
  */
pub struct InlineMultiStreamGuard<T> {

    impl_:            Option<T>,

    /**
      | The original streams that were active
      | on all devices.
      |
      */
    original_streams: Vec<Stream>,
}

impl<T> Drop for InlineMultiStreamGuard<T> {

    fn drop(&mut self) {
        todo!();
        /*
            for (const Stream& s : original_streams_) {
          this->impl_->exchangeStream(s);
        }
        */
    }
}

impl<T> InlineMultiStreamGuard<T> {

    /**
      | Calls `set_stream` on each of the streams in
      | the list.
      |
      | This may be useful if you need to set
      | different streams for different devices.
      */
    pub fn new_from_streams(streams: &[Stream]) -> Self {
    
        todo!();
        /*


            if (!streams.empty()) {
          impl_.emplace(getDeviceTypeOfStreams(streams));
          original_streams_.reserve(streams.size());
          for (const Stream& s : streams) {
            original_streams_.push_back(this->impl_->exchangeStream(s));
          }
        }
        */
    }
    
    pub fn get_device_type_of_streams(streams: &[Stream]) -> DeviceType {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!streams.empty());
        DeviceType type = streams[0].device_type();
        for (size_t idx = 1; idx < streams.size(); idx++) {
          TORCH_CHECK_VALUE(
              streams[idx].device_type() == type,
              "Streams have a mix of device types: stream 0 is on ",
              streams[0].device(),
              " while stream ",
              idx,
              " is on device ",
              streams[idx].device());
        }
        return type;
        */
    }
}

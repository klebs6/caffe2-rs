crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/DeviceGuardImplInterface.h]

/**
  | Flags defining the behavior of events.
  | 
  | PYTORCH_DEFAULT and BACKEND_DEFAULT
  | are valid for all backends. The
  | 
  | BACKEND_DEFAULT is what a particular
  | backend would select if no flags were
  | given.
  | 
  | PYTORCH_DEFAULT is the PyTorch's framework
  | default choice for events on that backend,
  | which may not be the same.
  | 
  | For example, when PyTorch creates a
  | Cuda event it sets the flag
  | 
  | CUDA_EVENT_DISABLING_TIMING by default
  | to improve performance.
  | 
  | The mapping of PYTORCH_DEFAULT and
  | BACKEND_DEFAULT is done by each backend
  | implementation. Backend-specific
  | flags, like CUDA_EVENT_DEFAULT, should
  | map one-to-one with actual event flags
  | for those backends.
  |
  */
pub enum EventFlag {
    PYTORCH_DEFAULT,
    BACKEND_DEFAULT,
    // Cuda flags
    CUDA_EVENT_DEFAULT,
    CUDA_EVENT_DISABLE_TIMING, // PyTorch-default for Cuda HIP flags
    HIP_EVENT_DEFAULT,
    HIP_EVENT_DISABLE_TIMING, // PyTorch-default for HIP FOR TESTING ONLY
    INVALID
}

/**
  | DeviceGuardImplInterface represents
  | the virtual interface which provides
  | functionality to provide an RAII class
  | for device and stream switching, via
  | DeviceGuard. Every distinct device
  | type, e.g., Cuda and HIP, is expected
  | to implement and register an implementation
  | of this interface.
  | 
  | All classes which inherit from DeviceGuardImplInterface
  | should be declared 'final'.
  | 
  | This class exists because we provide
  | a unified interface for performing
  | device guards via DeviceGuard, but
  | we cannot assume that we have actually
  | compiled against the, e.g., Cuda library,
  | which actually implements this guard
  | functionality. In this case, a dynamic
  | dispatch is required to cross the library
  | boundary.
  | 
  | If possible, you should directly use
  | implementations of this interface;
  | those uses will be devirtualized.
  |
  | Intended use of this class is to leak
  | the DeviceGuardImpl at program end.
  | 
  | So you better not call the destructor,
  | buster!
  |
  | NB: Implementations of exchangeDevice can be
  | a bit boilerplatey.
  |
  | You might consider replacing exchangeDevice
  | with a non-virtual function with a baked in
  | implementation; however, note that this will
  | triple the number of virtual calls (when you
  | implement exchangeDevice in a final subclass,
  | the compiler gets to devirtualize everything;
  | it won't do that if you don't define it in the
  | subclass!)
  |
  | A common way to solve this problem is to use
  | some sort of CRTP; however, we can template
  | DeviceGuardImplInterface since we really *do*
  | need it to be virtual.
  |
  | A little boilerplate seems easiest to explain.
  | (Another way around this problem is to provide
  | inline functions that provide the default
  | implementations, but this seems a little hard
  | to explain.
  |
  | In any case, we're only going to have on order
  | of ten implementations of this anyway.)
  |
  */
pub trait DeviceGuardImplInterface:
Ty
+ ExchangeDevice
+ GetDevice
+ SetDevice
+ UncheckedSetDevice
+ GetStream
+ ExchangeStream
+ DeviceCount 
+ RecordDataPtrOnStream {

    /**
      | Get the default stream for a given device.
      |
      */
    fn get_default_stream(&self, _0: Device) -> Stream {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support acquiring a default stream.")
        */
    }

    /**
      | Get a stream from the global pool for
      | a given device.
      |
      */
    fn get_stream_from_global_pool(&self, 
        _0:               Device,
        is_high_priority: Option<bool>) -> Stream {

        let is_high_priority: bool = is_high_priority.unwrap_or(false);

        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support acquiring a stream from pool.")
        */
    }

    /**
      | Destroys the given event.
      |
      */
    fn destroy_event(&self, 
        event:        *mut c_void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
        
        */
    }

    /**
      | Increments the event's version and
      | enqueues a job with this version in the
      | stream's work queue. When the stream
      | process that job it notifies all streams
      | waiting on / blocked by that version
      | of the event to continue and marks that
      | version as recorded.
      |
      */
    fn record(&self, 
        event:        *mut *mut c_void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support events.");
        */
    }

    /**
      | Does nothing if the event has not been
      | scheduled to be recorded.
      | 
      | If the event was previously enqueued
      | to be recorded, a command to wait for
      | the version of the event that exists
      | at the time of this call is inserted in
      | the stream's work queue.
      | 
      | When the stream reaches this command
      | it will stop processing additional
      | commands until that version of the event
      | is marked as recorded.
      |
      */
    fn block(&self, 
        event:  *mut c_void,
        stream: &Stream)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support events.");
        */
    }

    /**
      | Returns true if (and only if)
      | 
      | (1) the event has never been scheduled
      | to be recorded
      | 
      | (2) the current version is marked as
      | recorded.
      | 
      | Returns false otherwise.
      |
      */
    fn query_event(&self, event: *mut c_void) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support events.");
        */
    }

    /**
      | Return true if all the work previously
      | enqueued on the stream for asynchronous
      | execution has completed running on
      | the device.
      |
      */
    fn query_stream(&self, stream: &Stream) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support querying streams.");
        */
    }

    /**
      | Wait (by blocking the calling thread)
      | until all the work previously enqueued
      | on the stream has completed running
      | on the device.
      |
      */
    fn synchronize_stream(&self, stream: &Stream)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Backend doesn't support synchronizing streams.");
        */
    }
}

pub trait RecordDataPtrOnStream {

    /**
      | Ensure the caching allocator (if any)
      | is aware that the given DataPtr is being
      | used on the given stream, and that it
      | should thus avoid recycling the
      | 
      | DataPtr until all work on that stream
      | is done.
      |
      */
    fn record_data_ptr_on_stream(&self, 
        _0: &DataPtr,
        _1: &Stream)  {
        
        todo!();
        /*
        
        */
    }
}

pub trait Ty {

    /**
      | Return the type of device managed by
      | this guard implementation.
      |
      */
    fn ty(&self) -> DeviceType;
}

pub trait ExchangeDevice {

    /**
      | Set the current device to Device, and
      | return the previous Device.
      |
      */
    fn exchange_device(&self, _0: Device) -> Device;
}

pub trait GetDevice {

    /**
      | Get the current device.
      |
      */
    fn get_device(&self) -> Device;
}

pub trait SetDevice {

    /**
      | Set the current device to Device.
      |
      */
    fn set_device(&self, _0: Device);
}

pub trait UncheckedSetDevice {

    /**
      | Set the current device to Device, without
      | checking for errors (so, e.g., this
      | can be called from a destructor).
      |
      */
    fn unchecked_set_device(&self, _0: Device);
}

pub trait GetStream {

    /**
      | Get the current stream for a given device.
      |
      */
    fn get_stream(&self, _0: Device) -> Stream;
}

pub trait ExchangeStream {

    /**
      | Set a stream to be the thread local current
      | stream for its device.
      | 
      | Return the previous stream for that
      | device. You are NOT required to set the
      | current device to match the device of
      | this stream.
      |
      */
    fn exchange_stream(&self, _0: Stream) -> Stream;
}

pub trait DeviceCount {

    /**
      | Get the number of devices. WARNING:
      | This is REQUIRED to not raise an exception.
      | If there is some sort of problem, e.g.,
      | driver error, you should report that
      | there are zero available devices.
      |
      */
    fn device_count(&self) -> DeviceIndex;
}

/**
  | A no-op device guard impl that doesn't do
  | anything interesting.  Useful for devices that
  | don't actually have a concept of device index.
  | Prominent examples are CPU and Meta.
  |
  */
pub struct NoOpDeviceGuardImpl<const D: DeviceType> {
    base: dyn DeviceGuardImplInterface,
}

impl<const D: DeviceType> NoOpDeviceGuardImpl<D> {

    pub fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return D;
        */
    }
    
    pub fn exchange_device(&self, _0: Device) -> Device {
        
        todo!();
        /*
            return Device(D, -1); // no-op
        */
    }
    
    pub fn get_device(&self) -> Device {
        
        todo!();
        /*
            return Device(D, -1);
        */
    }
    
    pub fn set_device(&self, _0: Device)  {
        
        todo!();
        /*
            // no-op
        */
    }
    
    pub fn unchecked_set_device(&self, d: Device)  {
        
        todo!();
        /*
            // no-op
        */
    }
    
    pub fn get_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            // no-op
        return Stream(Stream::DEFAULT, Device(D, -1));
        */
    }

    /// NB: These do NOT set the current device
    ///
    pub fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            // no-op
        return Stream(Stream::DEFAULT, Device(D, -1));
        */
    }
    
    pub fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            return 1;
        */
    }

    /// Event-related functions
    pub fn record(&self, 
        event:        *mut *mut c_void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            TORCH_CHECK(false, D, " backend doesn't support events.");
        */
    }
    
    pub fn block(&self, 
        event:  *mut c_void,
        stream: &Stream)  {
        
        todo!();
        /*
            TORCH_CHECK(false, D, " backend doesn't support events.")
        */
    }
    
    pub fn query_event(&self, event: *mut c_void) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(false, D, " backend doesn't support events.")
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut c_void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
        
        */
    }

    /// Stream-related functions
    pub fn query_stream(&self, stream: &Stream) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn synchronize_stream(&self, stream: &Stream)  {
        
        todo!();
        /*
            // Don't wait for anything.
        */
    }
}

/**
  | The registry is NON-owning.  Each stored
  | pointer is atomic so that under all
  | interleavings of registry calls the structure
  | is race-free.
  |
  | This doesn't cost us anything on reads in X86.
  | (An unsynchronized implementation probably is
  | OK too, but I didn't want to prove that we
  | never read from device_guard_impl_registry at
  | the same time some registration is occurring.
  | Shiver.)
  |
  | I'd like this registry to be valid even at
  | program destruction time (in case someone uses
  | a DeviceGuard in a destructor to do some
  | cleanup in the Cuda API.)
  |
  | Since there are no direct accesses of the
  | underlying owning objects which I can use to
  | enforce initialization order (unlike in a Meyer
  | singleton), it implies that you must *leak*
  | objects when putting them in the registry.
  |
  | This is done by deleting the destructor on
  | DeviceGuardImplInterface.
  |
  */
lazy_static!{
    /*
    extern  atomic<const DeviceGuardImplInterface*>
        device_guard_impl_registry[static_cast<size_t>(
            DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
    */
}

/**
  | I can't conveniently use c10/util/Registry.h
  | for the following reason: c10/util/Registry.h
  | gives me a slow way of Create'ing a object of
  | some interface from the registry, but no way of
  | quickly accessing an already created object.
  |
  | I'll be banging on getDeviceGuardImpl every
  | time we do a DeviceGuard, so I really don't
  | want to be doing an unordered_map
  | lookup. Better if the registration mechanism
  | directly drops its implementation into
  | device_guard_impl_registry.
  */
pub struct DeviceGuardImplRegistrar {

}

#[macro_export] macro_rules! c10_register_guard_impl {
    ($DevType:ty, $DeviceGuardImpl:ty) => {
        /*
        
          static ::DeviceGuardImplRegistrar C10_ANONYMOUS_VARIABLE( 
              g_##DeviceType)(::DeviceType::DevType, new DeviceGuardImpl());
        */
    }
}

#[inline] pub fn get_device_guard_impl(ty: DeviceType) -> *const dyn DeviceGuardImplInterface {
    
    todo!();
        /*
            // Two adjacent int16_t fields DeviceType and DeviceIndex has field access
      // miscompiled on NVCC. To workaround this issue, we apply a mask to the
      // DeviceType. First check if the DeviceType is 16-bit.
      // FB employees can see
      //   https://fb.workplace.com/groups/llvm.gcc/permalink/4053565044692080/
      // for more details
      static_assert(sizeof(DeviceType) == 1, "DeviceType is not 8-bit");
      auto p = device_guard_impl_registry[static_cast<size_t>(type) & 0xFF].load();

      // This seems to be the first place where you make use of a device
      // when you pass devices to factory functions.  Give a nicer error
      // message in this case.
      TORCH_CHECK(p, "PyTorch is not linked with support for ", type, " devices");
      return p;
        */
}

#[inline] pub fn has_device_guard_impl(ty: DeviceType) -> bool {
    
    todo!();
        /*
            return device_guard_impl_registry[static_cast<size_t>(type)].load();
        */
}

//-------------------------------------------[.cpp/pytorch/c10/core/impl/DeviceGuardImplInterface.cpp]

lazy_static!{
    /*
    atomic<const DeviceGuardImplInterface*>
        device_guard_impl_registry[static_cast<size_t>(
            DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
    */
}

impl DeviceGuardImplRegistrar {
    
    pub fn new(
        ty:    DeviceType,
        impl_: *const dyn DeviceGuardImplInterface) -> Self {
    
        todo!();
        /*
            device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
        */
    }
}

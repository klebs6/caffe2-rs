crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/FakeGuardImpl.h]

/**
  | FakeGuardImpl is hardcoded to have eight
  | devices.  Not for any good reason, just to
  | simplify code.
  |
  */
pub const FAKE_GUARD_IMPL_MAX_DEVICES: usize = 8;

/**
  | A fake implementation of
  | DeviceGuardImplInterface suitable
  | for testing.
  | 
  | The current device is modeled as a mutable
  | field in the guard implementation class.
  | 
  | See DeviceGuard_test.cpp for an example
  | use.
  |
  */
pub struct FakeGuardImpl<const T: DeviceType> { }

impl<const T: DeviceType> DeviceGuardImplInterface for FakeGuardImpl<T> {

}

impl<const T: DeviceType> DeviceCount for FakeGuardImpl<T> {

    fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            return kFakeGuardImplMaxDevices;
        */
    }
}

impl<const T: DeviceType> ExchangeStream for FakeGuardImpl<T> {

    fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            auto old_id = current_streams_[s.device_index()];
        current_streams_[s.device_index()] = s.id();
        return Stream(Stream::UNSAFE, s.device(), old_id);
        */
    }
}

impl<const T: DeviceType> ExchangeDevice for FakeGuardImpl<T> {

    fn exchange_device(&self, d: Device) -> Device {
        
        todo!();
        /*
            AT_ASSERT(d.type() == type());
        AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
        Device old_device = getDevice();
        if (old_device.index() != d.index()) {
          current_device_ = d.index();
        }
        return old_device;
        */
    }
}

impl<const T: DeviceType> GetDevice for FakeGuardImpl<T> {

    fn get_device(&self) -> Device {
        
        todo!();
        /*
            return Device(type(), current_device_);
        */
    }
}

impl<const T: DeviceType> SetDevice for FakeGuardImpl<T> {

    fn set_device(&self, d: Device)  {
        
        todo!();
        /*
            AT_ASSERT(d.type() == type());
        AT_ASSERT(d.index() >= 0);
        AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
        current_device_ = d.index();
        */
    }
}

impl<const T: DeviceType> UncheckedSetDevice for FakeGuardImpl<T> {

    fn unchecked_set_device(&self, d: Device)  {
        
        todo!();
        /*
            current_device_ = d.index();
        */
    }
}

impl<const T: DeviceType> GetStream for FakeGuardImpl<T> {

    fn get_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return Stream(Stream::UNSAFE, d, current_streams_[d.index()]);
        */
    }
}

impl<const T: DeviceType> Ty for FakeGuardImpl<T> {

    fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return T;
        */
    }
}

impl<const T: DeviceType> RecordDataPtrOnStream for FakeGuardImpl<T> {

}

impl<const T: DeviceType> FakeGuardImpl<T> {

    lazy_static!{
        /*
        thread_local static DeviceIndex current_device_;
          thread_local static array<StreamId, kFakeGuardImplMaxDevices> current_streams_;
        */
    }

    pub const STATIC_TYPE: DeviceType = T;

    /// Runtime device type is not used
    pub fn new(_0: DeviceType) -> Self {
    
        todo!();
        /*
        
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
        
        */
    }
    
    pub fn block(&self, 
        event:  *mut c_void,
        stream: &Stream)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn query_event(&self, event: *mut c_void) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut c_void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
        
        */
    }

    /// Convenience methods for testing
    ///
    pub fn get_device_index() -> DeviceIndex {
        
        todo!();
        /*
            return current_device_;
        */
    }
    
    pub fn set_device_index(i: DeviceIndex)  {
        
        todo!();
        /*
            AT_ASSERT(i >= 0);
        AT_ASSERT(i < kFakeGuardImplMaxDevices);
        current_device_ = i;
        */
    }
    
    pub fn get_current_stream_id_for(i: DeviceIndex) -> StreamId {
        
        todo!();
        /*
            return current_streams_.at(i);
        */
    }
    
    pub fn reset_streams()  {
        
        todo!();
        /*
            current_streams_.fill(0);
        */
    }
}

lazy_static!{
    /*
    template <DeviceType T>
    thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;

    template <DeviceType T>
    constexpr DeviceType FakeGuardImpl<T>::static_type;

    template <DeviceType T>
    thread_local array<StreamId, kFakeGuardImplMaxDevices>
        FakeGuardImpl<T>::current_streams_ = {0, 0, 0, 0, 0, 0, 0, 0};
    */
}


crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/VirtualGuardImpl.h]

/**
  | An implementation of DeviceGuardImplInterface
  | which delegates to virtual dispatch
  | on the DeviceGuardImpl registry.
  |
  */
pub struct VirtualGuardImpl {
    impl_: Box<dyn DeviceGuardImplInterface>, // default = nullptr
}

impl DeviceGuardImplInterface for VirtualGuardImpl {

}

impl RecordDataPtrOnStream for VirtualGuardImpl {

   fn record_data_ptr_on_stream(&self, 
        data_ptr: &DataPtr,
        stream:   &Stream)  {
        
        todo!();
        /*
            impl_->recordDataPtrOnStream(data_ptr, stream);
        */
    }
}

impl ExchangeStream for VirtualGuardImpl {

    fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            return impl_->exchangeStream(s);
        */
    }
}

impl ExchangeDevice for VirtualGuardImpl {

    fn exchange_device(&self, d: Device) -> Device {
        
        todo!();
        /*
            return impl_->exchangeDevice(d);
        */
    }
}

impl DeviceCount for VirtualGuardImpl {

    fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            return impl_->deviceCount();
        */
    }
}

impl GetStream for VirtualGuardImpl {

    fn get_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return impl_->getStream(d);
        */
    }
}

impl SetDevice for VirtualGuardImpl {

    fn set_device(&self, d: Device)  {
        
        todo!();
        /*
            impl_->setDevice(d);
        */
    }
}

impl GetDevice for VirtualGuardImpl {

    fn get_device(&self) -> Device {
        
        todo!();
        /*
            return impl_->getDevice();
        */
    }
}

impl UncheckedSetDevice for VirtualGuardImpl {

    fn unchecked_set_device(&self, d: Device)  {
        
        todo!();
        /*
            impl_->uncheckedSetDevice(d);
        */
    }
}

impl Ty for VirtualGuardImpl {

    /// Copying and moving is OK!
    fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return impl_->type();
        */
    }
}

impl VirtualGuardImpl {

    pub fn new(device_type: DeviceType) -> Self {
    
        todo!();
        /*
        : impl_(getDeviceGuardImpl(device_type)),

        
        */
    }

    /// This constructor exists purely for testing
    ///
    pub fn new_with_guard(impl_: *const dyn DeviceGuardImplInterface) -> Self {
    
        todo!();
        /*
        : impl_(impl),

        
        */
    }
    
    pub fn get_default_stream(&self, d: Device) -> Stream {
        
        todo!();
        /*
            return impl_->getDefaultStream(d);
        */
    }
    
    pub fn get_stream_from_global_pool(&self, 
        d:                Device,
        is_high_priority: Option<bool>) -> Stream {

        let is_high_priority: bool = is_high_priority.unwrap_or(false);

        todo!();
        /*
            return impl_->getStreamFromGlobalPool(d, isHighPriority);
        */
    }

    /// Event functions
    pub fn record(&self, 
        event:        *mut *mut c_void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            impl_->record(event, stream, device_index, flag);
        */
    }
    
    pub fn block(&self, 
        event:  *mut c_void,
        stream: &Stream)  {
        
        todo!();
        /*
            impl_->block(event, stream);
        */
    }
    
    pub fn query_event(&self, event: *mut c_void) -> bool {
        
        todo!();
        /*
            return impl_->queryEvent(event);
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut c_void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
            impl_->destroyEvent(event, device_index);
        */
    }
    
    pub fn query_stream(&self, stream: &Stream) -> bool {
        
        todo!();
        /*
            return impl_->queryStream(stream);
        */
    }
    
    pub fn synchronize_stream(&self, stream: &Stream)  {
        
        todo!();
        /*
            impl_->synchronizeStream(stream);
        */
    }
}

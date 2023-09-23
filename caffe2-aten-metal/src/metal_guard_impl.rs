crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalGuardImpl.cpp]

#[derive(Default)]
pub struct MetalGuardImpl {
    base: DeviceGuardImplInterface,
}

impl MetalGuardImpl {

    pub fn new(t: DeviceType) -> Self {
    
        todo!();
        /*


            TORCH_INTERNAL_ASSERT(t == DeviceType_Metal);
        */
    }
    
    pub fn ty(&self) -> DeviceType {
        
        todo!();
        /*
            return DeviceType_Metal;
        */
    }
    
    pub fn exchange_device(&self, _0: Device) -> Device {
        
        todo!();
        /*
            // no-op
        return Device(DeviceType_Metal, -1);
        */
    }
    
    pub fn get_device(&self) -> Device {
        
        todo!();
        /*
            return Device(DeviceType_Metal, -1);
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
        return Stream(Stream::DEFAULT, Device(DeviceType_Metal, -1));
        */
    }

    /**
      | NB: These do NOT set the current device
      |
      */
    pub fn exchange_stream(&self, s: Stream) -> Stream {
        
        todo!();
        /*
            // no-op
        return Stream(Stream::DEFAULT, Device(DeviceType_Metal, -1));
        */
    }
    
    pub fn device_count(&self) -> DeviceIndex {
        
        todo!();
        /*
            return 1;
        */
    }

    /**
      | Event-related functions
      |
      */
    pub fn record(&self, 
        event:        *mut *mut void,
        stream:       &Stream,
        device_index: DeviceIndex,
        flag:         EventFlag)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Metal backend doesn't support events.");
        */
    }
    
    pub fn block(&self, 
        event:  *mut void,
        stream: &Stream)  {
        
        todo!();
        /*
            TORCH_CHECK(false, "Metal backend doesn't support events.")
        */
    }
    
    pub fn query_event(&self, event: *mut void) -> bool {
        
        todo!();
        /*
            TORCH_CHECK(false, "Metal backend doesn't support events.")
        */
    }
    
    pub fn destroy_event(&self, 
        event:        *mut void,
        device_index: DeviceIndex)  {
        
        todo!();
        /*
        
        */
    }
}

lazy_static!{
    /*
    c10_register_guard_impl!{Metal, MetalGuardImpl}
    */
}


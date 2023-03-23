crate::ix!();

pub struct CPUEventWrapper {
    mutex:         parking_lot::RawMutex,
    cv_completed:  std::sync::Condvar,
    status:        Atomic<i32>,
    err_msg:       String,
    callbacks:     Vec<EventCallbackFunction>,
}

impl CPUEventWrapper {

    pub fn new(option: &DeviceOption) -> Self {
    
        todo!();
        /*
            : status_(EventStatus::EVENT_INITIALIZED) 

        CAFFE_ENFORCE(
            option.device_type() == PROTO_CPU ||
                option.device_type() == PROTO_MKLDNN ||
                option.device_type() == PROTO_IDEEP,
            "Expected CPU/MKLDNN/IDEEP device type");
        */
    }
}

#[inline] pub fn event_createcpu(option: &DeviceOption, event: *mut Event)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_recordcpu(
    event:   *mut Event,
    unused:  *const c_void,
    err_msg: *const u8)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_finishcpu(event: *const Event)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_waitcpucpu(event: *const Event, context: *mut c_void)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_querycpu(event: *const Event) -> EventStatus {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_error_messagecpu<'a>(event: *const Event) -> &'a String {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_set_finishedcpu(event: *const Event, err_msg: *const u8)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_can_schedulecpu(e1: *const Event, e2: *const Event) -> bool {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn event_resetcpu(e: *mut Event)  {
    
    todo!();
    /*
    
    */
}

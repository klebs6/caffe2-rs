crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/InlineEvent.h]

pub struct InlineEvent<T> {
    event:                    *mut c_void, // default = nullptr
    backend:                  T,
    device_type:              DeviceType,
    device_index:             DeviceIndex, // default = -1

    /**
      | = EventFlag::PYTORCH_DEFAULT;
      |
      */
    flag:                     EventFlag,

    was_marked_for_recording: bool, // default = false
}

impl<T> Drop for InlineEvent<T> {

    fn drop(&mut self) {
        todo!();
        /*
            if (event_)
          backend_.destroyEvent(event_, device_index_);
        */
    }
}

impl<T> InlineEvent<T> {

    pub fn new(
        device_type: DeviceType,
        flag:        Option<EventFlag>) -> Self {

        let flag: EventFlag = flag.unwrap_or(EventFlag::PYTORCH_DEFAULT);

        todo!();
        /*
            : backend_{_device_type}, device_type_{_device_type}, flag_{_flag}
        */
    }

    /// Move constructor and move assignment operator
    ///
    pub fn new_from_inline_event(other: InlineEvent<T>) -> Self {
    
        todo!();
        /*


            : InlineEvent(other.device_type_, other.flag_) 
        swap(move(other));
        */
    }
    
    pub fn assign_from(&mut self, other: InlineEvent<T>) -> &mut InlineEvent<T> {
        
        todo!();
        /*
            swap(move(other));
        return *this;
        */
    }
    
    pub fn swap(&mut self, other: InlineEvent<T>)  {
        
        todo!();
        /*
            swap(event_, other.event_);
        swap(backend_, other.backend_);
        swap(device_type_, other.device_type_);
        swap(device_index_, other.device_index_);
        swap(flag_, other.flag_);
        swap(was_marked_for_recording_, other.was_marked_for_recording_);
        */
    }
    
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return device_type_;
        */
    }
    
    pub fn device_index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return device_index_;
        */
    }
    
    pub fn flag(&self) -> EventFlag {
        
        todo!();
        /*
            return flag_;
        */
    }
    
    
    pub fn was_marked_for_recording(&self) -> bool {
        
        todo!();
        /*
            return was_marked_for_recording_;
        */
    }
    
    
    pub fn record_once(&mut self, stream: &Stream)  {
        
        todo!();
        /*
            if (!was_marked_for_recording_)
          record(stream);
        */
    }
    
    
    pub fn record(&mut self, stream: &Stream)  {
        
        todo!();
        /*
            TORCH_CHECK(
            stream.device_type() == device_type_,
            "Event device type ",
            DeviceTypeName(device_type_),
            " does not match recording stream's device type ",
            DeviceTypeName(stream.device_type()),
            ".");

        backend_.record(&event_, stream, device_index_, flag_);
        was_marked_for_recording_ = true;
        device_index_ = stream.device_index();
        */
    }
    
    
    pub fn block(&self, stream: &Stream)  {
        
        todo!();
        /*
            if (!was_marked_for_recording_)
          return;

        TORCH_CHECK(
            stream.device_type() == device_type_,
            "Event device type ",
            DeviceTypeName(device_type_),
            " does not match blocking stream's device type ",
            DeviceTypeName(stream.device_type()),
            ".");

        backend_.block(event_, stream);
        */
    }
    
    
    pub fn query(&self) -> bool {
        
        todo!();
        /*
            if (!was_marked_for_recording_)
          return true;
        return backend_.queryEvent(event_);
        */
    }
}

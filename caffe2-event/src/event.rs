crate::ix!();

pub struct Event {

    /**
      | event_ is going to be accessed by the
      | EventCreate/Record/Wait/Finish functions,
      | but one should not use it outside the own
      | Event functionalities.
      |
      | In the future we may move it to a private
      | member.
      */
    event:  Arc<c_void>,

    type_:    i32,
    option:  DeviceOption,

    #[cfg(caffe2_use_exception_ptr)]
    caught_exception: *mut c_void,

    error_timestamp:  i64, // default = 0
}

impl Drop for Event {

    /**
      | Nothing needs to be done in the
      | destructor, as the event creator should
      | set the proper destruction process for the
      | unique_ptr.
      */
    fn drop(&mut self) {
        todo!();
        /*  */
    }
}

impl Event {

    pub fn new(option: &DeviceOption) -> Self {
    
        todo!();
        /*
            : event_(), type_(option.device_type()), option_(option) 

        CAFFE_ENFORCE_LT(type_, MaxDeviceTypes);
        CAFFE_ENFORCE(event_creator_[type_]);
        event_creator_[type_](option, this);
        */
    }
    
    #[inline] pub fn record(&mut self, 
        recorder_type: DeviceType,
        context:       *const c_void,
        err_msg:       Option<&str>)  {

        todo!();
        /*
            auto recorder_index = TypeToProto(recorder_type);
        CAFFE_ENFORCE_EQ(
            recorder_index,
            type_,
            "You are trying to record with a wrong device type.");
        CAFFE_ENFORCE(event_recorder_[recorder_index]);
        event_recorder_[recorder_index](this, context, err_msg);
        */
    }
    
    #[inline] pub fn wait(&self, waiter_type: DeviceType, context: *mut c_void)  {
        
        todo!();
        /*
            auto waiter_index = TypeToProto(waiter_type);
        CAFFE_ENFORCE(event_waiter_[waiter_index][type_]);
        event_waiter_[waiter_index][type_](this, context);
        */
    }
    
    #[inline] pub fn finish(&self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_finisher_[type_]);
        event_finisher_[type_](this);
        */
    }
    
    #[inline] pub fn query(&self) -> EventStatus {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_querier_[type_]);
        return event_querier_[type_](this);
        */
    }
    
    #[inline] pub fn error_message(&self) -> &String {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_err_msg_getter_[type_]);
        return event_err_msg_getter_[type_](this);
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_resetter_[type_]);
        event_resetter_[type_](this);
    #ifdef CAFFE2_USE_EXCEPTION_PTR
        caught_exception_ = nullptr;
    #endif // CAFFE2_USE_EXCEPTION_PTR
        error_timestamp_ = 0;
        */
    }
    
    #[inline] pub fn get_device_option(&self) -> &DeviceOption {
        
        todo!();
        /*
            return option_;
        */
    }
    
    #[inline] pub fn is_scheduled(&self) -> bool {
        
        todo!();
        /*
            return Query() == EventStatus::EVENT_SCHEDULED;
        */
    }
    
    #[inline] pub fn is_finished(&self) -> bool {
        
        todo!();
        /*
            auto status = Query();
        return status == EventStatus::EVENT_SUCCESS ||
            status == EventStatus::EVENT_FAILED;
        */
    }
    
    #[inline] pub fn set_finished(&mut self, err_msg: Option<&str>)  {

        todo!();
        /*
            typedef std::chrono::high_resolution_clock clock;
        error_timestamp_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               clock::now().time_since_epoch())
                               .count();

        CAFFE_ENFORCE(event_finished_setter_[type_]);
        return event_finished_setter_[type_](this, err_msg);
        */
    }
    
    #[inline] pub fn supports_callback(&self) -> bool {
        
        todo!();
        /*
            return event_callback_setter_[type_] != nullptr;
        */
    }
    
    #[inline] pub fn set_callback(&mut self, callback: EventCallbackFunction)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            event_callback_setter_[type_], "Event does not support callbacks");
        event_callback_setter_[type_](this, callback);
        */
    }

    /**
      | If parent op has succeeded, then we can
      | run any child op;
      | 
      | If parent op is in scheduled state, we
      | need to check that:
      | 
      | - child op supports async scheduling
      | 
      | - there's a way to setup synchronization
      | between async parent and child - both
      | child and parent should use the same
      | type of device, non-blocking synchronization
      | between different device types is not
      | supported
      | 
      | If parent op is in another state (initialized
      | or failed) then scheduling is not possible
      |
      */
    #[inline] pub fn can_schedule_child_event(
        &self, 
        child_event: &Event, 
        supports_async: bool) -> bool {

        todo!();
        /*
            return CanSchedule(type_, Query(), child_event.GetType(), supports_async);
        */
    }
    
    #[inline] pub fn can_schedule(
        parent_type:          i32,
        parent_status:        EventStatus,
        child_type:           i32,
        child_supports_async: bool) -> bool {

        todo!();
        /*
            if (parent_status == EventStatus::EVENT_SUCCESS) {
          return true;
        }
        if (parent_status == EventStatus::EVENT_SCHEDULED) {
          return (parent_type == child_type) && child_supports_async;
        }
        return false;
        */
    }
    
    #[inline] pub fn get_type(&self) -> i32 {
        
        todo!();
        /*
            return type_;
        */
    }
    
    #[inline] pub fn set_finished_with_exception(&mut self, err_msg: Option<&str>)  {

        todo!();
        /*
            #ifdef CAFFE2_USE_EXCEPTION_PTR
        if (!caught_exception_) {
          caught_exception_ = std::current_exception();
        }
        CAFFE_ENFORCE(caught_exception_, "No exception found");
    #else
        VLOG(1) << "No support for exceptions in Event";
    #endif // CAFFE2_USE_EXCEPTION_PTR
        if (err_msg) {
          SetFinished(err_msg);
        } else {
          SetFinished("Error happened during an operator run");
        }
        */
    }
    
    #[inline] pub fn has_exception(&self) -> bool {
        
        todo!();
        /*
            #ifdef CAFFE2_USE_EXCEPTION_PTR
        return (bool)caught_exception_;
    #else
        VLOG(1) << "No support for exceptions in Event";
        return false;
    #endif // CAFFE2_USE_EXCEPTION_PTR
        */
    }
    
    #[inline] pub fn error_timestamp(&self) -> i64 {
        
        todo!();
        /*
            return error_timestamp_;
        */
    }
    
    #[inline] pub fn rethrow_exception(&self)  {
        
        todo!();
        /*
            #ifdef CAFFE2_USE_EXCEPTION_PTR
        if (caught_exception_) {
          std::rethrow_exception(caught_exception_);
        }
    #else
        VLOG(1) << "No support for exceptions in Event";
    #endif // CAFFE2_USE_EXCEPTION_PTR
        */
    }
}

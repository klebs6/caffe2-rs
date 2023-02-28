crate::ix!();

pub const MaxDeviceTypes: usize = 11;//DeviceTypeProto::PROTO_COMPILE_TIME_MAX_DEVICE_TYPES;

pub enum EventStatus {
  EVENT_INITIALIZED = 0,
  EVENT_SCHEDULED   = 1,
  EVENT_SUCCESS     = 2,
  EVENT_FAILED      = 3,
}

/**
  | For the following functions, void*
  | shall be interpreted as the corresponding
  | context object corresponding to the
  | device type associated with the functions.
  |
  */

/**
  | Initializes event
  |
  */
pub type EventCreateFunction = fn(option: &DeviceOption, e: *mut Event);

/**
  | Called on event to signal that CPU part
  | of operation is finished,
  | 
  | Optionally accepts error message from
  | CPU part.
  | 
  | Should be called no more than once per
  | event
  |
  */
pub type EventRecordFunction = fn(*mut Event,*const c_void,*const u8);

/**
  | Waits and returns as soon as possible
  | in order schedule next operation, e.g.
  | for CUDA->CUDA waits only for CPU part
  | of CUDA op, for CUDA->CPU waits till
  | the CUDA op is fully completed.
  | 
  | Prepares context to synchronize device
  | part of operation.
  | 
  | Can be called concurrently from multiple
  | threads
  |
  */
pub type EventWaitFunction = fn(*const Event,*mut c_void);

/**
  | Waits till operation is fully finished,
  | can be called concurrently from multiple
  | threads
  |
  */
pub type EventFinishFunction = fn(*const Event);

/**
  | Queries current status of operation,
  | can be called concurrently from multiple
  | threads
  |
  */
pub type EventQueryFunction        = fn(*const Event);
pub type EventErrorMessageFunction = fn(*const Event);
pub type EventSetFinishedFunction  = fn(*const Event,*const u8);
pub type EventResetFunction        = fn(*mut Event);

/**
  | Sets callback that is called when event
  | is finished
  |
  */
pub type EventCallbackFunction    = fn() -> ();
pub type EventSetCallbackFunction = fn(*mut Event,EventCallbackFunction);

///------------------
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

lazy_static!{

    static ref event_creator:         [EventCreateFunction;       MaxDeviceTypes]                  = todo!();
    static ref event_recorder:        [EventRecordFunction;       MaxDeviceTypes]                  = todo!();
    static ref event_waiter:          [[EventWaitFunction;        MaxDeviceTypes]; MaxDeviceTypes] = todo!();
    static ref event_finisher:        [EventFinishFunction;       MaxDeviceTypes]                  = todo!();

    static ref event_querier:         [EventQueryFunction;        MaxDeviceTypes]                  = todo!();
    static ref event_err_msg_getter:  [EventErrorMessageFunction; MaxDeviceTypes]                  = todo!();
    static ref event_finished_setter: [EventSetFinishedFunction;  MaxDeviceTypes]                  = todo!();
    static ref event_resetter:        [EventResetFunction;        MaxDeviceTypes]                  = todo!();

    static ref event_callback_setter: [EventSetCallbackFunction;  MaxDeviceTypes]                  = todo!();
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

///----------------------
pub struct EventCreateFunctionRegisterer<const t: DeviceType> {
    
}

impl<const T: DeviceType> 
EventCreateFunctionRegisterer<T> {

    pub fn new(f: EventCreateFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_creator_[d] = f;
        */
    }
}

///-----------------------------------
pub struct EventRecordFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventRecordFunctionRegisterer<T> {

    pub fn new(f: EventRecordFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_recorder_[d] = f;
        */
    }
}

///----------------------------
pub struct EventWaitFunctionRegisterer<const waiter_type: DeviceType,const event_type: DeviceType> {
    
}

impl<const waiter_type: DeviceType,const event_type: DeviceType> 
EventWaitFunctionRegisterer<waiter_type,event_type> {

    pub fn new(f: EventWaitFunction) -> Self {
    
        todo!();
        /*
            auto waiter_index = TypeToProto(waiter_type);
        auto event_index = TypeToProto(event_type);
        Event::event_waiter_[waiter_index][event_index] = f;
        */
    }
}

///------------------------------
pub struct EventQueryFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventQueryFunctionRegisterer<T> {

    pub fn new(f: EventQueryFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_querier_[d] = f;
        */
    }
}

///---------------------------------
pub struct EventErrorMessageFunctionRegisterer<const t: DeviceType> {
    
}

impl<const T: DeviceType> EventErrorMessageFunctionRegisterer<T> {
    
    pub fn new(f: EventErrorMessageFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_err_msg_getter_[d] = f;
        */
    }
}

///---------------------------------
pub struct EventSetFinishedFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventSetFinishedFunctionRegisterer<T> {
    
    pub fn new(f: EventSetFinishedFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_finished_setter_[d] = f;
        */
    }
}

///-------------------------------
pub struct EventSetCallbackFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventSetCallbackFunctionRegisterer<T> {
    
    pub fn new(f: EventSetCallbackFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_callback_setter_[d] = f;
        */
    }
}

///--------------------------
pub struct EventFinishFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventFinishFunctionRegisterer<T> {
    
    pub fn new(f: EventFinishFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_finisher_[d] = f;
        */
    }
}

///--------------------------------------
pub struct EventResetFunctionRegisterer<const T: DeviceType> {
    
}

impl<const T: DeviceType> EventResetFunctionRegisterer<T> {
    
    pub fn new(f: EventResetFunction) -> Self {
    
        todo!();
        /*
            auto d = TypeToProto(t);
        Event::event_resetter_[d] = f;
        */
    }
}

pub const kNoError: &'static str = "No error";

#[inline] pub fn event_createcpu(option: &DeviceOption, event: *mut Event)  {
    
    todo!();
    /*
        event->event_ = std::make_shared<CPUEventWrapper>(option);
    */
}

#[inline] pub fn event_recordcpu(
    event:   *mut Event,
    unused:  *const c_void,
    err_msg: *const u8) {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      // Possible state changes:
      //  INITIALIZED -> SCHEDULED or SUCCESS/FAILED
      //  SCHEDULED -> SUCCESS/FAILED
      //  SUCCESS/FAILED - terminal, no further changes to status_/err_msg_

      CAFFE_ENFORCE(
          wrapper->status_ != EventStatus::EVENT_SCHEDULED,
          "Calling Record multiple times");

      // Event might be in SUCCESS/FAILED state in case an op has
      // finished async execution part first
      if (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
        if (!err_msg) {
          wrapper->status_ = EventStatus::EVENT_SCHEDULED;
        } else {
          wrapper->err_msg_ = err_msg;
          wrapper->status_ = EventStatus::EVENT_FAILED;
          wrapper->cv_completed_.notify_all();
        }
      }
    */
}


#[inline] pub fn event_finishcpu(event: *const Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);
      while (wrapper->status_ != EventStatus::EVENT_SUCCESS &&
             wrapper->status_ != EventStatus::EVENT_FAILED) {
        wrapper->cv_completed_.wait(lock);
      }
    */
}

#[inline] pub fn event_waitcpucpu(event: *const Event, context: *mut c_void)  {
    
    todo!();
    /*
        EventFinishCPU(event);
    */
}

#[inline] pub fn event_querycpu(event: *const Event) -> EventStatus {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      return static_cast<EventStatus>(wrapper->status_.load());
    */
}

#[inline] pub fn event_error_messagecpu<'a>(event: *const Event) -> &'a String {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      if (wrapper->status_ == EventStatus::EVENT_FAILED) {
        // Failed is a terminal state, not synchronizing,
        // err_msg_ should not be changed anymore
        return wrapper->err_msg_;
      } else {
        return kNoError;
      }
    */
}

#[inline] pub fn event_set_finishedcpu(event: *const Event, err_msg: *const u8)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      if (wrapper->status_ == EventStatus::EVENT_FAILED) {
        LOG(WARNING) << "SetFinished called on a finished event. "
                     << "Most likely caused by an external cancellation. "
                     << "old message: " << wrapper->err_msg_ << ", "
                     << "new message: " << err_msg;
        return;
      }

      CAFFE_ENFORCE(
          wrapper->status_ == EventStatus::EVENT_INITIALIZED ||
              wrapper->status_ == EventStatus::EVENT_SCHEDULED,
          "Calling SetFinished on finished event");

      if (!err_msg) {
        wrapper->status_ = EventStatus::EVENT_SUCCESS;
      } else {
        wrapper->err_msg_ = err_msg;
        wrapper->status_ = EventStatus::EVENT_FAILED;
      }

      for (auto& callback : wrapper->callbacks_) {
        callback();
      }

      wrapper->cv_completed_.notify_all();
    */
}

#[inline] pub fn event_set_callbackcpu(event: *mut Event, callback: EventCallbackFunction)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);

      wrapper->callbacks_.push_back(callback);
      if (wrapper->status_ == EventStatus::EVENT_SUCCESS ||
          wrapper->status_ == EventStatus::EVENT_FAILED) {
        callback();
      }
    */
}

#[inline] pub fn event_resetcpu(event: *mut Event)  {
    
    todo!();
    /*
        auto* wrapper = static_cast<CPUEventWrapper*>(event->event_.get());
      std::unique_lock<std::mutex> lock(wrapper->mutex_);
      wrapper->status_ = EventStatus::EVENT_INITIALIZED;
      wrapper->err_msg_ = "";
      wrapper->callbacks_.clear();
    */
}

register_event_create_function![CPU,           EventCreateCPU];
register_event_record_function![CPU,           EventRecordCPU];
register_event_wait_function![CPU,             CPU, EventWaitCPUCPU];
register_event_finish_function![CPU,           EventFinishCPU];
register_event_query_function![CPU,            EventQueryCPU];
register_event_error_message_function![CPU,    EventErrorMessageCPU];
register_event_set_finished_function![CPU,     EventSetFinishedCPU];
register_event_reset_function![CPU,            EventResetCPU];
register_event_set_callback_function![CPU,     EventSetCallbackCPU];

crate::ix!();

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


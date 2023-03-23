/*!
  | For the following functions, void*
  | shall be interpreted as the corresponding
  | context object corresponding to the
  | device type associated with the functions.
  |
  */

crate::ix!();

pub const MaxDeviceTypes: usize = 11;//DeviceTypeProto::PROTO_COMPILE_TIME_MAX_DEVICE_TYPES;

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


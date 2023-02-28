crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/C10Event.h]

/**
  | A backend-generic movable, not copyable,
  | not thread-safe event.
  | 
  | The design of this event follows that
  | of Cuda and HIP events. These events
  | are recorded and waited on by streams
  | and can be rerecorded to, each rerecording
  | essentially creating a new version
  | of the event.
  | 
  | For example, if (in CPU time), stream
  | X is asked to record E, stream Y waits
  | on E, and stream X is asked to record E
  | again, then Y will wait for X to finish
  | the first call to record and not the second,
  | because it's waiting on the first version
  | of event E, not the second.
  | 
  | Querying an event only returns the status
  | of its most recent version.
  | 
  | Backend-generic events are implemented
  | by this class and
  | 
  | InlineEvent. In addition to these events
  | there are also some backend-specific
  | events, like ATen's CudaEvent. Each
  | of these classes has its own use.
  | 
  | InlineEvent<...> or a backend-specific
  | event should be preferred when the backend
  | is known at compile time and known to
  | be compiled. Backend-specific events
  | may have additional functionality.
  | 
  | This C10Event should be used if a particular
  | backend may not be available, or the
  | backend required is not known at compile
  | time.
  | 
  | These generic events are built on top
  | of DeviceGuardImpls, analogous to
  | DeviceGuard and InlineDeviceGuard.
  | The name "DeviceGuardImpls," is no
  | longer entirely accurate, as these
  | classes implement the backend-specific
  | logic for a generic backend interface.
  | 
  | See DeviceGuardImplInterface.h for
  | a list of all supported flags.
  |
  */
pub struct C10Event {
    impl_: InlineEvent<VirtualGuardImpl>,
}

impl C10Event {
    
    pub fn new(
        device_type: DeviceType,
        flag:        Option<EventFlag>) -> Self {

        let flag: EventFlag = flag.unwrap_or(EventFlag::PYTORCH_DEFAULT);

        todo!();
        /*
            : impl_{_device_type, _flag}
        */
    }

    /**
      | Move constructor and move assignment
      | operator
      |
      */
    pub fn new_from_event(other: C10Event) -> Self {
    
        todo!();
        /*


            : impl_{move(other.impl_)}
        */
    }
    
    pub fn assign_from(&mut self, other: C10Event) -> &mut C10Event {
        
        todo!();
        /*
            impl_.swap(move(other.impl_));
        return *this;
        */
    }

    // Getters
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return Device(device_type(), device_index());
        */
    }
    
    
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return impl_.device_type();
        */
    }
    
    pub fn device_index(&self) -> DeviceIndex {
        
        todo!();
        /*
            return impl_.device_index();
        */
    }
    
    pub fn flag(&self) -> EventFlag {
        
        todo!();
        /*
            return impl_.flag();
        */
    }
    
    pub fn was_marked_for_recording(&self) -> bool {
        
        todo!();
        /*
            return impl_.was_marked_for_recording();
        */
    }

    /**
      | Calls record() if and only if record()
      | has never been called for this event.
      | 
      | -----------
      | @note
      | 
      | because C10Event is not thread-safe recordOnce()
      | may call record() multiple times if
      | called from multiple threads.
      |
      */
    pub fn record_once(&mut self, stream: &Stream)  {
        
        todo!();
        /*
            impl_.recordOnce(stream);
        */
    }

    /**
      | Increments the event's version and
      | enqueues a job with this version in the
      | stream's work queue.
      | 
      | When the stream process that job it nofifies
      | all streams waiting on / blocked by that
      | version of the event to continue and
      | marks that version as recorded.
      |
      */
    pub fn record(&mut self, stream: &Stream)  {
        
        todo!();
        /*
            impl_.record(stream);
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
    pub fn block(&self, stream: &Stream)  {
        
        todo!();
        /*
            impl_.block(stream);
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
    pub fn query(&self) -> bool {
        
        todo!();
        /*
            return impl_.query();
        */
    }
}

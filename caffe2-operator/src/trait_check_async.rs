crate::ix!();

pub trait CheckAsync {

    #[inline] fn has_async_part_fallback(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
    
    #[inline] fn supports_async_scheduling_fallback(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }

    /**
      | Returns whether operator has async on
      | device part.
      |
      | CUDA operators by default have async
      | parts, CPU operators by default don't have
      | async parts and are finished after
      | RunOnDevice call.
      |
      | Events of operators that don't have async
      | parts are automatically set to finished
      | state by RunAsync.
      |
      | Defaulting to the value from context (true
      | for CUDA, false for CPU).
      |
      | Override in case of async CPU operators
      |
      | Async CPU operators are expected to catch
      | all exceptions in async parts and set
      | Event to finished/failed state with
      | Event::SetFinished or
      | SetFinishedWithException call.
      */
    #[inline] fn has_async_part(&self) -> bool {

        todo!();
        /*
            return context_.HasAsyncPartDefault();
        */
    }

    /**
      | Returns whether operator's RunOnDevice
      | schedules async on device part and can be
      | run without waiting for parent operator's
      | async part to be finished on the same
      | device.
      |
      | Note: when true, RunOnDevice must not
      | access the content of the input blobs as
      | they might not be computed yet
      |
      | Note: when true, operator's device needs
      | to support async scheduling:
      |
      |  - supports concept of streams: async ops
      |    scheduled on the same stream are
      |    guaranteed to be executed in the same
      |    order they were scheduled
      |
      |  - provides non-blocking cross
      |    device/cross stream synchronization
      |    primitives
      |
      | By default, assuming an op with an async
      | part can be scheduled asynchronously if
      | device supports async scheduling
      */
    #[inline] fn supports_async_scheduling(&self) -> bool {
        
        todo!();
        /*
            return HasAsyncPart() && context_.SupportsAsyncScheduling();
        */
    }
}



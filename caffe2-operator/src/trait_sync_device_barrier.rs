crate::ix!();

pub trait SyncDeviceBarrierForObservers {

    #[inline] fn sync_device_barrier_for_observers_fallback(&mut self)  {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }

    /// Internal API invoked by observers. Normal callers shouldn't invoke it.
    #[inline] fn sync_device_barrier_for_observers(&mut self)  {
        
        todo!();
        /*
            context_.FinishDeviceComputation();
        */
    }
}

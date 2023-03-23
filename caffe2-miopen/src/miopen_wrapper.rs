crate::ix!();

/**
 | MIOPENWrapper is a class that wraps the miopen
 | handles and miopen workspaces.
 |
 | The wrapper ensures that for each thread and
 | each gpu, there is one identical miopen handle,
 | which is also associated with the thread-local
 | per-device hip stream. The wrapper also hosts
 | the device-specific miopen workspace (scratch
 | space for some miopen functions).
 |
 */
pub struct MIOPENWrapper
{
    /**
      Pointer to an external cuda context that
      the miopen wrapper will use.
      */
    context:  *mut hipCtx_t,
}

pub const CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES: usize = 4;

impl MIOPENWrapper {

    /**
     | Creates a miopen wrapper associated with
     | a HIPContext object. Note that the
     | HIPContext object should outlive the
     | MIOPENWrapper.
     */
    pub fn new(context: *mut hipCtx_t) -> Self {
    
        todo!();
        /*
            : context_(context)
        */
    }
    
    /**
     | Returns the inline miopen handle that
     | executes on the current thread's
     | hip_stream.
     */
    #[inline] pub fn inline_miopen_handle(&mut self) -> miopenHandle_t {
        
        todo!();
        /*
            return context_->miopen_handle();
        */
    }
    
    /**
      | Executes the closure F on the MIOpenState
      | associated with state_idx
      |
      */
    #[inline] pub fn with_miopen_state<F>(&mut self, state_idx: usize, f: F)  {
    
        todo!();
        /*
            CAFFE_ENFORCE(state_idx < CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES, "Invalid state_idx");
            auto& sync_state = miopen_states()[context_->device_id()][state_idx];

            HIPGuard dg(context_->device_id());

            // We need to serialize execution on the MIOpenState as we can't
            // allow multiple threads to race through the cudaEventRecord
            // calls (so a worker thread might wait on another worker thread's
            // execution)
            std::lock_guard<std::mutex> g(sync_state.mutex);
            if(!sync_state.state.get())
            {
              sync_state.state.reset(new MIOpenState(context_->device_id()));
            }
            CHECK_NOTNULL(sync_state.state.get())->execute(context_->hip_stream(), f);
        */
    }
    
    #[inline] pub fn miopen_states<'a>() -> &'a mut PerGPUMIOPENStates {
        
        todo!();
        /*
        
        */
    }
}

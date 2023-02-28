crate::ix!();

use crate::*;

/**
 | MIOpenWorkspace is a wrapper around a raw cuda
 | pointer that holds the miopen scratch
 | space. This struct is meant to be only used in
 | MIOPENWrapper to provide a program-wide scratch
 | space for MIOPEN. The reason behind it is that
 | miopen function calls are usually very
 | efficient, hence one probably does not want to
 | run multiple miopen calls at the same time. As
 | a result, one should not need more than one
 | miopen workspace per device.
 */
pub struct MIOpenWorkspace
{
    data:    DataPtr,
    nbytes:  usize, // default = 0
}

impl MIOpenWorkspace {
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            data_.clear();
          nbytes_ = 0;
        */
    }
    
    #[inline] pub fn get(&mut self, nbytes: usize)  {
        
        todo!();
        /*
            if(nbytes_ < nbytes)
            {
                reset();
                data_ = HIPContext::New(nbytes);
                nbytes_               = nbytes;
            }
            CAFFE_ENFORCE_GE(nbytes_, nbytes);
            return data_.get();
        */
    }
}

/**
  | MIOpenState is the owner of the MIOpenWorkspace,
  | and serializes all executions of operations that
  | use the state onto it's own stream (so multiple
  | Net workers can reuse the same workspace from
  | different threads and HIP streams).
  */
pub struct MIOpenState {
    
    miopen_handle:  miopenHandle_t, // default = nullptr
    before:         hipEvent_t,     // default = nullptr
    after:          hipEvent_t,     // default = nullptr
    stream:         hipStream_t,    // default = nullptr
    workspace:      MIOpenWorkspace,
    gpu_id:         usize, // default = 0
}

impl MIOpenState {

    pub fn new(gpu_id: usize) -> Self {
    
        todo!();
        /*
            : gpu_id_(gpu_id)

            HIPGuard g(gpu_id_);
            MIOPEN_ENFORCE(miopenCreate(&miopen_handle_));
            HIP_ENFORCE(hipEventCreate(&before_));
            HIP_ENFORCE(hipEventCreate(&after_));
            HIP_ENFORCE(hipStreamCreate(&stream_));
            MIOPEN_ENFORCE(miopenSetStream(miopen_handle_, stream_));
        */
    }
}

impl Drop for MIOpenState {

    fn drop(&mut self) {
        todo!();
        /* 
            HIPGuard g(gpu_id_);
            MIOPEN_CHECK(miopenDestroy(miopen_handle_));
            HIP_CHECK(hipStreamDestroy(stream_));
            HIP_CHECK(hipEventDestroy(after_));
            HIP_CHECK(hipEventDestroy(before_));
         */
    }
}

impl MIOpenState {
    
    #[inline] pub fn miopen_handle(&mut self) -> &mut miopenHandle_t {
        
        todo!();
        /*
            return miopen_handle_;
        */
    }
    
    #[inline] pub fn workspace(&mut self) -> &mut MIOpenWorkspace {
        
        todo!();
        /*
            return workspace_;
        */
    }
    
    #[inline] pub fn execute<F>(&mut self, stream: hipStream_t, f: F)  {
    
        todo!();
        /*
            HIP_ENFORCE(hipEventRecord(before_, stream));
            HIP_ENFORCE(hipStreamWaitEvent(stream_, before_, 0));
            f(this);
            HIP_ENFORCE(hipEventRecord(after_, stream_));
            HIP_ENFORCE(hipStreamWaitEvent(stream, after_, 0));
        */
    }
}

///------------------------
pub struct SyncedMIOPENState
{
    mutex:  parking_lot::RawMutex,
    state:  Box<MIOpenState>,
}

pub type PerGPUMIOPENStates = [[SyncedMIOPENState; CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES]; COMPILE_TIME_MAX_GPUS];

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

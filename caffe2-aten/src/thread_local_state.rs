crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ThreadLocalState.h]

/**
  | Thread local state contains values that are
  | preserved across thread boundaries
  | (e.g. launch/JIT fork, autograd).
  |
  | Note parallel_for doesn't preserve TLS across
  | thread boundaries.
  |
  */
pub struct ThreadLocalState {
 
    dispatch_key: LocalDispatchKeySet,

    /**
      | ThreadLocalDebugInfo does not change
      | after being created with DebugInfoGuard
      |
      */
    debug_info:   Arc<ThreadLocalDebugInfo>,

    /**
      | RecordFunction TLS
      |
      */
    rf_tls:       RecordFunctionTLS,

    #[cfg(all(not(CAFFE2_IS_XPLAT_BUILD),not(C10_MOBILE)))]
    keep_grad_mode:    bool, // default = true

    #[cfg(all(not(CAFFE2_IS_XPLAT_BUILD),not(C10_MOBILE)))]
    grad_mode_enabled: bool,

    /**
      | TLS for InferenceMode
      |
      */
    inference_mode_enabled:      bool,

    /**
      | Whether pre-sampling RecordFunction
      | optimization was enabled
      |
      */
    bumped_record_all_functions: bool, // default = false
}

impl ThreadLocalState {

    /**
      | Saves the thread local variables' values and
      | returns them as a ThreadLocalState
      |
      | keep_grad_mode - whether grad mode has to be
      | preserved
      |
      |  (e.g. not preserved when passing from
      |   forward pass into the autograd engine,
      |   autograd engine takes care of grad mode)
      */
    pub fn new(keep_grad_mode: bool) -> Self {

        let keep_grad_mode: bool = keep_grad_mode.unwrap_or(true);

        todo!();
        /*
        
        */
    }

    /**
      | Sets thread local variables in the current
      | thread, according to the thread boundary
      | specified
      |
      */
    pub fn set_thread_local_state(state: &ThreadLocalState)  {
        
        todo!();
        /*
        
        */
    }
}

/**
  | Guard to set and reset the thread local
  | state
  |
  */
pub struct ThreadLocalStateGuard {

    prev_state:                  ThreadLocalState,

    /**
      | Whether pre-sampling RecordFunction
      | optimization was enabled
      |
      */
    bumped_record_all_functions: bool, // default = false
}

impl Drop for ThreadLocalStateGuard {

    fn drop(&mut self) {
        todo!();
        /*
            // restore previously set variables
        ThreadLocalState::setThreadLocalState(prev_state_);
        if (bumped_record_all_functions_) {
          releaseRecordAllFunctions();
        }
        */
    }
}

impl ThreadLocalStateGuard {
    
    pub fn new(state: &ThreadLocalState) -> Self {
    
        todo!();
        /*
        : prev_state(ThreadLocalState()),
        : bumped_record_all_functions(state.bumped_record_all_functions_),

            // Special handling of RecordFunction pre-sampling optimization:
        // pre-samping is enabled (bumped) when there're non-sampled
        // (or high-frequency) global or TLS callbacks.
        //
        // ThreadLocalStateGuard simply resets RecordFunction's TLS and
        // hence its thread local callbacks.
        //
        // Checking if the pre-sampling was enabled and preserving it in the
        // async task by calling bumpRecordAllFunctions() and the corresponding
        // releaseRecordAllFunctions()
        if (bumped_record_all_functions_) {
          bumpRecordAllFunctions();
        }
        // set the given state across the thread boundary
        ThreadLocalState::setThreadLocalState(state);
        */
    }
}

pub fn wrap_propagate_tls_state<T>(callback: T) -> Auto {

    todo!();
        /*
            return [tls_state = ThreadLocalState(),
              callback = move(callback)](auto&&... args) {
        ThreadLocalStateGuard g(tls_state);
        // Propagate value returned by callback().
        return callback(forward<decltype(args)>(args)...);
      };
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ThreadLocalState.cpp]

impl ThreadLocalState {
    
    pub fn new(keep_grad_mode: bool) -> Self {
    
        todo!();
        /*


            : dispatch_key_(tls_local_dispatch_key_set()),
          debug_info_(ThreadLocalDebugInfo::current()),
          inference_mode_enabled_(InferenceMode::is_enabled()) 
      rf_tls_ = get_record_function_tls_();

    #if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      keep_grad_mode_ = keep_grad_mode;
      if (keep_grad_mode_) {
        grad_mode_enabled_ = GradMode::is_enabled();
      }
    #endif
      bumped_record_all_functions_ = checkRecordAllFunctions();
        */
    }
    
    pub fn set_thread_local_state(&mut self, state: &ThreadLocalState)  {
        
        todo!();
        /*
            #if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      if (state.keep_grad_mode_) {
        GradMode::set_enabled(state.grad_mode_enabled_);
      }
    #endif

      set_record_function_tls_(state.rf_tls_);

      ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

      _force_tls_local_dispatch_key_set(state.dispatch_key_);

      InferenceMode::_set_enabled(state.inference_mode_enabled_);
        */
    }
}

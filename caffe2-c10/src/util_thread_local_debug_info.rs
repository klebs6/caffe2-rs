crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/ThreadLocalDebugInfo.h]

#[repr(u8)]
pub enum DebugInfoKind {
    PRODUCER_INFO = 0,
    MOBILE_RUNTIME_INFO,
    PROFILER_STATE,

    /// for inference usage
    INFERENCE_CONTEXT, 
    PARAM_COMMS_INFO,

    /// used only in tests
    TEST_INFO, 

    /// used only in tests
    TEST_INFO_2, 
}

pub trait DebugInfoBaseInterface { }

/**
  | Thread local debug information is propagated
  | across the forward (including async fork tasks)
  | and backward passes and is supposed to be
  | utilized by the user's code to pass extra
  | information from the higher layers (e.g. model
  | id) down to the lower levels (e.g. to the
  | operator observers used for debugging, logging,
  | profiling, etc)
  */
pub struct ThreadLocalDebugInfo {
    info:        Arc<dyn DebugInfoBaseInterface>,
    kind:        DebugInfoKind,
    parent_info: Arc<ThreadLocalDebugInfo>,
}

impl Default for ThreadLocalDebugInfo {

    fn default() -> Self {
        todo!();
    }
}

impl ThreadLocalDebugInfo {
    
    pub fn get(&mut self, kind: DebugInfoKind) -> *mut dyn DebugInfoBaseInterface {
        
        todo!();
        /*
            ThreadLocalDebugInfo* cur = debug_info.get();
      while (cur) {
        if (cur->kind_ == kind) {
          return cur->info_.get();
        }
        cur = cur->parent_info_.get();
      }
      return nullptr;
        */
    }
    
    /// get current ThreadLocalDebugInfo
    ///
    pub fn current(&mut self) -> Arc<ThreadLocalDebugInfo> {
        
        todo!();
        /*
            return debug_info;
        */
    }
    
    /// Internal, use
    /// DebugInfoGuard/ThreadLocalStateGuard
    ///
    pub fn force_current_debug_info(&mut self, info: &Arc<ThreadLocalDebugInfo>)  {
        
        todo!();
        /*
            debug_info = info;
        */
    }
    
    /// Push debug info struct of a given kind
    ///
    pub fn push(&mut self, 
        kind: DebugInfoKind,
        info: Arc<dyn DebugInfoBaseInterface>)  {
        
        todo!();
        /*
            auto prev_info = debug_info;
      debug_info = make_shared<ThreadLocalDebugInfo>();
      debug_info->parent_info_ = prev_info;
      debug_info->kind_ = kind;
      debug_info->info_ = info;
        */
    }
    
    /**
      | Pop debug info, throws in case the last
      | pushed debug info is not of a given kind
      |
      */
    pub fn pop(&mut self, kind: DebugInfoKind) -> Arc<dyn DebugInfoBaseInterface> {
        
        todo!();
        /*
            TORCH_CHECK(
          debug_info && debug_info->kind_ == kind,
          "Expected debug info of type ",
          (size_t)kind);
      auto res = debug_info;
      debug_info = debug_info->parent_info_;
      return res->info_;
        */
    }
    
    /**
      | Peek debug info, throws in case the last
      | pushed debug info is not of the given
      | kind
      |
      */
    pub fn peek(&mut self, kind: DebugInfoKind) -> Arc<dyn DebugInfoBaseInterface> {
        
        todo!();
        /*
            TORCH_CHECK(
          debug_info && debug_info->kind_ == kind,
          "Expected debug info of type ",
          (size_t)kind);
      return debug_info->info_;
        */
    }
}


/**
  | DebugInfoGuard is used to set debug
  | information, ThreadLocalDebugInfo is
  | semantically immutable, the values are set
  | through the scope-based guard object.
  |
  | Nested DebugInfoGuard adds/overrides existing
  | values in the scope, restoring the original
  | values after exiting the scope.
  |
  | Users can access the values through the
  | ThreadLocalDebugInfo::get() call;
  |
  */
pub struct DebugInfoGuard {
    active:    bool, // default = false
    prev_info: Arc<ThreadLocalDebugInfo>, // default = nullptr
}

impl Drop for DebugInfoGuard {

    fn drop(&mut self) {

        todo!();

        /*
        if (active_) {
            debug_info = prev_info_;
        }
       */
    }
}

impl DebugInfoGuard {
    
    pub fn new(
        kind: DebugInfoKind,
        info: Arc<dyn DebugInfoBaseInterface>) -> Self {
    
        todo!();
        /*
          if (!info) {
            return;
          }

          prev_info_ = debug_info;
          ThreadLocalDebugInfo::_push(kind, info);
          active_ = true;
        */
    }
    
    /**
      | Used only for setting a debug info after
      | crossing the thread boundary; in this
      | case we assume that thread pool's thread
      | does not have an active debug info
      |
      */
    pub fn new_with_threadlocal(info: Arc<ThreadLocalDebugInfo>) -> Self {
    
        todo!();
        /*
          if (!info) {
            return;
          }

          prev_info_ = debug_info;
          debug_info = info;
          active_ = true;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/ThreadLocalDebugInfo.cpp]

thread_local! {
    pub static TLS_DEBUG_INFO: Arc<ThreadLocalDebugInfo> = Default::default();
}

macro_rules! debug_info {
    () => {
        /*
                (tls_debug_info.get())
        */
    }
}

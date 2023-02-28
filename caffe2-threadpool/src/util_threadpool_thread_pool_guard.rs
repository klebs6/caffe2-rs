crate::ix!();

/**
  | A RAII, thread local (!) guard that enables
  | or disables grad mode upon construction,
  | and sets it back to the original value
  | upon destruction.
  |
  */
pub struct _NoPThreadPoolGuard {
    prev_mode: bool,
}

impl _NoPThreadPoolGuard {

    #[inline] pub fn is_enabled() -> bool {
        todo!();
    }

    #[inline] pub fn set_enabled(enabled: bool) {
        todo!();
    }
}

impl Default for _NoPThreadPoolGuard {
    
    fn default() -> Self {
        todo!();
        /*
            : prev_mode_(_NoPThreadPoolGuard::is_enabled()) 

          _NoPThreadPoolGuard::set_enabled(true)
        */
    }
}

impl Drop for _NoPThreadPoolGuard {
    fn drop(&mut self) {
        todo!();
        /* 
          _NoPThreadPoolGuard::set_enabled(prev_mode_);
       */
    }
}


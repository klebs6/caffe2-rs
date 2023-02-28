crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/GradMode.h]

pub struct GradMode {

}

/**
  | A RAII, thread local (!) guard that enables
  | or disables grad mode upon construction,
  | and sets it back to the original value
  | upon destruction.
  |
  */
pub struct AutoGradMode {
    prev_mode: bool,
}

impl Drop for AutoGradMode {

    fn drop(&mut self) {
        todo!();
        /*
            GradMode::set_enabled(prev_mode);
        */
    }
}

impl AutoGradMode {

    pub fn new(enabled: bool) -> Self {
    
        todo!();
        /*
           : prev_mode(GradMode::is_enabled()) 

        GradMode::set_enabled(enabled);
        */
    }
}

/**
  | A RAII, thread local (!) guard that stops
  | future operations from building gradients.
  |
  */
pub struct NoGradGuard {
    base: AutoGradMode,
}

impl Default for NoGradGuard {
    
    fn default() -> Self {
        todo!();
        /*


            : AutoGradMode(/*enabled=*/false)
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/GradMode.cpp]

lazy_static!{
    /*
    thread_local bool GradMode_enabled = true;
    */
}

impl GradMode {
    
    pub fn is_enabled(&mut self) -> bool {
        
        todo!();
        /*
            return GradMode_enabled;
        */
    }
    
    pub fn set_enabled(&mut self, enabled: bool)  {
        
        todo!();
        /*
            GradMode_enabled = enabled;
        */
    }
}

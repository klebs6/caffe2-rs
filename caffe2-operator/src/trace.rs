crate::ix!();

macro_rules! trace_name_concatenate {
    ($s1:ident, $s2:ident) => {
        todo!();
        /*
                s1##s2
        */
    }
}

macro_rules! trace_anonymous_name {
    ($str:ident) => {
        todo!();
        /*
                TRACE_NAME_CONCATENATE(str, __LINE__)
        */
    }
}

macro_rules! trace_event_init {
    ($($arg:ident),*) => {
        todo!();
        /*
        
          TRACE_ANONYMOUS_NAME(trace_guard).init(tracer_.get());      
          TRACE_ANONYMOUS_NAME(trace_guard).addArgument(__VA_ARGS__); 
          TRACE_ANONYMOUS_NAME(trace_guard).recordEventStart();
        */
    }
}

/**
  | Supposed to be used only once per scope
  | in AsyncNetBase-derived nets
  |
  */
macro_rules! trace_event {
    (, $($arg:ident),*) => {
        todo!();
        /*
        
          tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); 
          if (tracer_ && tracer_->isEnabled()) {                  
            TRACE_EVENT_INIT(__VA_ARGS__)                         
          }
        */
    }
}

macro_rules! trace_event_if {
    ($cond:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); 
          if (tracer_ && tracer_->isEnabled() && (cond)) {        
            TRACE_EVENT_INIT(__VA_ARGS__)                         
          }
        */
    }
}

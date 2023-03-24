crate::ix!();

pub struct TracerGuard {
    enabled:  bool, // default = false
    event:    TracerEvent,
    tracer:   *mut Tracer,
}

thread_local!{
    pub static current_tracer_guard: *mut TracerGuard = todo!();
}

impl Drop for TracerGuard {

    fn drop(&mut self) {
        todo!();
        /* 
      if (enabled_) {
        event_.is_beginning_ = false;
        event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
        tracer_->recordEvent(event_);
        if (current_tracer_guard == this) {
          current_tracer_guard = nullptr;
        }
      }
 */
    }
}

impl TracerGuard {

    #[inline] pub fn init_from_tracer(&mut self, tracer: *mut Tracer)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn add_argument_with_args<T, Args>(&mut self, 
        field: TracingField,
        value: &T,
        args:  &Args)  {

        todo!();
        /*
            addArgument(field, value);
        addArgument(args...);
        */
    }
    
    #[inline] pub fn init(&mut self, tracer: *mut Tracer)  {
        
        todo!();
        /*
            enabled_ = tracer && tracer->isEnabled();
      if (enabled_) {
        current_tracer_guard = this;
      }
      tracer_ = tracer;
        */
    }
    
    #[inline] pub fn add_argument_with_value(
        &mut self, 
        field: TracingField, 
        value: *const u8)  
    {
        todo!();
        /*
            switch (field) {
        case TRACE_NAME: {
          event_.name_ = value;
          break;
        }
        case TRACE_CATEGORY: {
          event_.category_ = value;
          break;
        }
        default: {
          CAFFE_THROW("Unexpected tracing string field ", field);
        }
      }
        */
    }
    
    #[inline] pub fn add_argument(&mut self, field: TracingField, value: i32)  {
        
        todo!();
        /*
            switch (field) {
        case TRACE_OP: {
          event_.op_id_ = value;
          break;
        }
        case TRACE_TASK: {
          event_.task_id_ = value;
          break;
        }
        case TRACE_STREAM: {
          event_.stream_id_ = value;
          break;
        }
        case TRACE_THREAD: {
          event_.thread_label_ = value;
          break;
        }
        case TRACE_ITER: {
          event_.iter_ = value;
          break;
        }
        default: {
          CAFFE_THROW("Unexpected tracing int field ", field);
        }
      }
        */
    }
    
    #[inline] pub fn record_event_start(&mut self)  {
        
        todo!();
        /*
            if (enabled_) {
        if (event_.thread_label_ < 0) {
          event_.tid_ = std::this_thread::get_id();
        }
        event_.is_beginning_ = true;
        event_.timestamp_ = (long)caffe2::round(tracer_->timer_.MicroSeconds());
        tracer_->recordEvent(event_);
      }
        */
    }
    
    #[inline] pub fn disable(&mut self)  {
        
        todo!();
        /*
            enabled_ = false;
        */
    }
    
    #[inline] pub fn get_current_tracer_guard(&mut self) -> *mut TracerGuard {
        
        todo!();
        /*
            return current_tracer_guard;
        */
    }
}

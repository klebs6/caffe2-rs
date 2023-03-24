crate::ix!();

pub struct TimeObserver {
    counter:  TimeCounter,
    observer: OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver>,
}

impl TimeObserver {
    
    pub fn new(subject: *mut NetBase) -> Self {
    
        todo!();
        /*
            : OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver>(
                subject,
                this)
        */
    }
    
    #[inline] pub fn average_time_children(&self) -> f32 {
        
        todo!();
        /*
            float sum = 0.0f;
        for (const auto* observer : operator_observers_) {
          sum += observer->average_time();
        }
        return sum / subject_->GetOperators().size();
        */
    }
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            start_time_ = timer_.MilliSeconds();
      ++iterations_;
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
            double current_run = timer_.MilliSeconds() - start_time_;
      total_time_ += current_run;
      VLOG(1) << "This net iteration took " << current_run << " ms to complete.\n";
        */
    }
}

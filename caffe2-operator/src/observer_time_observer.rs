crate::ix!();


pub struct TimeCounter {
    timer:      Timer,
    start_time: f32, // default = 0.0
    total_time: f32, // default = 0.0
    iterations: i32, // default = 0
}

impl TimeCounter {
    
    #[inline] pub fn average_time(&self) -> f32 {
        
        todo!();
        /*
            return total_time_ / iterations_;
        */
    }
}

///-----------------------------------------
pub struct TimeOperatorObserver {
    counter:  TimeCounter,
    observer: ObserverBase<OperatorStorage>,
}

impl TimeOperatorObserver {
    
    pub fn new(subject: *mut OperatorStorage, unused: *mut TimeObserver) -> Self {
    
        todo!();
        /*
            : ObserverBase<OperatorStorage>(subject)
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
      VLOG(1) << "This operator iteration took " << current_run
              << " ms to complete.\n";
        */
    }
    
    #[inline] pub fn rnn_copy(
        &self, 
        subject:   *mut OperatorStorage, 
        rnn_order: i32) -> Box<ObserverBase<OperatorStorage>> 
    {
        todo!();
        /*
            return std::unique_ptr<ObserverBase<OperatorStorage>>(
          new TimeOperatorObserver(subject, nullptr));
        */
    }
}

///-------------------------------------
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

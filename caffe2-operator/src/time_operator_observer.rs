crate::ix!();

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


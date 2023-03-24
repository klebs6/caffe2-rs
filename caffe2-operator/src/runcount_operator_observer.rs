crate::ix!();

pub struct RunCountOperatorObserver {
    base:         ObserverBase<OperatorStorage>,
    net_observer: *mut RunCountNetObserver,
}

pub struct RunCountNetObserver {
    base: OperatorAttachingNetObserver<RunCountOperatorObserver,RunCountNetObserver>,
    cnt:  Atomic<i32>,
}

impl RunCountNetObserver {

    pub fn new(subject: *mut NetBase) -> Self {
    
        todo!();
        /*
            : OperatorAttachingNetObserver<
                RunCountOperatorObserver,
                RunCountNetObserver>(subject_, this),
            cnt_(0)
        */
    }
    
    #[inline] pub fn debug_info(&mut self) -> String {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}

///---------------------------------------

impl RunCountOperatorObserver {
    
    pub fn new(op: *mut OperatorStorage, net_observer: *mut RunCountNetObserver) -> Self {
    
        todo!();
        /*
            : ObserverBase<OperatorStorage>(op), netObserver_(netObserver) 

      CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
        */
    }
    
    #[inline] pub fn debug_info(&mut self) -> String {
        
        todo!();
        /*
            #ifdef C10_ANDROID
      // workaround
      int foo = cnt_;
      return "This operator runs " + c10::to_string(foo) + " times.";
    #else
      return "This operator runs " + c10::to_string(cnt_) + " times.";
    #endif
        */
    }
    
    #[inline] pub fn stop(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn start(&mut self)  {
        
        todo!();
        /*
            ++netObserver_->cnt_;
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
          new RunCountOperatorObserver(subject, netObserver_));
        */
    }
}

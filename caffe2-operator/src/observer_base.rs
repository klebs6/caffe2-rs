crate::ix!();

/**
  | Use this to implement a Observer using
  | the Observer Pattern template.
  |
  */
pub struct ObserverBase<T> {
    subject: *mut T,
}

impl<T> ObserverBase<T> {
    
    pub fn new(subject: *mut T) -> Self {
    
        todo!();
        /*
            : subject_(subject)
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
    
    #[inline] pub fn debug_info(&mut self) -> String {
        
        todo!();
        /*
            return "Not implemented.";
        */
    }
    
    #[inline] pub fn subject(&self) -> *mut T {
        
        todo!();
        /*
            return subject_;
        */
    }
    
    #[inline] pub fn rnn_copy(&self, subject: *mut T, rnn_order: i32) -> Box<ObserverBase<T>> {
        
        todo!();
        /*
            return nullptr;
        */
    }
}

crate::ix!();

/**
  | A macro that wraps around a miopen statement
  | so we can check if the miopen execution
  | finishes or not.
  |
  */
#[macro_export] macro_rules! miopen_enforce {
    ($condition:ident) => {
        todo!();
        /*
        
            do                                                                      
            {                                                                       
                miopenStatus_t status = condition;                                  
                CAFFE_ENFORCE_EQ(status,                                            
                                 miopenStatusSuccess,                               
                                 ", Error at: ",                                    
                                 __FILE__,                                          
                                 ":",                                               
                                 __LINE__,                                          
                                 ": ",                                              
                                 ::caffe2::internal::miopenGetErrorString(status)); 
            } while(0)
        */
    }
}


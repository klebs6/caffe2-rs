crate::ix!();

/**
  | A macro that wraps around a cudnn statement
  | so we can check if the cudnn execution
  | finishes or not.
  |
  */
#[macro_export] macro_rules! cudnn_enforce {
    ($condition:expr) => {
        todo!();
        /*
        cudnnStatus_t status = condition;                     
        CAFFE_ENFORCE_EQ(                                     
            status,                                           
            CUDNN_STATUS_SUCCESS,                             
            ", Error at: ",                                   
            __FILE__,                                         
            ":",                                              
            __LINE__,                                         
            ": ",                                             
            ::caffe2::internal::cudnnGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! cudnn_check {
    ($condition:expr) => {
        todo!();
        /*
        cudnnStatus_t status = condition;                       
        CHECK(status == CUDNN_STATUS_SUCCESS)                   
            << ::caffe2::internal::cudnnGetErrorString(status); 
        */
    }
}

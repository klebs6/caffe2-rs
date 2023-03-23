crate::ix!();

#[macro_export] macro_rules! cublas_enforce {
    ($condition:ident) => {
        todo!();
        /*
        cublasStatus_t status = condition;           
        CAFFE_ENFORCE_EQ(                            
            status,                                  
            CUBLAS_STATUS_SUCCESS,                   
            "Error at: ",                            
            __FILE__,                                
            ":",                                     
            __LINE__,                                
            ": ",                                    
            ::caffe2::cublasGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! cublas_check {
    ($condition:ident) => {
        todo!();
        /*
        cublasStatus_t status = condition;             
        CHECK(status == CUBLAS_STATUS_SUCCESS)         
            << ::caffe2::cublasGetErrorString(status); 
        */
    }
}


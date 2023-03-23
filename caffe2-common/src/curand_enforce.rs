crate::ix!();

#[macro_export] macro_rules! curand_enforce {
    ($condition:ident) => {
        todo!();
        /*
        curandStatus_t status = condition;           
        CAFFE_ENFORCE_EQ(                            
            status,                                  
            CURAND_STATUS_SUCCESS,                   
            "Error at: ",                            
            __FILE__,                                
            ":",                                     
            __LINE__,                                
            ": ",                                    
            ::caffe2::curandGetErrorString(status)); 
        */
    }
}

#[macro_export] macro_rules! curand_check {
    ($condition:ident) => {
        todo!();
        /*
        curandStatus_t status = condition;             
        CHECK(status == CURAND_STATUS_SUCCESS)         
            << ::caffe2::curandGetErrorString(status); 
        */
    }
}


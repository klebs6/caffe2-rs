crate::ix!();

/// CUDA: various checks for different function calls.
#[macro_export] macro_rules! cuda_enforce {
    ($condition:ident, $($arg:ident),*) => {
        /*
        cudaError_t error = condition;   
        CAFFE_ENFORCE_EQ(                
            error,                       
            cudaSuccess,                 
            "Error at: ",                
            __FILE__,                    
            ":",                         
            __LINE__,                    
            ": ",                        
            cudaGetErrorString(error),   
            ##__VA_ARGS__);              
        */
    }
 }

#[macro_export] macro_rules! cuda_check {
    ($condition:ident) => {
        /*
        cudaError_t error = condition;                            
        CHECK(error == cudaSuccess) << cudaGetErrorString(error); 
        */
    }
 }

#[macro_export] macro_rules! cuda_driverapi_enforce {
    ($condition:ident) => {
        /*
        CUresult result = condition;                                     
        if (result != CUDA_SUCCESS) {                                    
            const char* msg;                                               
            cuGetErrorName(result, &msg);                                  
            CAFFE_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg); 
        }                                                                
        */
    }
}

#[macro_export] macro_rules! cuda_driverapi_check {
    ($condition:ident) => {
        /*
        CUresult result = condition;                                        
        if (result != CUDA_SUCCESS) {                                       
            const char* msg;                                                  
            cuGetErrorName(result, &msg);                                     
            LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": " 
                << msg;                                                
        }                                                                   
        */
    }
}


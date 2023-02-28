crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/cuda/CUDAException.h]

/**
  | Note [CHECK macro]
  | ~~~~~~~~~~~~~~~~~~
  |
  | This is a macro so that AT_ERROR can get
  | accurate __LINE__ and __FILE__ information.
  |
  | We could split this into a short macro and
  | a function implementation if we pass along
  | __LINE__ and __FILE__, but no one has found
  | this worth doing.
  |
  | Used to denote errors from Cuda framework.
  |
  | This needs to be declared here instead
  | util/Exception.h for proper conversion during
  | hipify.
  |
  */
#[cfg(feature = "cuda")]
pub struct CUDAError {
    base: Error,
}

// For Cuda Runtime API
#[cfg(STRIP_ERROR_MESSAGES)]
macro_rules! c10_cuda_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                           
            cudaError_t __err = EXPR;                                    
            if (__err != cudaSuccess) {                                  
              throw CUDAError(                                      
                  {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, 
                  TORCH_CHECK_MSG(false, ""));                           
            }                                                            
          } while (0)
        */
    }
}

#[cfg(not(STRIP_ERROR_MESSAGES))]
macro_rules! c10_cuda_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                              
            cudaError_t __err = EXPR;                                       
            if (__err != cudaSuccess) {                                     
              auto error_unused  = cudaGetLastError();            
              auto _cuda_check_suffix = get_cuda_check_suffix(); 
              throw CUDAError(                                         
                  {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},    
                  TORCH_CHECK_MSG(                                          
                      false,                                                
                      "",                                                   
                      "Cuda error: ",                                       
                      cudaGetErrorString(__err),                            
                      _cuda_check_suffix));                                 
            }                                                               
          } while (0)
        */
    }
}

macro_rules! c10_cuda_check_warn {
    ($EXPR:ident) => {
        /*
        
          do {                                                         
            cudaError_t __err = EXPR;                                  
            if (__err != cudaSuccess) {                                
              auto error_unused  = cudaGetLastError();       
              TORCH_WARN("Cuda warning: ", cudaGetErrorString(__err)); 
            }                                                          
          } while (0)
        */
    }
}

/**
  | This should be used directly after every kernel
  | launch to ensure the launch happened correctly
  | and provide an early, close-to-source
  | diagnostic if it didn't.
  |
  */
macro_rules! c10_cuda_kernel_launch_check {
    () => {
        /*
                C10_CUDA_CHECK(cudaGetLastError())
        */
    }
}

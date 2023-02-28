crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/Exceptions.h]

pub struct CuDNNError {
    base: Error,
}

#[macro_export] macro_rules! at_cudnn_check_with_shapes {
    ($EXPR:ident, $($arg:ident),*) => {
        /*
                AT_CUDNN_CHECK(EXPR, "\n", ##__VA_ARGS__)
        */
    }
}

/// See Note [CHECK macro]
///
#[macro_export] macro_rules! at_cudnn_check {
    ($EXPR:ident, $($arg:ident),*) => {
        /*
        
          do {                                                                                          
            cudnnStatus_t status = EXPR;                                                                
            if (status != CUDNN_STATUS_SUCCESS) {                                                       
              if (status == CUDNN_STATUS_NOT_SUPPORTED) {                                               
                TORCH_CHECK_WITH(CuDNNError, false,                                                     
                    "cuDNN error: ",                                                                    
                    cudnnGetErrorString(status),                                                        
                    ". This error may appear if you passed in a non-contiguous input.", ##__VA_ARGS__); 
              } else {                                                                                  
                TORCH_CHECK_WITH(CuDNNError, false,                                                     
                    "cuDNN error: ", cudnnGetErrorString(status), ##__VA_ARGS__);                       
              }                                                                                         
            }                                                                                           
          } while (0)
        */
    }
}

pub fn cublas_get_error_enum(error: CuBlasStatus) -> *const u8 {
    
    todo!();
        /*
        
        */
}

#[macro_export] macro_rules! torch_cudablas_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                          
            cublasStatus_t __err = EXPR;                                
            TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 
                        "CUDA error: ",                                 
                        blas::_cublasGetErrorEnum(__err),     
                        " when calling `" #EXPR "`");                   
          } while (0)
        */
    }
}

pub fn cusparse_get_error_string(status: CuSparseStatus) -> *const u8 {
    
    todo!();
        /*
        
        */
}

#[macro_export] macro_rules! torch_cudasparse_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                          
            cusparseStatus_t __err = EXPR;                              
            TORCH_CHECK(__err == CUSPARSE_STATUS_SUCCESS,               
                        "CUDA error: ",                                 
                        cusparseGetErrorString(__err),                  
                        " when calling `" #EXPR "`");                   
          } while (0)
        */
    }
}


// cusolver related headers are only supported on cuda now
#[cfg(CUDART_VERSION)]
pub fn cusolver_get_error_message(status: CuSolverStatus) -> *const u8 {
    
    todo!();
        /*
        
        */
}

#[cfg(CUDART_VERSION)]
#[macro_export] macro_rules! torch_cusolver_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                            
            cusolverStatus_t __err = EXPR;                                
            TORCH_CHECK(__err == CUSOLVER_STATUS_SUCCESS,                 
                        "cusolver error: ",                               
                        solver::cusolverGetErrorMessage(__err), 
                        ", when calling `" #EXPR "`");                    
          } while (0)
        */
    }
}

#[cfg(not(CUDART_VERSION))]
#[macro_export] macro_rules! torch_cusolver_check {
    ($EXPR:ident) => {
        /*
                EXPR
        */
    }
}

#[macro_export] macro_rules! at_cuda_check {
    ($EXPR:ident) => {
        /*
                C10_CUDA_CHECK(EXPR)
        */
    }
}

/**
  | For CUDA Driver API
  |
  | This is here instead of in c10 because NVRTC is
  | loaded dynamically via a stub in ATen, and we
  | need to use its nvrtcGetErrorString.
  |
  | See NOTE [ USE OF NVRTC AND DRIVER API ].
  */
#[cfg(not(__HIP_PLATFORM_HCC__))]
#[macro_export] macro_rules! at_cuda_driver_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                                                                           
            CUresult __err = EXPR;                                                                                       
            if (__err != CUDA_SUCCESS) {                                                                                 
              const char* err_str;                                                                                       
              CUresult get_error_str_err  = globalContext().getNVRTC().cuGetErrorString(__err, &err_str);  
              if (get_error_str_err != CUDA_SUCCESS) {                                                                   
                AT_ERROR("CUDA driver error: unknown error");                                                            
              } else {                                                                                                   
                AT_ERROR("CUDA driver error: ", err_str);                                                                
              }                                                                                                          
            }                                                                                                            
          } while (0)
        */
    }
}

#[cfg(__HIP_PLATFORM_HCC__)]
#[macro_export] macro_rules! at_cuda_driver_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                                            
            CUresult __err = EXPR;                                                        
            if (__err != CUDA_SUCCESS) {                                                  
              AT_ERROR("CUDA driver error: ", static_cast<int>(__err));                   
            }                                                                             
          } while (0)
        */
    }
}

/**
  | For CUDA NVRTC
  |
  | Note: As of CUDA 10, nvrtc error code 7,
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
  | incorrectly produces the error string "NVRTC
  | unknown error."
  |
  | The following maps it correctly.
  |
  | This is here instead of in c10 because NVRTC is
  | loaded dynamically via a stub in ATen, and we
  | need to use its nvrtcGetErrorString.
  |
  | See NOTE [ USE OF NVRTC AND DRIVER API ].
  */
#[macro_export] macro_rules! at_cuda_nvrtc_check {
    ($EXPR:ident) => {
        /*
        
          do {                                                                                              
            nvrtcResult __err = EXPR;                                                                       
            if (__err != NVRTC_SUCCESS) {                                                                   
              if (static_cast<int>(__err) != 7) {                                                           
                AT_ERROR("CUDA NVRTC error: ", globalContext().getNVRTC().nvrtcGetErrorString(__err));  
              } else {                                                                                      
                AT_ERROR("CUDA NVRTC error: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE");                        
              }                                                                                             
            }                                                                                               
          } while (0)
        */
    }
}

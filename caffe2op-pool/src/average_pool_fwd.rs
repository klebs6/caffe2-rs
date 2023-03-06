crate::ix!();

#[macro_export] macro_rules! caffe2_specialized_average_pool_functor_forward {
    ($T:ident, $($kOrder:ident)::+) => {
        /*
        
          template <>                                                                
          template <>                                                                
          bool AveragePoolFunctor<CPUContext>::Forward<T, kOrder>(                   
              const int N,                                                           
              const int C,                                                           
              const std::vector<int>& X_dims,                                        
              const std::vector<int>& Y_dims,                                        
              const std::vector<int>& kernel,                                        
              const std::vector<int>& dilation,                                      
              const std::vector<int>& stride,                                        
              const std::vector<int>& pads,                                          
              const T* X,                                                            
              T* Y,                                                                  
              CPUContext* /* context */) const {                                     
            const int ndim = X_dims.size();                                          
            switch (ndim) {                                                          
              case 1: {                                                              
                RunAveragePool1D<T, kOrder>(                                         
                    N,                                                               
                    C,                                                               
                    X_dims[0],                                                       
                    Y_dims[0],                                                       
                    kernel[0],                                                       
                    stride[0],                                                       
                    pads[0],                                                         
                    count_include_pad,                                               
                    X,                                                               
                    Y);                                                              
                return true;                                                         
              }                                                                      
              case 2: {                                                              
                if (std::is_same<T, float>::value && kOrder == StorageOrder::NCHW && 
                    pool_op_util::IsNeon4x4p0s0Eligible(                             
                        X_dims[0],                                                   
                        X_dims[1],                                                   
                        Y_dims[0],                                                   
                        Y_dims[1],                                                   
                        kernel[0],                                                   
                        kernel[1],                                                   
                        stride[0],                                                   
                        stride[1],                                                   
                        pads[0],                                                     
                        pads[1],                                                     
                        pads[2],                                                     
                        pads[3],                                                     
                        dilation[0],                                                 
                        dilation[1],                                                 
                        X,                                                           
                        Y)) {                                                        
                  pool_op_util::RunNeonAveragePool4x4p0s0NCHW(                       
                      N, C, X_dims[0], X_dims[1], X, Y);                             
                } else {                                                             
                  RunAveragePool2D<T, kOrder>(                                       
                      N,                                                             
                      C,                                                             
                      X_dims[0],                                                     
                      X_dims[1],                                                     
                      Y_dims[0],                                                     
                      Y_dims[1],                                                     
                      kernel[0],                                                     
                      kernel[1],                                                     
                      stride[0],                                                     
                      stride[1],                                                     
                      pads[0],                                                       
                      pads[1],                                                       
                      count_include_pad,                                             
                      X,                                                             
                      Y);                                                            
                }                                                                    
                return true;                                                         
              }                                                                      
              case 3: {                                                              
                RunAveragePool3D<T, kOrder>(                                         
                    N,                                                               
                    C,                                                               
                    X_dims[0],                                                       
                    X_dims[1],                                                       
                    X_dims[2],                                                       
                    Y_dims[0],                                                       
                    Y_dims[1],                                                       
                    Y_dims[2],                                                       
                    kernel[0],                                                       
                    kernel[1],                                                       
                    kernel[2],                                                       
                    stride[0],                                                       
                    stride[1],                                                       
                    stride[2],                                                       
                    pads[0],                                                         
                    pads[1],                                                         
                    pads[2],                                                         
                    count_include_pad,                                               
                    X,                                                               
                    Y);                                                              
                return true;                                                         
              }                                                                      
              default: {                                                             
                CAFFE_THROW("Unsupported pooling dim: ", ndim);                      
                return false;                                                        
              }                                                                      
            }                                                                        
          }
        */
    }
}

caffe2_specialized_average_pool_functor_forward!{f32, StorageOrder::NCHW}
caffe2_specialized_average_pool_functor_forward!{f32, StorageOrder::NHWC}

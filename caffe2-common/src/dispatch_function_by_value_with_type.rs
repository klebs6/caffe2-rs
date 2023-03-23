crate::ix!();

#[macro_export] macro_rules! dispatch_function_by_value_with_type_1 {
    ($val:ident, $Func:ident, $T:ident, $($arg:ident),*) => {
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                    
        switch (val) {                                                
            case 1: {                                                   
                Func<T, 1>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 2: {                                                   
                Func<T, 2>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 3: {                                                   
                Func<T, 3>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 4: {                                                   
                Func<T, 4>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 5: {                                                   
                Func<T, 5>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 6: {                                                   
                Func<T, 6>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 7: {                                                   
                Func<T, 7>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            case 8: {                                                   
                Func<T, 8>(__VA_ARGS__);                                  
                break;                                                    
            }                                                           
            default: {                                                  
                break;                                                    
            }                                                           
        }                                                             
        */
    }
}

#[macro_export] macro_rules! dispatch_function_by_value_with_type_2 {
    ($val:ident, 
    $Func:ident, 
    $T1:ident, 
    $T2:ident, 
    $($arg:ident),*) => { 
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                         
        switch (val) {                                                     
            case 1: {                                                        
                Func<T1, T2, 1>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 2: {                                                        
                Func<T1, T2, 2>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 3: {                                                        
                Func<T1, T2, 3>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 4: {                                                        
                Func<T1, T2, 4>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 5: {                                                        
                Func<T1, T2, 5>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 6: {                                                        
                Func<T1, T2, 6>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 7: {                                                        
                Func<T1, T2, 7>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            case 8: {                                                        
                Func<T1, T2, 8>(__VA_ARGS__);                                  
                break;                                                         
            }                                                                
            default: {                                                       
                break;                                                         
            }                                                                
        }                                                                  
        */
    }
}

#[macro_export] macro_rules! dispatch_function_by_value_with_type_3 {
    ($val:ident, $Func:ident, $T1:ident, $T2:ident, $T3:ident, $($arg:ident),*) => {
        /*
        CAFFE_ENFORCE_LE(val, kCUDATensorMaxDims);                             
        switch (val) {                                                         
            case 1: {                                                            
                Func<T1, T2, T3, 1>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 2: {                                                            
                Func<T1, T2, T3, 2>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 3: {                                                            
                Func<T1, T2, T3, 3>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 4: {                                                            
                Func<T1, T2, T3, 4>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 5: {                                                            
                Func<T1, T2, T3, 5>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 6: {                                                            
                Func<T1, T2, T3, 6>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 7: {                                                            
                Func<T1, T2, T3, 7>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            case 8: {                                                            
                Func<T1, T2, T3, 8>(__VA_ARGS__);                                  
                break;                                                             
            }                                                                    
            default: {                                                           
                break;                                                             
            }                                                                    
        }                                                                      
        */
    }
}

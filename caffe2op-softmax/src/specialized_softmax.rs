crate::ix!();

#[inline] pub fn softmaxcpu<T>(
    n:           i32,
    d:           i32,
    logarithmic: bool,
    x:           *const T,
    y:           *mut T,
    scratch:     *mut T,
    context:     *mut CPUContext)  {

    todo!();
    /*
    
    */
}

caffe2_specialized_softmax_cpu!{f32}

#[macro_export] macro_rules! caffe2_specialized_softmax_cpu {
    ($T:ident) => {
        /*
        
          template <>                                                    
          void SoftmaxCPU<T>(                                            
              const int N,                                               
              const int D,                                               
              const bool logarithmic,                                    
              const T* X,                                                
              T* Y,                                                      
              T* scratch,                                                
              CPUContext* context) {                                     
            ConstEigenArrayMap<T> X_arr(X, D, N);                        
            EigenArrayMap<T> Y_arr(Y, D, N);                             
            EigenVectorArrayMap<T> scratch_arr(scratch, N);              
            scratch_arr = X_arr.colwise().maxCoeff().transpose();        
            Y_arr = X_arr.rowwise() - scratch_arr.transpose();           
            math::Exp<T, CPUContext>(N * D, Y, Y, context);              
            if (logarithmic) {                                           
              scratch_arr += Y_arr.colwise().sum().log().transpose();    
              Y_arr = X_arr.rowwise() - scratch_arr.transpose();         
            } else {                                                     
              scratch_arr = Y_arr.colwise().sum().inverse().transpose(); 
              Y_arr = Y_arr.rowwise() * scratch_arr.transpose();         
            }                                                            
          }
        */
    }
}

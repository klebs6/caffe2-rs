crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ForeachOpsKernels.cpp]

#[macro_export] macro_rules! foreach_binary_op_scalar {
    ($OP:ident) => {
        /*
        
        void foreach_tensor_##OP##_scalar_kernel_slow_(&[Tensor] tensors, const Scalar& scalar) {                       
          check_foreach_api_restrictions(tensors);                                                                
                                                                                                                  
          for (auto& t: tensors) {                                                                                
            t.OP##_(scalar);                                                                                      
          }                                                                                                       
        }                                                                                                         
                                                                                                                  
        vector<Tensor> foreach_tensor_##OP##_scalar_kernel_slow(&[Tensor] tensors, const Scalar& scalar) {         
          check_foreach_api_restrictions(tensors);                                                                
                                                                                                                  
          vector<Tensor> result;                                                                             
          result.reserve(tensors.size());                                                                         
          for (const auto& t: tensors) {                                                                          
            result.emplace_back(t.OP(scalar));                                                                    
          }                                                                                                       
                                                                                                                  
          return result;                                                                                          
        }
        */
    }
}

#[macro_export] macro_rules! foreach_binary_op_scalarlist {
    ($OP:ident) => {
        /*
        
        void foreach_tensor_##OP##_scalarlist_kernel_slow_(&[Tensor] tensors, ArrayRef<Scalar> scalars) {                  
          check_foreach_api_restrictions(tensors, scalars);                                                                     
                                                                                                                                
          for (const auto i : irange(tensors.size())) {                                                                    
              tensors[i].OP##_(scalars[i]);                                                                                     
            }                                                                                                                   
        }                                                                                                                       
                                                                                                                                
        vector<Tensor> foreach_tensor_##OP##_scalarlist_kernel_slow(&[Tensor] tensors, ArrayRef<Scalar> scalars) {    
          check_foreach_api_restrictions(tensors, scalars);                                                                     
          vector<Tensor> result;                                                                                           
          result.reserve(tensors.size());                                                                                       
          for (const auto i : irange(tensors.size())) {                                                                    
            result.emplace_back(tensors[i].OP(scalars[i]));                                                                     
          }                                                                                                                     
                                                                                                                                
          return result;                                                                                                        
        }
        */
    }
}

#[macro_export] macro_rules! foreach_binary_op_list {
    ($OP:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(&[Tensor] tensors1, &[Tensor] tensors2) {    
          check_foreach_api_restrictions(tensors1, tensors2);                                                     
                                                                                                                  
          vector<Tensor> result;                                                                             
          result.reserve(tensors1.size());                                                                        
          for (const auto i : irange(tensors1.size())) {                                                     
            result.emplace_back(tensors1[i].OP(tensors2[i]));                                                     
          }                                                                                                       
                                                                                                                  
          return result;                                                                                          
        }                                                                                                         
                                                                                                                  
        void foreach_tensor_##OP##_list_kernel_slow_(&[Tensor] tensors1, &[Tensor] tensors2) {                  
          check_foreach_api_restrictions(tensors1, tensors2);                                                     
                                                                                                                  
          for (const auto i : irange(tensors1.size())) {                                                     
            tensors1[i].OP##_(tensors2[i]);                                                                       
          }                                                                                                       
        }
        */
    }
}

#[macro_export] macro_rules! foreach_binary_op_list_alpha {
    ($OP:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##OP##_list_kernel_slow(&[Tensor] tensors1, &[Tensor] tensors2, const Scalar& alpha) {    
          check_foreach_api_restrictions(tensors1, tensors2);                                                                   
                                                                                                                                
          vector<Tensor> result;                                                                                           
          result.reserve(tensors1.size());                                                                                      
          for (const auto i : irange(tensors1.size())) {                                                                   
            result.emplace_back(tensors1[i].OP(tensors2[i], alpha));                                                            
          }                                                                                                                     
                                                                                                                                
          return result;                                                                                                        
        }                                                                                                                       
                                                                                                                                
        void foreach_tensor_##OP##_list_kernel_slow_(&[Tensor] tensors1, &[Tensor] tensors2, const Scalar& alpha) {                  
          check_foreach_api_restrictions(tensors1, tensors2);                                                                   
                                                                                                                                
          for (const auto i : irange(tensors1.size())) {                                                                   
            tensors1[i].OP##_(tensors2[i], alpha);                                                                              
          }                                                                                                                     
        }
        */
    }
}

#[macro_export] macro_rules! foreach_unary_op {
    ($OP:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##OP##_slow(&[Tensor] tensors) {       
          check_foreach_api_restrictions(tensors);                                 
                                                                                   
          vector<Tensor> result;                                              
          result.reserve(tensors.size());                                          
          for (const auto& t : tensors) {                                          
            result.emplace_back(t.OP());                                           
          }                                                                        
                                                                                   
          return result;                                                           
        }                                                                          
                                                                                   
        void foreach_tensor_##OP##_slow_(&[Tensor] tensors) {                     
          check_foreach_api_restrictions(tensors);                                 
                                                                                   
          for (auto& t : tensors) {                                                
            t.OP##_();                                                             
          }                                                                        
        }
        */
    }
}

#[macro_export] macro_rules! foreach_pointwise_op_scalar {
    ($OP:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##OP##_scalar_slow(&[Tensor] input, &[Tensor] tensors1, &[Tensor] tensors2, const Scalar& scalar) {   
          check_foreach_api_restrictions(input, tensors1, tensors2);                                                                         
                                                                                                                                             
          vector<Tensor> result;                                                                                                        
          for(const auto i : irange(input.size())) {                                                                                    
            result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar));                                                              
          }                                                                                                                                  
                                                                                                                                             
          return result;                                                                                                                     
        }                                                                                                                                    
                                                                                                                                             
        void foreach_tensor_##OP##_scalar_slow_(&[Tensor] input, &[Tensor] tensors1, &[Tensor] tensors2, const Scalar& scalar) {                 
          check_foreach_api_restrictions(input, tensors1, tensors2);                                                                         
                                                                                                                                             
          for(const auto i : irange(input.size())) {                                                                                    
            input[i].OP##_(tensors1[i], tensors2[i], scalar);                                                                                
          }                                                                                                                                  
        }                                                                                                                                    
        */
    }
}

#[macro_export] macro_rules! foreach_pointwise_op_scalarlist {
    ($OP:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##OP##_scalarlist_slow(&[Tensor] input, &[Tensor] tensors1, &[Tensor] tensors2, ArrayRef<Scalar> scalars) {   
          check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                   
                                                                                                                                                                
          vector<Tensor> result;                                                                                                                           
          for(const auto i : irange(input.size())) {                                                                                                       
            result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i]));                                                                             
          }                                                                                                                                                     
                                                                                                                                                                
          return result;                                                                                                                                        
        }                                                                                                                                                       
                                                                                                                                                                
        void foreach_tensor_##OP##_scalarlist_slow_(&[Tensor] input, &[Tensor] tensors1, &[Tensor] tensors2, ArrayRef<Scalar> scalars) {                 
          check_foreach_api_restrictions(input, tensors1, tensors2, scalars);                                                                                   
                                                                                                                                                                
          for(const auto i : irange(input.size())) {                                                                                                       
            input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                                                                                               
          }                                                                                                                                                     
        }                                                                                                                                                       
        */
    }
}


foreach_binary_op_list_alpha!(add);
foreach_binary_op_list_alpha!(sub);

foreach_binary_op_scalar!(add);
foreach_binary_op_scalar!(sub);
foreach_binary_op_scalar!(mul);
foreach_binary_op_scalar!(div);

foreach_binary_op_scalarlist!(add);
foreach_binary_op_scalarlist!(sub);
foreach_binary_op_scalarlist!(mul);
foreach_binary_op_scalarlist!(div);

foreach_binary_op_list!(mul);
foreach_binary_op_list!(div);

foreach_unary_op!(sqrt);
foreach_unary_op!(exp);
foreach_unary_op!(abs);
foreach_unary_op!(acos);
foreach_unary_op!(asin);
foreach_unary_op!(atan);
foreach_unary_op!(ceil);
foreach_unary_op!(cos);
foreach_unary_op!(cosh);
foreach_unary_op!(erf);
foreach_unary_op!(erfc);
foreach_unary_op!(expm1);
foreach_unary_op!(floor);
foreach_unary_op!(log);
foreach_unary_op!(log10);
foreach_unary_op!(log1p);
foreach_unary_op!(log2);
foreach_unary_op!(neg);
foreach_unary_op!(tan);
foreach_unary_op!(tanh);
foreach_unary_op!(sin);
foreach_unary_op!(sinh);
foreach_unary_op!(round);
foreach_unary_op!(lgamma);
foreach_unary_op!(frac);
foreach_unary_op!(trunc);
foreach_unary_op!(reciprocal);
foreach_unary_op!(sigmoid);

foreach_pointwise_op_scalar!(addcdiv);
foreach_pointwise_op_scalar!(addcmul);

foreach_pointwise_op_scalarlist!(addcdiv);
foreach_pointwise_op_scalarlist!(addcmul);

#[macro_export] macro_rules! foreach_maximum_minimum_op {
    ($NAME:ident) => {
        /*
        
        vector<Tensor> foreach_tensor_##NAME##_slow(&[Tensor] tensors1, &[Tensor] tensors2) { 
          check_foreach_api_restrictions(tensors1, tensors2);                                        
                                                                                                     
          vector<Tensor> result;                                                                
          result.reserve(tensors1.size());                                                           
          for (const auto i : irange(tensors1.size())) {                                        
            result.emplace_back(NAME(tensors1[i], tensors2[i]));                                 
          }                                                                                          
                                                                                                     
          return result;                                                                             
        }                                                                                            
        */
    }
}

foreach_maximum_minimum_op!{maximum}
foreach_maximum_minimum_op!{minimum}

pub fn foreach_tensor_zero_slow(tensors: &[Tensor])  {
    
    todo!();
        /*
            check_foreach_api_restrictions(tensors);

      for (auto& t : tensors) {
        t.zero_();
      }
        */
}

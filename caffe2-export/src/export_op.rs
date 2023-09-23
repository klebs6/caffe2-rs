crate::ix!();

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_cpu {
    ($OperatorName:ident, $Name:ident) => {
        todo!();
        /*
        
          REGISTER_CPU_OPERATOR_CREATOR(                               
              Name,                                                    
              ::caffe2::detail::createC10OperatorWrapper<CPUContext>(  
                  {OperatorName, ""}))
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_cuda {
    ($OperatorName:ident, $Name:ident) => {
        todo!();
        /*
        
          REGISTER_CUDA_OPERATOR_CREATOR(                              
              Name,                                                    
              ::caffe2::detail::createC10OperatorWrapper<CUDAContext>( 
                  {OperatorName, ""}))
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_hip {
    ($OperatorName:ident, $Name:ident) => {
        todo!();
        /*
        
          REGISTER_HIP_OPERATOR_CREATOR(                               
              Name,                                                    
              ::caffe2::detail::createC10OperatorWrapper<HIPContext>(  
                  {OperatorName, ""}))
        */
    }
}

#[cfg(any(caffe2_is_xplat_build, c10_mobile))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_cpu {
    () => {
        todo!();
        /*
                (OperatorName, Name)
        */
    }
}

#[cfg(any(caffe2_is_xplat_build, c10_mobile))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_cuda {
    () => {
        todo!();
        /*
                (OperatorName, Name)
        */
    }
}

#[cfg(any(caffe2_is_xplat_build, c10_mobile))]
#[macro_export] macro_rules! export_c10_op_to_caffe2_hip {
    () => {
        todo!();
        /*
                (OperatorName, Name)
        */
    }
}

/**
  | To register a caffe2 operator caffe2::MyOperator
  | with the c10 dispatcher, call:
  | 
  | In caffe2/operators/MyOperator.h:
  | > C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(C10MyOperator)
  | // C10MyOperator is the name used by
  | c10 for this operator
  | 
  | In caffe2/operators/MyOperator.cc
  | > C10_EXPORT_CAFFE2_OP_TO_C10_CPU
  | ( > C10MyOperator, > "_caffe2::C10MyOperator(Tensor
  | input1, int argument2, float argument3)
  | 
  | -> (Tensor output1, Tensor output2)"
  | > caffe2::MyOperator<caffe2::CPUContext>
  | // This is the caffe2 operator > // class
  | template > )
  | 
  | In caffe2/operators/MyOperator.cu
  | > C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(C10MyOperator
  | , caffe2::MyOperator<caffe2::CUDAContext>)
  | 
  | Notes:
  | 
  | - all macros must be defined in the top
  | level namespace, not in namespace caffe2.
  | 
  | - all operators must call
  | 
  | C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10
  | and
  | 
  | C10_EXPORT_CAFFE2_OP_TO_C10_CPU
  | .
  | 
  | - calling C10_EXPORT_CAFFE2_OP_TO_C10_CUDA
  | is optional and can be omitted i f you
  | don't want to expose the operator for
  | CUDA operations.
  | 
  | - caffe2 arguments must come after caffe2
  | inputs, in other words, any tensor inputs
  | must precede any non-tensor inputs.
  | 
  | More complex use cases:
  | 
  | - If your operator has a variable number
  | of input tensors, make the first (!)
  | input an input of type &[Tensor]. There
  | must be no other tensor inputs.
  |
  */
#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_declare_export_caffe2_op_to_c10 {
    ($OperatorName:ident) => {
        todo!();
        /*
        
          namespace caffe2 {                                        
          namespace _c10_ops {                                      
          TORCH_API const FunctionSchema& schema_##OperatorName(); 
          }                                                         
          }
        */
    }
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_export_caffe2_op_to_c10_schema_only {
    ($OperatorName:ident, $OperatorSchema:ident) => {
        todo!();
        /*
        
          /* Register the op schema with the c10 dispatcher */                        
          namespace caffe2 {                                                          
          namespace _c10_ops {                                                        
          C10_EXPORT const FunctionSchema& schema_##OperatorName() {                  
            static const FunctionSchema schema =                                      
                ::caffe2::detail::make_function_schema_for_c10(OperatorSchema);       
            return schema;                                                            
          }                                                                           
          TORCH_LIBRARY_FRAGMENT(_caffe2, m) {                                        
              m.def(::caffe2::detail::make_function_schema_for_c10(OperatorSchema));  
          }                                                                           
          }                                                                           
          }
        */
    }
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_export_caffe2_op_to_c10_cpu_kernel_only {
    ($OperatorName:ident, $OperatorClass:ident) => {
        todo!();
        /*
        
          /* Register call_caffe2_op_from_c10 as a kernel with the c10 dispatcher */ 
            TORCH_LIBRARY_IMPL(_caffe2, CPU, m) {                                    
                m.impl("_caffe2::" #OperatorName,                                    
                    torch::CppFunction::makeFromBoxedFunction<                       
                        ::caffe2::detail::call_caffe2_op_from_c10<                   
                            ::caffe2::_c10_ops::schema_##OperatorName,               
                            OperatorClass>>());                                      
            }
        */
    }
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_export_caffe2_op_to_c10_cpu {
    ($OperatorName:ident, $OperatorSchema:ident, $OperatorClass:ident) => {
        todo!();
        /*
        
          C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(OperatorName, OperatorSchema)      
          C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(OperatorName, OperatorClass)
        */
    }
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_export_caffe2_op_to_c10_cuda {
    ($OperatorName:ident, $OperatorClass:ident) => {
        todo!();
        /*
        
          /* Register call_caffe2_op_from_c10 as a kernel with the c10 dispatcher */ 
            TORCH_LIBRARY_IMPL(_caffe2, CUDA, m) {                                   
                m.impl("_caffe2::" #OperatorName,                                    
                    torch::CppFunction::makeFromBoxedFunction<                       
                        ::caffe2::detail::call_caffe2_op_from_c10<                   
                            ::caffe2::_c10_ops::schema_##OperatorName,               
                            OperatorClass>>());                                      
            }
        */
    }
}

/**
  | You should never manually call the
  | C10_EXPORT_CAFFE2_OP_TO_C10_HIP macro .
  |
  | The C10_EXPORT_CAFFE2_OP_TO_C10_CUDA macro
  | from above will be automatically rewritten to
  | C10_EXPORT_CAFFE2_OP_TO_C10_HIP by hipify .
  */
#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
macro_rules! c10_export_caffe2_op_to_c10_hip {
    ($OperatorName:ident, $OperatorClass:ident) => {
        todo!();
        /*
        
          /* Register call_caffe2_op_from_c10 as a kernel with the c10 dispatcher */ 
            TORCH_LIBRARY_IMPL(_caffe2, HIP, m) {                                    
                m.impl("_caffe2::" #OperatorName,                                    
                    torch::CppFunction::makeFromBoxedFunction<                       
                        ::caffe2::detail::call_caffe2_op_from_c10<                   
                            ::caffe2::_c10_ops::schema_##OperatorName,               
                            OperatorClass>>());                                      
            }
        */
    }
}

/**
  | Don't use c10 dispatcher on mobile because
  | of binary size
  |
  */
#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_declare_export_caffe2_op_to_c10 {
    () => {
        todo!();
        /*
                (OperatorName)
        */
    }
}

#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_export_caffe2_op_to_c10_schema_only {
    () => {
        todo!();
        /*
                (OperatorName, OperatorSchema)
        */
    }
}

#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_export_caffe2_op_to_c10_cpu_kernel_only {
    () => {
        todo!();
        /*
                (OperatorName, OperatorClass)
        */
    }
}

#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_export_caffe2_op_to_c10_cpu {
    () => {
        todo!();
        /*
                ( OperatorName, OperatorSchema, OperatorClass)
        */
    }
}

#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_export_caffe2_op_to_c10_cuda {
    () => {
        todo!();
        /*
                (OperatorName, OperatorClass)
        */
    }
}

#[cfg(not(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile)))))]
macro_rules! c10_export_caffe2_op_to_c10_hip {
    () => {
        todo!();
        /*
                (OperatorName, OperatorClass)
        */
    }
}

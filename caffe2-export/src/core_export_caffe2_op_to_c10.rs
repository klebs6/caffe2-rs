crate::ix!();

use crate::{
    FunctionSchema,
    IValue,
    OperatorHandle,
    Stack,
    Tensor,
};

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
pub const PREALLOCATED_OUTPUT_ARGNAME: &'static str = "_caffe2_preallocated_outputs";

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
pub type CallCaffe2OpFunc = 
    fn(schema: &FunctionSchema,
        inputs: Vec<IValue>,
        outputs: LinkedList<Tensor>) -> LinkedList<Tensor>;

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
pub fn call_caffe2_op<Caffe2Operator>(
    schema: &FunctionSchema,
    inputs: Vec<IValue>,
    outputs: LinkedList<Tensor>) -> LinkedList<Tensor> 
{
    todo!();
    /*
      Caffe2Operator op(schema, std::move(inputs), std::move(outputs));
      op.Run();
      return std::move(op).move_newstyle_outputs();
    */
}

/**
  | This function is inline in the hope that
  | compilers optimizing for speed will
  | inline it into call_caffe2_op_from_c10,
  | allowing call_op to be inlined and avoiding
  | the function pointer indirection,
  | while compilers optimizing for binary
  | size will keep it a separate function
  | instead of inlining it into a template
  | and will reuse the binary code of this
  | function between ops.
  | 
  | We measured and confirmed that binary
  | size off the instagram ios app is reduced
  | when having _call_caffe2_op_from_c10
  | separate from the templated call_caffe2_op_from_c10.
  |
  */
#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn call_caffe2_op_from_c10(
    stack:   *mut Stack,
    schema:  &FunctionSchema,
    call_op: *mut CallCaffe2OpFunc)
{
    
    todo!();
    /*
        // precondition: on the stack, there's one IValue for each argument of the
      // c10 schema. The last argument is an optional tensor list that
      // (if not ivalue::None) contains a preallocated output tensor for each
      // operator output.

      // As an invariant, we don't want any autograd gradients to be tracked in
      // Caffe2 operators.
      at::NoGradGuard guard;

      AT_ASSERT(
          schema.arguments().size() != 0 &&
          schema.arguments().back().type()->isSubtypeOf(
              OptionalType::create(ListType::ofTensors())));
      IValue preallocated_outputs = torch::jit::pop(*stack);

      const size_t num_outputs = schema.returns().size();
      const size_t num_inputs = schema.arguments().size() -
          1; // -1 because the last argument is the list of preallocated tensors

      c10::List<at::Tensor> outputs;
      if (preallocated_outputs.isNone()) {
        // either the schema doesn't support preallocated outputs or it does but
        // they haven't been passed in. Pass a list of uninitialized tensors to
        // the caffe2 operator as preallocated outputs.
        outputs.resize(num_outputs);
      } else {
        AT_ASSERT(preallocated_outputs.isTensorList());
        outputs = std::move(preallocated_outputs).toTensorList();
      }

      // TODO Avoid vector allocation. One idea would be to keep the std::vector
      // instances in the cache.
      std::vector<IValue> inputs = torch::jit::pop(*stack, num_inputs);

      outputs = (*call_op)(schema, std::move(inputs), std::move(outputs));

      bool return_tensor_list = false;
      if (schema.returns().size() == 1) {
        auto type = schema.returns()[0].type();
        if (c10::ListTypePtr list_type = type->cast<c10::ListType>()) {
          if (list_type->getElementType()->kind() == c10::TypeKind::TensorType) {
            return_tensor_list = true;
          }
        }
      }
      if (return_tensor_list) {
        // We should not unwrap the list if we expect tensor list in the schema.
        torch::jit::push(*stack, outputs);
      } else {
        for (size_t i = 0; i < outputs.size(); ++i) {
          torch::jit::push(*stack, outputs.extract(i));
        }
      }

      // postcondition: All inputs are cleared from the stack, there's now one
      //                IValue for each output which holds the result. This
      //                might reuse one of the preallocated tensors but doesn't have
      //                to.
    */
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn call_caffe2_op_from_c10_default_schema<'a, Caffe2Operator>(
    op_handle: &OperatorHandle, 
    stack:     *mut Stack,
    schema:    fn() -> &'a FunctionSchema)  
{
    
    todo!();
    /*
        _call_caffe2_op_from_c10(stack, Schema(), &_call_caffe2_op<Caffe2Operator>);
    */
}

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn make_function_schema_for_c10(schema_str: *const u8) -> FunctionSchema {
    
    todo!();
    /*
        #if !defined(EXPOSE_C2_OPS) && \
        (defined(CAFFE2_IS_XPLAT_BUILD) || defined(C10_MOBILE))
      throw std::logic_error(
          "We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
    #else
      c10::FunctionSchema parsed_schema = torch::jit::parseSchema(schema_str);
      std::vector<c10::Argument> arguments = parsed_schema.arguments();
      arguments.emplace_back(
          PREALLOCATED_OUTPUT_ARGNAME,
          c10::OptionalType::create(c10::ListType::ofTensors()),
          nullopt,
          IValue());

      return FunctionSchema(
          parsed_schema.name(),
          parsed_schema.overload_name(),
          std::move(arguments),
          parsed_schema.returns(),
          parsed_schema.is_vararg(),
          parsed_schema.is_varret());
    #endif
    */
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
  | input an input of type TensorList. There
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

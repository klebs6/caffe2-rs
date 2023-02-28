crate::ix!();

use crate::{
    OperatorHandle,
    List,
    IValue,
    OperatorName,
    OperatorStorage,
    Argument,
    Tensor,
    Workspace,
    Operator,
    OperatorDef,
};

/**
  | To make a c10 operator "C10Add" callable
  | from caffe2 as "C2MyAddOpName", just
  | write
  | 
  | To export the CPU kernel
  | 
  | C10_EXPORT_C10_OP_TO_CAFFE2_CPU(C10Add,
  | C2MyAddOp)
  | 
  | To export the CUDA kernel
  | 
  | C10_EXPORT_C10_OP_TO_CAFFE2_CUDA(C10Add,
  | C2MyAddOp)
  |
  */
#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
pub struct C10OperatorWrapper<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    op: OperatorHandle,

    /**
      | has_preallocated_outputs_ is true iff the
      | operator schema has a last argument that
      | is a TensorList and has a name equal to
      | with the name equal to
      | detail::PREALLOCATED_OUTPUT_ARGNAME. This
      | argument is then used to pass in
      | preallocated output tensors to the caffe2
      | operator.
      */
    has_preallocated_outputs: bool,

    /**
      | this is stored as a member here to avoid
      | having to re-allocate a stack for each
      | call. Between kernel calls, stack_.size()
      | == 0, but capacity should not need to be
      | grown anymore after the first call.
      */
    stack: Vec<IValue>,
    mutex: parking_lot::RawMutex,
}

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
impl<Context> C10OperatorWrapper<Context> {

    pub fn new(
        op:           &OperatorHandle,
        operator_def: &OperatorDef,
        ws:           *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            op_(op),
            has_preallocated_outputs_(
                op_.schema().arguments().size() != 0 &&
                op_.schema().arguments().back().name() ==
                    detail::PREALLOCATED_OUTPUT_ARGNAME) 

        AT_ASSERT(
            !has_preallocated_outputs_ ||
            op_.schema().arguments().back().type()->isSubtypeOf(
                OptionalType::create(ListType::ofTensors())));

        AT_ASSERT(operator_def.output_size() == op_.schema().returns().size());
        AT_ASSERT(
            operator_def.input_size() + (has_preallocated_outputs_ ? 1 : 0) <=
            op_.schema()
                .arguments()
                .size()); // '<=' because there might be caffe2 nontensor arguments
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // due to caching the stack_, concurrent calling is not allowed.
        // TODO thread_local might fix this
        std::lock_guard<std::mutex> lock(mutex_);

        pushInputs_();
        callKernel_();
        popOutputs_();

        return true;
        */
    }
    
    #[inline] pub fn push_inputs(&mut self)  {
        
        todo!();
        /*
            AT_ASSERT(stack_.size() == 0);
        stack_.reserve(
            op_.schema().arguments().size() + (has_preallocated_outputs_ ? 1 : 0));

        size_t input_tensor_index = 0;

        for (const auto& argument : op_.schema().arguments()) {
          if (argument.name() == detail::PREALLOCATED_OUTPUT_ARGNAME) {
            // note: if detail::PREALLOCATED_OUTPUT_ARGNAME was at the end of the
            // argument list, then has_preallocated_outputs_ would be true.
            AT_ASSERTM(
                has_preallocated_outputs_,
                "Error in caffe2->c10 wrapper: Operator schema has a parameter named ",
                detail::PREALLOCATED_OUTPUT_ARGNAME,
                ", but it's not at the end of the argument list");

            AT_ASSERTM(
                argument.type()->isSubtypeOf(
                    OptionalType::create(ListType::ofTensors())),
                "Error in caffe2->c10 wrapper: Operator schema has a parameter named ",
                detail::PREALLOCATED_OUTPUT_ARGNAME,
                ", but it's not of type TensorList?");
            stack_.emplace_back(preallocated_outputs_());

          } else if (argument.type()->isSubtypeOf(TensorType::get())) {
            AT_ASSERTM(
                input_tensor_index < InputSize(),
                "Error in caffe2->c10 wrapper: Too few tensor arguments given (",
                InputSize(),
                "), operator schema expected more.");
            stack_.emplace_back(at::Tensor(Input(input_tensor_index++)));
          } else if (argument.type()->isSubtypeOf(OptionalType::ofTensor())) {
            if (input_tensor_index < InputSize()) {
              stack_.emplace_back(at::Tensor(Input(input_tensor_index++)));
            } else {
              stack_.emplace_back(IValue());
            }
          } else if (argument.type()->isSubtypeOf(ListType::ofTensors())) {
            AT_ASSERTM(
                input_tensor_index == 0,
                "Error in caffe2->c10 wrapper: Schema can only have either one or more Tensor inputs or one TensorList input.");
            stack_.emplace_back(array_inputs_());
            input_tensor_index = InputSize();

          } else {
            stack_.emplace_back(get_nontensor_argument_(argument));
          }
        }
        AT_ASSERTM(
            input_tensor_index == InputSize(),
            "Error in caffe2->c10 wrapper: Number of caffe2 operator inputs (",
            InputSize(),
            ") doesn't match number of tensor arguments (",
            input_tensor_index,
            ") in the c10 operator schema.");
        */
    }
    
    #[inline] pub fn call_kernel(&mut self)  {
        
        todo!();
        /*
            AT_ASSERT(stack_.size() == op_.schema().arguments().size());
        op_.callBoxed(&stack_);
        */
    }
    
    #[inline] pub fn pop_outputs(&mut self)  {
        
        todo!();
        /*
            AT_ASSERT(stack_.size() == op_.schema().returns().size());
        for (size_t i = 0; i < op_.schema().returns().size(); ++i) {
          OperatorStorage::SetOutputTensor(i, Tensor(std::move(stack_[i]).toTensor()));
        }
        stack_.clear();
        */
    }
    
    #[inline] pub fn array_inputs(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            c10::List<at::Tensor> result;
        result.reserve(InputSize());
        for (size_t i = 0; i < InputSize(); ++i) {
          result.emplace_back(Input(i));
        }
        return result;
        */
    }
    
    #[inline] pub fn preallocated_outputs(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            c10::List<at::Tensor> result;
        result.reserve(OutputSize());
        for (size_t i = 0; i < OutputSize(); ++i) {
          result.emplace_back(OperatorStorage::OutputTensorOrUndefined(i));
        }
        return result;
        */
    }
    
    #[inline] pub fn get_nontensor_argument(&mut self, argument: &Argument) -> IValue {
        
        todo!();
        /*
            if (argument.type()->isSubtypeOf(IntType::get())) {
          return get_nontensor_argument_<int>(
              argument.name(), argument.default_value());
        } else if (argument.type()->isSubtypeOf(FloatType::get())) {
          return get_nontensor_argument_<double>(
              argument.name(), argument.default_value());
        } else if (argument.type()->isSubtypeOf(BoolType::get())) {
          return get_nontensor_argument_<bool>(
              argument.name(), argument.default_value());
        } else {
          // TODO Support more types
          AT_ERROR(
              "Error in caffe2->c10 wrapper: Unsupported argument type ",
              argument.type()->str(),
              " in c10 operator schema");
        }
        */
    }
    
    #[inline] pub fn get_nontensor_argument_with_name_and_default_value<T>(
        &mut self, 
        name: &String, 
        default_value: &Option<IValue>) -> IValue 
    {
        todo!();
        /*
            if (default_value.has_value()) {
          return this->template GetSingleArgument<T>(name, default_value->to<T>());
        } else {
          TORCH_CHECK(
              this->template HasSingleArgumentOfType<T>(name),
              "Error in caffe2->c10 wrapper: Expected argument '",
              name,
              "' missing or wrong type.");
          return this->template GetSingleArgument<T>(name, 0);
        }
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build), not(c10_mobile)))]
#[inline] pub fn create_c10operator_wrapper<Context>(op_name: &OperatorName) 
-> fn(_u0: &OperatorDef, _u1: *mut Workspace) -> Box<OperatorStorage> {

    todo!();
    /*
        return [op_name](const OperatorDef& op_def, Workspace* ws) {
        auto op_handle =
            c10::Dispatcher::singleton().findSchema(op_name);
        AT_ASSERTM(
            op_handle.has_value(),
            "Tried to register c10 operator ",
            op_name.name,
            ".",
            op_name.overload_name,
            " with caffe2, but didn't find the c10 operator.");
        return std::make_unique<C10OperatorWrapper<Context>>(
            *op_handle, op_def, ws);
    */
}

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

define_registry![
    /*
    C10OperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
];

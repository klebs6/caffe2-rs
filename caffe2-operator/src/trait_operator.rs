crate::ix!();

/**
  | Operator is the class that you usually
  | want to derive, if your operator will
  | run on different devices. You should
  | then implement the RunOnDevice() function.
  |
  */
pub trait Operator {

    fn new_with_operator_def_and_workspace(
        operator_def: &OperatorDef, 
        ws: *mut Workspace) -> Self where Self: Sized
    {
        todo!();
        /*
            : OperatorStorage(operator_def, ws), context_(operator_def.device_option()) 

        // In the constructor, we switch to the device so that the child class
        // constructors will run on that device.
        context_.SwitchToDevice();
        */
    }
    
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    fn new_from_fn_schema_inputs_and_outputs(
        fn_schema: &FunctionSchema,
        inputs: Vec<IValue>,
        outputs: List<Tensor>) -> Self where Self: Sized
    {
        todo!();
        /*
            : OperatorStorage(fn_schema, std::move(inputs), std::move(outputs)) 
                  // In the constructor, we switch to the device so that the child class
                  // constructors will run on that device.
                  context_.SwitchToDevice();
        */
    }
    
    fn new_with_operator_def_and_workspace_base(
        operator_def: &OperatorDef, 
        ws: *mut Workspace) -> Self where Self: Sized
    {
        todo!();
        /*
            : operator_ws_(ws),
          operator_def_(std::make_shared<OperatorDef>(operator_def)),
          device_option_(
              operator_def.has_device_option() ? operator_def.device_option()
                                               : DeviceOption()),
    #if defined(EXPOSE_C2_OPS) || \
        !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
          newstyle_outputs_(),
    #endif
          input_size_(operator_def.input_size()),
          event_(std::make_unique<Event>(device_option_)) 



      static GlobalInitIsCalledGuard guard;
      inputs_.reserve(operator_def.input_size());
      for (const string& input_str : operator_def.input()) {
        auto* blob = ws->GetBlob(input_str);
        CAFFE_ENFORCE(
            blob != nullptr,
            "op ",
            operator_def.type(),
            ": Encountered a non-existing input blob: ",
            input_str);
        inputs_.push_back(blob);
      }

      GetOperatorLogger()(operator_def);

      outputs_.reserve(operator_def.output_size());
      for (const string& output_str : operator_def.output()) {
        outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
      }

      type_ = operator_def.type();
        */
    }
    
    /**
      | Notes: All outputs ivalues must be tensors.
      | Input ivalue list must start with all
      | tensors ("inputs" in caffe2 terminology),
      | followed by non-tensors ("arguments"
      | in caffe2 terminology).
      | 
      | Alternatively, inputs can be one tensor
      | list ivalue followed by non-tensors
      | to represent operators with a variable
      | number of inputs.
      |
      */
    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    fn new_from_fn_schema_inputs_and_outputs_base(
        fn_schema: &FunctionSchema,
        inputs: Vec<IValue>,
        outputs: List<Tensor>) -> Self where Self: Sized
    {
        todo!();
        /*
            : fn_schema_(make_unique<c10::FunctionSchema>(std::move(fn_schema))),
            newstyle_inputs_(std::move(inputs)),
            newstyle_outputs_(std::move(outputs)),
            input_size_(compute_input_size_(newstyle_inputs_)) 

                input_tensors_.resize(input_size_);
            output_tensors_.resize(newstyle_outputs_.size());
        */
    }
}

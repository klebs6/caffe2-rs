crate::ix!();

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

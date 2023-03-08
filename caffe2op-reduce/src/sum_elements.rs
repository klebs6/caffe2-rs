crate::ix!();

/**
  | Sums the elements of the input tensor.
  | Tensor type must be float32.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumElementsOp<T,Context> {
    storage:  OperatorStorage,
    context:  Context,
    average:  bool,
    scratch:  Tensor, // {Context::GetDeviceType()};
    phantom:  PhantomData<T>,
}

#[cfg(not(all(not(caffe2_is_xplat_build),not(c10_mobile))))]
impl<T,Context> SumElementsOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        operator_def: &OperatorDef,
        ws:           *mut Workspace,
        average:      bool) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws), average_(average)
        */
    }
}

#[cfg(all(not(caffe2_is_xplat_build),not(c10_mobile)))]
impl<T,Context> SumElementsOp<T,Context> {
    
    pub fn new(
    schema:  &FunctionSchema,
    inputs:  Vec<IValue>,
    outputs: Vec<*mut IValue>) -> Self {
    
        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)),
            average_(this->template GetSingleArgument<bool>("average", false))
        */
    }
    
    pub fn new_with_avg(
        schema:  &FunctionSchema,
        inputs:  Vec<IValue>,
        outputs: Vec<*mut IValue>,
        average: bool) -> Self {

        todo!();
        /*
            : Operator<Context>(schema, std::move(inputs), std::move(outputs)), average_(average)
        */
    }
}

impl<T,Context> SumElementsOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());

        T* data = sum->template mutable_data<T>();

        math::Sum<T, Context>(
            X.numel(), X.template data<T>(), data, &context_, &scratch_);
        if (average_ && X.numel() > 0) {
          math::Scale<float, T, Context>(
              1,
              static_cast<T>(1.) / X.numel(),
              sum->template data<T>(),
              data,
              &context_);
        }
        return true;
        */
    }
}

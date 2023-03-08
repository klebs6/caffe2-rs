crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumElementsGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    average:  bool,
    phantom: PhantomData<T>,
}

num_inputs!{SumElementsGradient, 2}

num_outputs!{SumElementsGradient, 1}

#[cfg(not(all(not(caffe2_is_xplat_build),not(c10_mobile))))]
impl<T,Context> SumElementsGradientOp<T,Context> {

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
impl<T,Context> SumElementsGradientOp<T,Context> {
    
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

impl<T, Context> SumElementsGradientOp<T, Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // TODO: T21635077 fix float-divide-by-zero undefined behavior
      auto& X = Input(0);
      Tensor sum_grad(Input(1), CPU);

      auto* dX = Output(0, X.sizes(), at::dtype<T>());
      DCHECK_EQ(sum_grad.numel(), 1);
      math::Set<T, Context>(
          dX->numel(),
          static_cast<T>(
              sum_grad.template data<T>()[0] * (average_ ? 1.0 / X.numel() : 1)),
          dX->template mutable_data<T>(),
          &context_);
      return true;
        */
    }
}

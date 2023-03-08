crate::ix!();

///-----------------------------
/// Sums the squares elements of the input tensor.
///
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_SIMPLE_CTOR_DTOR("SumSqrElementsOp")]
pub struct SumSqrElementsOp<Context> {
    storage: OperatorStorage,
    context: Context,
    scratch: Tensor, // {Context::GetDeviceType()};
}

impl<Context> SumSqrElementsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            bool average = this->template GetSingleArgument<bool>("average", false);
        auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
        math::SumSqr<T, Context>(
            X.numel(),
            X.template data<T>(),
            sum->template mutable_data<T>(),
            &context_,
            &scratch_);
        if (average && X.numel() > 0) {
          math::Scale<float, T, Context>(
              1,
              float(1.) / X.numel(),
              sum->template data<T>(),
              sum->template mutable_data<T>(),
              &context_);
        }
        return true;
        */
    }
}

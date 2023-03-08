crate::ix!();

/// Sums the integer elements of the input tensor.
///
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumElementsIntOp<T,Context> {
    storage: OperatorStorage,
    context: Context,

    scratch:  Tensor, // {Context::GetDeviceType()};
    phantom: PhantomData<T>,
}

impl<T,Context> SumElementsIntOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* sum = Output(0, vector<int64_t>(), at::dtype<T>());
        T* data = sum->template mutable_data<T>();
        math::Sum<T, Context>(
            X.numel(), X.template data<T>(), data, &context_, &scratch_);
        return true;
        */
    }
}
